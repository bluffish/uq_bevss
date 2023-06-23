import random
from time import time

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from nuscenes.map_expansion.map_api import NuScenesMap

from glob import glob
import torchvision


class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))

denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage(),
))


def gen_dx_bx(x_bound, y_bound, z_bound):
    dx = torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [x_bound, y_bound, z_bound]])

    return dx, bx, nx


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


def get_transformation_matrix(R, t, inv=False):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R if not inv else R.T
    pose[:3, -1] = t if not inv else R.T @ -t

    return pose


def get_pose(rotation, translation, inv=False, flat=False):
    if flat:
        yaw = Quaternion(rotation).yaw_pitch_roll[0]
        R = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix

    t = np.array(translation, dtype=np.float32)

    return get_transformation_matrix(R, t, inv=inv)


class NuscData(torch.utils.data.Dataset):
    def __init__(self,
                 nusc,
                 nusc_maps,
                 final_dim,
                 is_train=False,
                 H=900, W=1600,
                 resize_lim=(0.193, 0.225),
                 bot_pct_lim=(0.0, 0.22),
                 rot_lim=(-5.4, 5.4),
                 rand_flip=True,
                 ncams=5,
                 xbound=[-50.0, 50.0, 0.5],
                 ybound=[-50.0, 50.0, 0.5],
                 zbound=[-10.0, 10.0, 20.0],
                 dbound=[4.0, 45.0, 1.0],
                 ood=False,
                 flipped=True
                 ):

        self.ood = ood
        self.flipped = flipped

        self.grid_conf = {
            'xbound': xbound,
            'ybound': ybound,
            'zbound': zbound,
            'dbound': dbound,
        }
        self.data_aug_conf = {
            'resize_lim': resize_lim,
            'final_dim': final_dim,
            'rot_lim': rot_lim,
            'H': H, 'W': W,
            'rand_flip': rand_flip,
            'bot_pct_lim': bot_pct_lim,
            'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
            'Ncams': ncams,
        }

        # self.ood_classes_train = ['vehicle.bus.rigid', "vehicle.bus.bendy"]
        # self.ood_classes_val = ['vehicle.construction']
        # self.ood_classes_train = ["static_object.bicycle_rack"]
        self.ood_classes_val = []
        self.ood_classes_train = []
        self.all_ood = self.ood_classes_train + self.ood_classes_val

        if is_train:
            self.ood_labels = self.ood_classes_train
        else:
            self.ood_labels = self.ood_classes_train

        print(f"OOD labels: {self.ood_labels}")

        self.nusc = nusc
        self.nusc_maps = nusc_maps
        self.is_train = is_train
        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        self.v = 0
        self.r = 0
        self.l = 0
        self.b = 0

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        # self.fix_nuscenes_formatting()
        self.scene2map = {}
        for rec in nusc.scene:
            log = nusc.get('log', rec['log_token'])
            self.scene2map[rec['name']] = log['location']

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (
                        rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):

        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]
        ood = []
        id = []

        for rec in samples:
            if self.ood and len(ood) >= 64:
                break

            ego_pose = self.nusc.get('ego_pose', rec['data']['LIDAR_TOP'])

            ego_coord = ego_pose['translation']

            c = False

            for tok in rec['anns']:
                inst = self.nusc.get('sample_annotation', tok)

                box_coord = inst['translation']

                if max(abs(ego_coord[0] - box_coord[0]), abs(ego_coord[1] - box_coord[1])) > 100 or int(inst['visibility_token']) <= 2:
                    continue

                if inst['category_name'] in self.ood_labels:
                    ood.append(rec)

                if inst['category_name'] in self.all_ood:
                    c = True
                    break

            if not c:
                id.append(rec)

        # sort by scene, timestamp (only to make chronological viz easier)
        ood.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        id.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return ood if self.ood else id

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims

        crop_h = int(max(0, newH - fH))
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def parse_pose(self, record, *args, **kwargs):
        return get_pose(record['rotation'], record['translation'], *args, **kwargs)

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        extrins = []
        post_rots = []
        post_trans = []
        oods = []

        lidar_record = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        egolidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])
        world_from_egolidarflat = self.parse_pose(egolidar, flat=True)

        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])

            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])

            egocam = self.nusc.get('ego_pose', samp['ego_pose_token'])

            cam_from_egocam = self.parse_pose(sens, inv=True)
            egocam_from_world = self.parse_pose(egocam, inv=True)
            extrin = torch.tensor(cam_from_egocam @ egocam_from_world @ world_from_egolidarflat)

            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                       resize=resize,
                                                       resize_dims=resize_dims,
                                                       crop=crop,
                                                       flip=flip,
                                                       rotate=rotate,
                                                       )

            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(torch.tensor(np.array(img)).permute(2, 0, 1) / 255)
            intrins.append(intrin)
            extrins.append(extrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(extrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_label(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse

        vehicles = np.zeros((self.nx[0], self.nx[1]))
        ood = np.zeros((self.nx[0], self.nx[1]))

        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)

            if int(inst['visibility_token']) <= 2:
                continue

            if inst['category_name'] in self.ood_labels:
                box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
                box.translate(trans)
                box.rotate(rot)

                pts = box.bottom_corners()[:2].T
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
                ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(ood, [pts], 1.0)

            if not inst['category_name'] in self.all_ood and inst['category_name'].split('.')[0] == 'vehicle':
                box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
                box.translate(trans)
                box.rotate(rot)

                pts = box.bottom_corners()[:2].T
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
                ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(vehicles, [pts], 1.0)

        road, lane = get_map(rec, self.nusc_maps, self.nusc, self.scene2map, self.dx[:2], self.bx[:2])

        road[lane == 1] = 0
        road[vehicles == 1] = 0
        lane[vehicles == 1] = 0

        empty = np.ones((200, 200))
        empty[vehicles == 1] = 0
        empty[road == 1] = 0
        empty[lane == 1] = 0

        if not self.flipped:
            vehicles = cv2.flip(vehicles, 0)
            vehicles = cv2.flip(vehicles, 1)
            road = cv2.flip(road, 0)
            road = cv2.flip(road, 1)
            lane = cv2.flip(lane, 0)
            lane = cv2.flip(lane, 1)
            empty = cv2.flip(empty, 0)
            empty = cv2.flip(empty, 1)
            ood = cv2.flip(ood, 0)
            ood = cv2.flip(ood, 1)

        label = np.stack((vehicles, road, lane, empty))

        return torch.tensor(label).float(), torch.tensor(ood)

    def choose_cams(self):
        return self.data_aug_conf['cams']

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, extrins, post_rots, post_trans = self.get_image_data(rec, cams)
        label, ood = self.get_label(rec)
        return imgs, rots, trans, intrins, extrins, post_rots, post_trans, label, ood


def worker_rnd_init(x):
    np.random.seed(13 + x)


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                                       map_name=map_name) for map_name in [
                     "singapore-hollandvillage",
                     "singapore-queenstown",
                     "boston-seaport",
                     "singapore-onenorth",
                 ]}
    return nusc_maps


def get_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]
    center = np.array([egopose['translation'][0], egopose['translation'][1]])

    rota = quaternion_yaw(Quaternion(egopose['rotation'])) / np.pi * 180
    road = np.any(nusc_maps[map_name].get_map_mask((center[0], center[1], 100, 100), rota, ['road_segment', 'lane'],
                                                   canvas_size=(200, 200)), axis=0).T
    lane = np.any(
        nusc_maps[map_name].get_map_mask((center[0], center[1], 100, 100), rota, ['road_divider', 'lane_divider'],
                                         canvas_size=(200, 200)), axis=0).T

    return road.astype(np.uint8), lane.astype(np.uint8)


def compile_data(version, config, ood=False, augment_train=False, shuffle_train=True, seg=False):
    dataroot = os.path.join("../data", config['dataset'])
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=False)
    flipped = config['backbone'] == 'lss'
    dims = (128, 352)

    print(f"Flipped: {flipped}")
    print(f"Dims: {dims}")
    print(os.path.join(dataroot, version))
    
    nusc_maps = get_nusc_maps(os.path.join(dataroot, version))

    train_data = NuscData(nusc, nusc_maps, dims, is_train=True, ood=ood, flipped=flipped)
    val_data = NuscData(nusc, nusc_maps, dims, is_train=False, ood=ood, flipped=flipped)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'],
                                               shuffle=shuffle_train,
                                               num_workers=config['num_workers'],
                                               drop_last=True,
                                               pin_memory=True,
                                               worker_init_fn=worker_rnd_init)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'],
                                             shuffle=True,
                                             num_workers=config['num_workers'],
                                             drop_last=True,
                                             pin_memory=True,
                                             )

    return train_loader, val_loader
