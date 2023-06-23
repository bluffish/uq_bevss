from transforms3d.euler import euler2mat
from PIL import Image

import numpy as np

import torchvision
import math
import torch
import json
import os
import cv2

def get_camera_info(translation, rotation, sensor_options):
    roll = math.radians(rotation[2] - 90)
    pitch = -math.radians(rotation[1])
    yaw = -math.radians(rotation[0])
    rotation_matrix = euler2mat(roll, pitch, yaw)

    calibration = np.identity(3)
    calibration[0, 2] = sensor_options['image_size_x'] / 2.0
    calibration[1, 2] = sensor_options['image_size_y'] / 2.0
    calibration[0, 0] = calibration[1, 1] = sensor_options['image_size_x'] / (
            2.0 * np.tan(sensor_options['fov'] * np.pi / 360.0))

    return torch.tensor(rotation_matrix), torch.tensor(translation), torch.tensor(calibration)


def mask(img, target):
    m = np.all(img == target, axis=2).astype(int)
    return m


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


class CarlaDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path,
            H=128, W=352,
            resize_lim=(0.193, 0.225),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            rand_flip=True,
            is_train=False,
            ood=False,
            seg=False,
    ):
        self.ood = ood

        self.data_path = data_path
        self.type = type
        self.vehicles = len(os.listdir(os.path.join(self.data_path, 'agents')))
        self.ticks = len(os.listdir(os.path.join(self.data_path, 'agents/0/back_camera')))
        self.length = self.ticks*self.vehicles
        self.normalize_img = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
        ))

        with open(os.path.join(self.data_path, 'agents/0/sensors.json'), 'r') as f:
            self.sensors_info = json.load(f)

        self.H = H
        self.W = W
        self.resize_lim = resize_lim
        self.final_dim = final_dim
        self.bot_pct_lim = bot_pct_lim
        self.rot_lim = rot_lim
        self.rand_flip = rand_flip
        self.is_train = is_train

        self.offset = 0
        self.seg = seg

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        agent_number = math.floor(idx / self.ticks)
        agent_path = os.path.join(self.data_path, f"agents/{agent_number}/")
        idx = (idx + self.offset) % self.ticks

        imgs = []
        img_segs = []
        depths = []
        rots = []
        trans = []
        intrins = []
        extrins = []

        post_rots = []
        post_trans = []

        label_r = Image.open(os.path.join(agent_path + "birds_view_semantic_camera", str(idx) + '.png'))
        label = np.array(label_r)
        label_r.close()

        empty = np.ones((200, 200))

        road = mask(label, (128, 64, 128))
        lane = mask(label, (157, 234, 50))
        vehicles = mask(label, (0, 0, 142))
        ood = torch.tensor(mask(label, (0, 0, 142)))

        empty[vehicles == 1] = 0
        empty[road == 1] = 0
        empty[lane == 1] = 0
        label = np.stack((vehicles, road, lane, empty))

        label = torch.tensor(label)

        for sensor_name, sensor_info in self.sensors_info['sensors'].items():
            if sensor_info["sensor_type"] == "sensor.camera.rgb" and sensor_name != "birds_view_camera":
                image = Image.open(os.path.join(agent_path + sensor_name, str(idx) + '.png'))
                image_seg = Image.open(os.path.join(agent_path + sensor_name + "_semantic", str(idx) + '.png'))
                # image_pid = Image.open(os.path.join(agent_path + sensor_name + "_tripid2", str(idx) + '.png'))

                depth_p = Image.open(os.path.join(agent_path + sensor_name + "_depth", str(idx) + '.png'))

                tran = sensor_info["transform"]["location"]
                rot = sensor_info["transform"]["rotation"]
                sensor_options = sensor_info["sensor_options"]

                rot, tran, intrin = get_camera_info(tran, rot, sensor_options)
                resize, resize_dims, crop, flip, rotate = self.sample_augmentation()

                extrin = torch.eye(4)

                extrin[:3, :3] = rot
                extrin[:3, -1] = tran

                extrins.append(extrin)

                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                img_seg, _, _ = img_transform(image_seg, post_rot, post_tran,
                                              resize=resize,
                                              resize_dims=resize_dims,
                                              crop=crop,
                                              flip=flip,
                                              rotate=rotate, )

                depth, _, _ = img_transform(depth_p, post_rot, post_tran,
                                            resize=resize,
                                            resize_dims=resize_dims,
                                            crop=crop,
                                            flip=flip,
                                            rotate=rotate, )

                img, post_rot2, post_tran2 = img_transform(image, post_rot, post_tran,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate, )

                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                depth = np.array(depth)
                depth = depth[:, :, 0] + depth[:, :, 1] * 256 + depth[:, :, 2] * 256 * 256
                depth = depth / (256 * 256 * 256 - 1)
                depth = depth * 1000

                if np.max(depth) > 0:
                    depth = depth / np.max(depth)

                depth = torch.tensor(depth)[None, :, :]

                ni = self.normalize_img(img)

                if self.seg:
                    img_seg = np.array(img_seg)

                    vehicles = mask(img_seg, (0, 0, 142))[None]
                    road = mask(img_seg, (128, 64, 128))[None]
                    lane = mask(img_seg, (157, 234, 50))[None]

                    img_seg = np.concatenate((vehicles, road, lane))
                    img_seg = torch.tensor(img_seg)

                    ni = torch.cat((ni, img_seg), 0)

                imgs.append(ni)
                img_segs.append(img_seg)
                depths.append(depth)

                intrins.append(intrin)
                rots.append(rot)
                trans.append(tran)
                post_rots.append(post_rot)
                post_trans.append(post_tran)

                image.close()
                image_seg.close()
                depth_p.close()

        return (torch.stack(imgs).float(),
                torch.stack(rots).float(), torch.stack(trans).float(),
                torch.stack(intrins).float(), torch.stack(extrins).float(), torch.stack(post_rots).float(), torch.stack(post_trans).float(),
                label.float(), ood.float())

    def sample_augmentation(self):
        H, W = self.H, self.W
        fH, fW = self.final_dim

        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate


def compile_data(version, config, ood=False, shuffle_train=True, seg=False, cvp=None):
    dataroot = os.path.join("../data", config['dataset'])
    train_dataset = CarlaDataset(os.path.join(dataroot, "train/"), seg=seg)
    import random

    torch.manual_seed(0)

    if ood:
        val_dataset = CarlaDataset(os.path.join(dataroot, "val_ood/"), ood=ood, seg=seg)
    else:
        val_dataset = CarlaDataset(os.path.join(dataroot, "val/" if cvp is None else cvp), ood=ood, seg=seg)

    if version == "mini":
        val_dataset.length = 64

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=shuffle_train,
                                               num_workers=config['num_workers'],
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config['batch_size'],
                                             shuffle=True,
                                             num_workers=config['num_workers'],
                                             drop_last=True)

    return train_loader, val_loader