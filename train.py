from sklearn.metrics import roc_auc_score, average_precision_score
from tensorboardX import SummaryWriter

from datasets.carla import compile_data as compile_data_carla
from datasets.nuscenes import compile_data as compile_data_nuscenes

from tools.utils import *
import torch
import torch.nn as nn
import argparse
import yaml
from tqdm import tqdm
import random
import warnings
from torch.profiler import profile, record_function, ProfilerActivity

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.enabled = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.multiprocessing.set_sharing_strategy('file_system')
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def get_val(model, val_loader, device, loss_fn, activation, num_classes):
    total_loss = 0.0
    iou = [0.0] * num_classes

    y_true = []
    y_score = []
    c = 0

    if config['type'] == 'dropout':
        model.module.tests = 10
        model.module.train()

    with torch.no_grad():
        for (imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels, ood) in tqdm(val_loader):
            imgs, s_labels = parse(imgs, gt)

            preds, s_preds = model(imgs, rots, trans, intrins, extrins, post_rots, post_trans)

            labels = labels.to(device)
            s_labels = s_labels.to(device)

            if config['type'] == 'enn' or config['type'] == 'postnet':
                uncertainty = dissonance(preds).cpu()
            else:
                uncertainty = entropy(preds).cpu()

            loss = loss_fn(preds, labels, s_preds, s_labels)
            preds = activation(preds)

            total_loss += loss * preds.shape[0]
            intersection, union = get_iou(preds, labels)

            for cl in range(0, num_classes):
                iou[cl] += 1 if union[0] == 0 else intersection[cl] / union[cl] * preds.shape[0]

            if c <= 64:
                pmax = torch.argmax(preds, dim=1).cpu()
                lmax = torch.argmax(labels, dim=1).cpu()

                u = uncertainty.ravel()

                misclassified = pmax != lmax

                y_true += misclassified.ravel()
                y_score += u

                c += preds.shape[0]
            else:
                if config['type'] == 'dropout':
                    model.module.tests = 1

    iou = [i / len(val_loader.dataset) for i in iou]

    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)

    return total_loss / len(val_loader.dataset), iou, auroc, aupr


def train():
    device = torch.device('cpu') if len(config['gpus']) < 0 else torch.device(f'cuda:{config["gpus"][0]}')
    num_classes, classes = 4, ["vehicle", "road", "lane", "background"]

    compile_data = compile_data_carla if config['dataset'] == 'carla' else compile_data_nuscenes
    train_loader, val_loader = compile_data("trainval", config, shuffle_train=True, seg=True)

    class_proportions = {
        "nuscenes": [.015, .2, .05, .735],
        "carla": [0.0141, 0.3585, 0.02081, 0.6064]
    }

    activation, loss_fn, model = get_model(config['type'], config['backbone'], num_classes, device, use_seg=config['seg'])

    if "postnet" in config['type']:
        if config['backbone'] == 'lss':
            model.bevencode.p_c = torch.tensor(class_proportions[config['dataset']])
        else:
            model.p_c = torch.tensor(class_proportions[config['dataset']])

    model = nn.DataParallel(model, device_ids=config['gpus']).to(device).train()

    if "pretrained" in config:
        print(f"Loading pretrained weights from {config['pretrained']}")
        model.load_state_dict(torch.load(config["pretrained"]), strict=False)

    if config['backbone'] == 'lss':
        opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = None
        print("Using Adam")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, div_factor=10, pct_start=.3, final_div_factor=10,
                                                        max_lr=config['learning_rate'], total_steps=config['num_steps'])
        print("Using AdamW and OneCycleLR")

    os.makedirs(config['logdir'], exist_ok=True)

    print("--------------------------------------------------")
    print(f"Starting training on {config['type']} model with {config['backbone']} backbone")
    print(f"Using GPUS: {config['gpus']}")
    print("Training using CARLA")
    print(f"Train loader: {len(train_loader.dataset)}")
    print(f"Val loader: {len(val_loader.dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Output directory: {config['logdir']} ")
    if config['seg']:
        print("Using segmentation")
        os.makedirs(os.path.join(config['logdir'], 'seg'), exist_ok=True)
    print("--------------------------------------------------")

    writer = SummaryWriter(logdir=config['logdir'])

    best_iou, best_auroc, best_aupr = 0, 0, 0

    step = 0
    epoch = 1

    while True:
        for batchi, (imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels, ood) in enumerate(
                train_loader):
            t0 = time()
            opt.zero_grad(set_to_none=True)
            imgs, s_labels = parse(imgs, gt)

            preds, s_preds = model(imgs, rots, trans, intrins, extrins, post_rots, post_trans)
            print(preds.shape)
            labels = labels.to(device)
            s_labels = s_labels.to(device)

            loss = loss_fn(preds, labels, s_preds, s_labels)

            preds = activation(preds)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            step += 1
            t1 = time()

            if scheduler is not None:
                scheduler.step()

            if step % 10 == 0:
                print(step, loss.item())

                writer.add_scalar('train/step_time', t1 - t0, step)
                writer.add_scalar('train/loss', loss, step)
                save_pred(preds, labels, config['logdir'])

                if s_preds is not None:
                    save_pred(activation(s_preds), s_labels, os.path.join(config['logdir'], 'seg'))

            if step % 50 == 0:
                intersection, union = get_iou(preds, labels)
                iou = [intersection[i] / union[i] for i in range(0, num_classes)]

                print(step, "iou: ", iou)

                for i in range(0, num_classes):
                    writer.add_scalar(f'train/{classes[i]}_iou', iou[i], step)

                writer.add_scalar('train/epoch', epoch, step)

            if step % config['val_step'] == 0:
                model.eval()
                print("Running EVAL...")
                val_loss, val_iou, auroc, aupr = get_val(model, val_loader, device, loss_fn, activation, num_classes)
                print(f"VAL loss: {val_loss}, iou: {val_iou}, auroc {auroc}, aupr {aupr}")

                save_path = os.path.join(config['logdir'], f"model{step}.pt")
                print(f"Saving Model: {save_path}")
                torch.save(model.state_dict(), save_path)

                if sum(val_iou) / len(val_iou) >= best_iou:
                    best_iou = sum(val_iou) / len(val_iou)
                    print(f"New best IOU model found. iou: {val_iou}")
                    torch.save(model.state_dict(), os.path.join(config['logdir'], "best_iou.pt"))
                if auroc >= best_auroc:
                    best_auroc = auroc
                    print(f"New best AUROC model found. iou: {auroc}")
                    torch.save(model.state_dict(), os.path.join(config['logdir'], "best_auroc.pt"))
                if aupr >= best_aupr:
                    best_aupr = aupr
                    print(f"New best AUPR model found. iou: {aupr}")
                    torch.save(model.state_dict(), os.path.join(config['logdir'], "best_aupr.pt"))

                model.train()

                writer.add_scalar('val/loss', val_loss, step)
                writer.add_scalar('val/auroc', auroc, step)
                writer.add_scalar('val/aupr', aupr, step)

                for i in range(0, num_classes):
                    writer.add_scalar(f'val/{classes[i]}_iou', val_iou[i], step)

            if step == config['num_steps']:
                return

        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False)
    parser.add_argument('-l', '--logdir', required=False)
    parser.add_argument('-b', '--batch_size', required=False)
    parser.add_argument('--gt', default=False, action='store_true')

    args = parser.parse_args()
    gt = args.gt

    print(f"Using config {args.config}")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if args.gpus is not None:
        config['gpus'] = [int(i) for i in args.gpus]
    if args.logdir is not None:
        config['logdir'] = args.logdir
    if args.batch_size is not None:
        config['batch_size'] = int(args.batch_size)

    train()
