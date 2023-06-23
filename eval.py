import torch

from datasets.nuscenes import compile_data as compile_data_nuscenes
from datasets.carla import compile_data as compile_data_carla
from tqdm import tqdm

from tools.utils import *
from tools.uncertainty import *
from tools.loss import *

import argparse
import yaml

import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn

sns.set_style('white')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.25,
                rc={"lines.linewidth": 2.5})

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.multiprocessing.set_sharing_strategy('file_system')
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def eval(config, is_ood=False, gt=False, save=False):
    name = f"{config['backbone']}_{config['type']}"
    device = torch.device('cpu') if len(config['gpus']) < 0 else torch.device(f'cuda:{config["gpus"][0]}')
    num_classes, classes = 4, ["vehicle", "road", "lane", "background"]

    compile_data = compile_data_carla if config['dataset'] == 'carla' else compile_data_nuscenes
    train_loader, val_loader = compile_data("mini" if is_ood else "mini", config, shuffle_train=True, ood=is_ood, seg=True)

    class_proportions = {
        "nuscenes": [.015, .2, .05, .735],
        "carla": [0.0141, 0.3585, 0.02081, 0.6064]
    }

    activation, loss_fn, model = get_model(config['type'], config['backbone'], num_classes, device, use_seg=config['seg'])

    if config['type'] == 'baseline' or config['type'] == 'dropout' or config['type'] == 'ensemble':
        uncertainty_function = entropy
    elif config['type'] == 'enn' or config['type'] == 'postnet':
        if is_ood:
            uncertainty_function = vacuity
        else:
            uncertainty_function = aleatoric
            # uncertainty_function = dissonance

    if "postnet" in config['type']:
        if config['backbone'] == 'lss':
            model.bevencode.p_c = torch.tensor(class_proportions[config['dataset']])
        else:
            model.p_c = torch.tensor(class_proportions[config['dataset']])

    model = nn.DataParallel(model, device_ids=config['gpus']).to(device).eval()
    model.load_state_dict(torch.load(config['model_path']), strict=False)

    if config['type'] == 'dropout':
        model.module.tests = 10
        model.module.train()


    print("--------------------------------------------------")
    print(f"Starting eval on {name}")
    print(f"Using GPUs: {config['gpus']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Eval using {config['dataset']} ")
    print("Val loader: ", len(val_loader.dataset))
    print(f"Output directory: {config['logdir']} ")
    print(f"OOD: {is_ood}")
    print(f"Model path: {config['model_path']}")
    if config['seg']: print("Using segmentation")
    print("--------------------------------------------------")

    os.makedirs(config['logdir'], exist_ok=True)

    predictions = torch.zeros((len(val_loader.dataset), 4, 200, 200))
    ground_truths = torch.zeros((len(val_loader.dataset), 4, 200, 200))
    uncertainty_scores = torch.zeros((len(val_loader.dataset), 200, 200))
    uncertainty_labels = torch.zeros((len(val_loader.dataset), 200, 200))

    total = 0

    with torch.no_grad():
        for i, (imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels, ood) in enumerate(tqdm(val_loader)):
            range_i = slice(i * config['batch_size'], i * config['batch_size'] + config['batch_size'])

            imgs, s_labels = parse(imgs, gt)

            t = time()
            preds, _ = model(imgs, rots, trans, intrins, extrins, post_rots, post_trans)
            total += time()-t

            uncertainty = uncertainty_function(preds).cpu()

            preds = activation(preds)
            labels = labels.to(device)

            predictions[range_i] = preds
            ground_truths[range_i] = labels
            uncertainty_scores[range_i] = torch.squeeze(uncertainty, dim=1)

            cv2.imwrite(os.path.join(config['logdir'], "uncertainty_map.png"),
                       cv2.cvtColor((plt.cm.jet(uncertainty[0][0])*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            save_pred(preds, labels, config['logdir'])

            if is_ood:
                cv2.imwrite(os.path.join(config['logdir'], f"ood.png"),
                           ood[0].cpu().numpy()*255)

                uncertainty_labels[range_i] = ood
            else:
                pmax = torch.argmax(preds, dim=1).cpu()
                lmax = torch.argmax(labels, dim=1).cpu()
                misclassified = pmax != lmax

                cv2.imwrite(os.path.join(config['logdir'], "misclassified.png"), misclassified[0].cpu().numpy()*255)

                uncertainty_labels[range_i] = misclassified

    intersect, union = get_iou(predictions, ground_truths)

    iou = [intersect[i]/union[i] for i in range(len(intersect))]

    print(f'iou: {iou}')
    print(f'Average time per forward pass {1000*total/len(val_loader.dataset):.2f}ms')

    if save:
        torch.save(uncertainty_scores, os.path.join(config['logdir'], "uncertainty_scores.pt"))
        torch.save(uncertainty_labels, os.path.join(config['logdir'], "uncertainty_labels.pt"))
        torch.save(predictions, os.path.join(config['logdir'], "predictions.pt"))
        torch.save(ground_truths, os.path.join(config['logdir'], "ground_truths.pt"))

    return predictions, ground_truths, uncertainty_scores, uncertainty_labels, iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False)
    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-b', '--batch_size', required=False)
    parser.add_argument('-p', '--model_path', required=False)
    parser.add_argument('-l', '--logdir', required=False)
    parser.add_argument('-s', '--save', default=False, action='store_true')
    parser.add_argument('--gt', default=False, action='store_true')
    parser.add_argument('-m', '--metric', default="rocpr", required=False)

    args = parser.parse_args()
    gt = args.gt

    print(f"Using config {args.config}")

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if args.gpus is not None:
        config['gpus'] = [int(i) for i in args.gpus]
    if args.batch_size is not None:
        config['batch_size'] = int(args.batch_size)
    if args.model_path is not None:
        config['model_path'] = args.model_path
    if args.logdir is not None:
        config['logdir'] = args.logdir

    is_ood = args.ood
    save = args.save
    metric = args.metric
    name = f"{config['backbone']}_{config['type']}"

    predictions, ground_truths, uncertainty_scores, uncertainty_labels, iou = eval(config, is_ood=is_ood, gt=gt, save=save)
    print(torch.mean(uncertainty_scores))
    print(calculate_pavpu(uncertainty_scores, uncertainty_labels, uncertainty_threshold=torch.mean(uncertainty_scores)))
    if metric == 'patch':
        pavpu, agc, ugi, thresholds, au_pavpu, au_agc, au_ugi = patch_metrics(uncertainty_scores, uncertainty_labels)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        ax1.plot(thresholds, agc, 'g.-', label=f"AU-p(accurate|certain): {au_agc:.3f}")
        ax1.set_xlabel('Uncertainty Percentiles')
        ax1.set_ylabel('p(accurate|certain)')
        ax1.legend(frameon=True)

        ax2.plot(thresholds, ugi, 'r.-', label=f"AU-p(uncertain|inaccurate): {au_ugi:.3f}")
        ax2.set_xlabel('Uncertainty Percentiles')
        ax2.set_ylabel('p(uncertain|inaccurate)')
        ax2.legend(frameon=True)

        ax3.plot(thresholds, pavpu,'b.-', label=f"AU-PAvPU: {au_pavpu:.3f}")
        ax3.set_xlabel('Uncertainty Percentiles')
        ax3.set_ylabel('PAVPU')
        ax3.legend(frameon=True)

        fig.suptitle(f"{'OOD' if is_ood else 'Misclassification'} - {name}")

        save_path = os.path.join(config['logdir'], f"patch_{'o' if is_ood else 'm'}_{name}.png")

        print(f"AU-PAvPU: {au_pavpu:.3f}, AU-p(accurate|certain): {au_agc:.3f}, AU-P(uncertain|inaccurate): {au_ugi:.3f}")
    elif metric == "rocpr":
        fpr, tpr, rec, pr, auroc, aupr, no_skill = roc_pr(uncertainty_scores, uncertainty_labels)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(fpr, tpr, 'b-', label=f'AUROC - {auroc:.3f}')
        ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill - 0.500')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend(frameon=True)

        ax2.plot(rec, pr, 'r-', label=f'AUPR - {aupr:.3f}')
        ax2.plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray', label=f'No Skill - {no_skill:.3f}')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend(frameon=True)

        fig.suptitle(f"{'OOD' if is_ood else 'Misclassification'} - {name}")

        save_path = os.path.join(config['logdir'], f"rocpr_{'o' if is_ood else 'm'}_{name}.png")

        print(f"AUROC: {auroc:.3f} AUPR: {aupr:.3f}")
    else:
        raise ValueError("Please pick a valid metric.")

    fig.savefig(save_path)
