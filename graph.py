import argparse, yaml
from eval import eval
import seaborn as sns
import matplotlib.pyplot as plt
from tools.utils import *
import os

sns.set_style('white')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.25,
                rc={"lines.linewidth": 2.5})


def replace_last_section(path, new_section):
    # Split the path into a head and tail
    # The head contains all the directory names and the tail is the last section
    head, tail = os.path.split(path)

    # Join the head with the new section
    new_path = os.path.join(head, new_section)

    return new_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metric")
    parser.add_argument("set")

    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-l', '--logdir', required=False)
    parser.add_argument('-g', '--gpus', nargs='+', required=False)

    args = parser.parse_args()

    with open(args.set, 'r') as file:
        set = yaml.safe_load(file)

    set_name = args.set.split('.')[-2].split('/')[-1]
    names = list(set.keys())

    if args.logdir is not None:
        logdir = args.logdir
    else:
        logdir = './outputs'

    os.makedirs(logdir, exist_ok=True)

    metric = args.metric
    is_ood = args.ood

    scale = 1.5

    if metric == 'patch':
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(left=0.05, right=0.95)
        ax1.set_xlabel('Uncertainty Percentiles')
        ax1.set_ylabel('p(accurate|certain)')

        ax2.set_xlabel('Uncertainty Percentiles')
        ax2.set_ylabel('p(uncertain|inaccurate)')

        ax3.set_xlabel('Uncertainty Percentiles')
        ax3.set_ylabel('PAVPU')
    elif metric == 'rocpr':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12*scale, 6*scale))

        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
    elif metric == 'all':
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(left=0.05, right=0.95)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')

        ax3.set_xlabel('Uncertainty Percentiles')
        ax3.set_ylabel('PAVPU')
    else:
        raise ValueError("Please pick a valid metric.")

    no_skill_total = 0

    for name in names:
        with open(set[name]['config'], 'r') as file:
            config = yaml.safe_load(file)
            config['model_path'] = set[name]['path']
            if args.gpus is not None:
                config['gpus'] = [int(i) for i in args.gpus]

        gt = set[name]['gt'] if 'gt' in set[name] else False

        predictions, ground_truths, uncertainty_scores, uncertainty_labels, iou = eval(config=config, is_ood=is_ood, gt=gt)

        label = set[name]['label'] if 'label' in set[name] else name

        if metric == 'patch':
            pavpu, agc, ugi, thresholds, au_pavpu, au_agc, au_ugi = patch_metrics(uncertainty_scores,
                                                                                  uncertainty_labels)

            ax1.plot(thresholds, agc, '.-', label=f"{label}: {au_agc:.3f}")
            ax2.plot(thresholds, ugi, '.-', label=f"{label}: {au_ugi:.3f}")
            ax3.plot(thresholds, pavpu, '.-', label=f"{label}: {au_pavpu:.3f}")

            print(f"AU-PAvPU - {au_pavpu:.3f}, AU-p(accurate|certain) - {au_agc:.3f}, AU-P(uncertain|inaccurate) - {au_ugi:.3f}")
        elif metric == "rocpr":
            fpr, tpr, rec, pr, auroc, aupr, no_skill = roc_pr(uncertainty_scores, uncertainty_labels)

            ax1.plot(fpr, tpr, '-', label=f'{label}: {auroc:.3f}')
            ax2.plot(rec, pr, '-', label=f'{label}: {aupr:.3f}')

            no_skill_total += no_skill

            print(f"AUROC: {auroc:.3f}, AUPR: {aupr:.3f}")
        elif metric == "all":
            fpr, tpr, rec, pr, auroc, aupr, no_skill = roc_pr(uncertainty_scores, uncertainty_labels)
            config['model_path'] = replace_last_section(set[name]['path'], 'best_iou.pt')

            predictions, ground_truths, uncertainty_scores, uncertainty_labels, iou = eval(config=config, is_ood=is_ood,
                                                                                               gt=gt)

            pavpu, agc, ugi, thresholds, au_pavpu, au_agc, au_ugi = patch_metrics(uncertainty_scores,
                                                                                  uncertainty_labels)

            ax1.plot(fpr, tpr, '-', label=f'{label}: {auroc:.3f}')
            ax2.plot(rec, pr, '-', label=f'{label}: {aupr:.3f}')
            ax3.plot(thresholds, pavpu, '.-', label=f"{label}: {au_pavpu:.3f}")

            print(f"AUROC: {auroc:.3f}, AUPR: {aupr:.3f}, AU-PAvPU - {au_pavpu:.3f}")

    if metric == 'all':
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])
        ax2.set_xlim([-0.05, 1.05])
        ax2.set_ylim([-0.05, 1.05])
        ax3.set_xlim([-0.05, 1.05])
        ax3.set_ylim([-0.05, 1.05])

        ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill - 0.500')
        ax2.plot([0, 1], [no_skill_total / len(names), no_skill_total / len(names)], linestyle='--', color='gray', label=f'No Skill: {no_skill:.3f}')

        ax1.legend(frameon=True)
        ax2.legend(frameon=True)
        ax3.legend(frameon=True)
    elif metric == 'patch':
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])
        ax2.set_xlim([-0.05, 1.05])
        ax2.set_ylim([-0.05, 1.05])
        ax3.set_xlim([-0.05, 1.05])
        ax3.set_ylim([-0.05, 1.05])
        ax1.legend(frameon=True)
        ax2.legend(frameon=True)
        ax3.legend(frameon=True)

    elif metric == 'rocpr':
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])
        ax2.set_xlim([-0.05, 1.05])
        ax2.set_ylim([-0.05, 1.05])
        ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill - 0.500')
        ax2.plot([0, 1], [no_skill_total / len(names), no_skill_total / len(names)], linestyle='--', color='gray', label=f'No Skill: {no_skill:.3f}')

        ax1.legend(frameon=True)
        ax2.legend(frameon=True)

    # fig.suptitle(f"{'OOD' if is_ood else 'Misclassification'} - {set_name}")
    fig.suptitle(f"{'OOD' if is_ood else 'Misclassification'}")
    save_path = f"{logdir}/{metric}_{'o' if is_ood else 'm'}_{set_name}.png"
    fig.savefig(save_path)
