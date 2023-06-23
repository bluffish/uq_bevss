import matplotlib.pyplot as plt
import yaml
import argparse
import seaborn as sns
from eval import eval
from tools.utils import *
from datasets.carla import compile_data
from matplotlib import rc

from sklearn.metrics import *
import matplotlib
from time import time

sns.set_style('white')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

params = {'font.size': 3}
plt.rcParams.update(params)

class_proportions = {
    "nuscenes": [.015, .2, .05, .735],
    "carla": [0.0141, 0.3585, 0.02081, 0.6064]
}

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),))


def graph():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-g', '--gpus', nargs='+', required=False)
    parser.add_argument('--gt', default=False, action='store_true')
    parser.add_argument("config")
    args = parser.parse_args()

    is_ood = args.ood
    gt = args.gt
    num_classes = 4
    device = int(args.gpus[0])
    print(device)
    gpus = [int(i) for i in args.gpus]

    if is_ood:
        print("USING OOD")

    set_name = args.config.split('.')[-2].split('/')[-1]

    with open(args.config, 'r') as file:
        models = yaml.safe_load(file)

    samples = 5
    fig, axs = plt.subplots(samples, len(models)*2 + 1, figsize=(2*len(models)*2, 2*samples))

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    letters = ['(b)', '(c)', '(d)', '(e)', '(f)',]
    for i, name in enumerate(models.keys()):
        with open(models[name]['config'], 'r') as file:
            config = yaml.safe_load(file)

        config['model_path'] = models[name]['path']
        config['batch_size'] = 1
        config['num_workers'] = 1
        axs[-1, i*2+1].set_xlabel(f"                     {letters[i]} {models[name]['label'].split(' ')[-1]}")

        activation, loss_fn, model = get_model(config['type'], config['backbone'], num_classes, device,
                                               use_seg=config['seg'])
        train_loader, val_loader = compile_data("mini", config, shuffle_train=True, ood=False, cvp=None)

        if "postnet" in config['type']:
            if config['backbone'] == 'lss':
                model.bevencode.p_c = torch.tensor(class_proportions[config['dataset']])
            else:
                model.p_c = torch.tensor(class_proportions[config['dataset']])

        model = nn.DataParallel(model, device_ids=gpus).to(device).eval()
        model.load_state_dict(torch.load(config['model_path']))

        if config['type'] == 'baseline' or config['type'] == 'dropout' or config['type'] == 'ensemble':
            uncertainty_function = entropy
        elif config['type'] == 'enn' or config['type'] == 'postnet':
            uncertainty_function = aleatoric

        if config['type'] == 'dropout':
            model.module.tests = 20
            model.module.train()
        
        k = 0
        for (imgs, rots, trans, intrins, extrins, post_rots, post_trans, labels, ood) in val_loader:

            if torch.sum(labels[0,0]) == 0: continue
            imgs, s_labels = parse(imgs, gt)

            preds, _ = model(imgs, rots, trans, intrins, extrins, post_rots, post_trans)
            uncertainty = uncertainty_function(preds).cpu()
            preds = activation(preds).detach().cpu()

            axs[k, 0].imshow(map(labels[0].permute(1, 2, 0))/255)
            axs[k, i*2+1].imshow(plt.cm.jet(uncertainty[0][0].detach()))
            axs[k, i*2+2].imshow(~(preds.argmax(1).cpu() == labels.argmax(1).cpu())[0] * 255)

            k += 1
            
            if k >= samples:
                break

        model.cpu()

    axs[-1, 0].set_xlabel(f"(a) Ground truth")

    plt.tight_layout()

    fig.savefig(f"viz_{set_name}.png", bbox_inches='tight', dpi=300)


graph()
