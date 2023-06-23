import yaml
import argparse
import os
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

from eval import eval

classes = ["vehicle", "road", "lane", "background"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument('-g', '--gpus', nargs='+', required=False)
    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-p', '--model_path', required=True)
    parser.add_argument('-s', '--steps', required=False)
    parser.add_argument('-c', '--cl', required=False)
    parser.add_argument('-l', '--logdir', required=True)

    args = parser.parse_args()

    is_ood = False

    os.makedirs(args.logdir, exist_ok=True)

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if args.steps is not None:
        steps = int(args.steps)
    else: steps = 15000

    if args.cl is not None:
        c = int(args.cl)
    else:
        c = 0

    print(f"Using config {args.config}, with class {c}")

    writer = SummaryWriter(logdir=config['logdir'])

    if args.gpus is not None:
        config['gpus'] = [int(i) for i in args.gpus]
    if args.ood is not None:
        is_ood = args.ood

    steps_values = range(1000, steps+1, 1000)

    for i in steps_values:
        config['model_path'] = os.path.join(args.model_path, f"model{i}.pt")
        print(config['model_path'])
        pavpus, agcs, ugis, thresholds, pavpu_score, agcs_score, ugis_score = eval(config, plot=False, is_ood=is_ood)
        writer.add_scalar('metrics/pavpu', pavpu_score, i)
        writer.add_scalar('metrics/p(a|c)', agcs_score, i)
        writer.add_scalar('metrics/p(u|i)', ugis_score, i)
