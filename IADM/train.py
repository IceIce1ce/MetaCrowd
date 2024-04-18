from utils.regression_trainer import RegTrainer
import argparse
import torch
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='datasets/bayes-RGBT-CC', type=str)
    parser.add_argument('--save-dir', default='ckpt_rgbtcc', type=str)
    parser.add_argument('--dataset', default='RGBTCC', choices=['RGBTCC', 'ShanghaiTechRGBD'], type=str)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--resume', default=None, help='the path of resume training model')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--val-start', type=int, default=20, help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--downsample-ratio', type=int, default=8, help='downsample ratio')
    parser.add_argument('--use-background', type=bool, default=True, help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0, help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.15, help='background ratio')
    parser.add_argument('--type-model', type=str, default='bl', choices=['bl', 'csrnet'])
    args = parser.parse_args()

    setup_seed(0)
    print('Training dataset:', args.dataset)
    print('Training model:', args.type_model)
    print()
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()