from utils.regression_trainer import RegTrainer
import argparse
import torch
import numpy as np
import random
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
    parser.add_argument('--data-dir', default='datasets/UCF-Train-Val-Test')
    parser.add_argument('--save-dir', default='ckpt_ucf')
    parser.add_argument('--dataset', default='ucf', type=str)
    parser.add_argument('--crop-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-epoch', type=int, default=1000)
    parser.add_argument('--val-start', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--is-gray', type=bool, default=False)
    parser.add_argument('--downsample-ratio', type=int, default=8, help='downsample ratio')
    parser.add_argument('--use-background', type=bool, default=True, help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0, help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=1.0, help='background ratio')
    args = parser.parse_args()

    setup_seed(0)
    print('Training dataset:', args.dataset)
    print()
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()