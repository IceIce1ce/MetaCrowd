import torch
import os
import argparse
from datasets.crowd import Crowd_RGBTCC
from models.BL_IADM import BL_IADM
from models.CSRNet_IADM import CSRNet_IADM
from utils.evaluation import eval_game, eval_relative
import warnings
warnings.filterwarnings('ignore')

def test(args):
    if args.dataset == 'RGBTCC':
        datasets = Crowd_RGBTCC(os.path.join(args.data_dir, 'test'), method='test')
        dataloader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    # load model
    if args.type_model == 'bl':
        model = BL_IADM()
    elif args.type_model == 'csrnet':
        model = CSRNet_IADM()
    model.cuda()
    checkpoint = torch.load(args.save_dir + '/best_model.pth', map_location='cuda')
    model.load_state_dict(checkpoint)
    model.eval()
    # testing
    game = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]
    total_relative_error = 0
    for inputs, target, name in dataloader:
        if type(inputs) == list:
            inputs[0] = inputs[0].cuda()
            inputs[1] = inputs[1].cuda()
        else:
            inputs = inputs.cuda()
        if type(inputs) == list:
            assert inputs[0].size(0) == 1
        else:
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            for L in range(4):
                abs_error, square_error = eval_game(outputs, target, L)
                game[L] += abs_error
                mse[L] += square_error
            relative_error = eval_relative(outputs, target)
            total_relative_error += relative_error
    N = len(dataloader)
    game = [m / N for m in game]
    mse = [torch.sqrt(m / N) for m in mse]
    total_relative_error = total_relative_error / N
    print('GAME0: {:.4f}, GAME1 {:.4f}, GAME2: {:.4f}, GAME3: {:.4f}, MSE: {:.4f}, Relative Error: {:.4f}'.format(game[0], game[1], game[2], game[3], mse[0], total_relative_error))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='datasets/bayes-RGBT-CC')
    parser.add_argument('--dataset', default='RGBTCC', choices=['RGBTCC', 'ShanghaiTechRGBD'], type=str)
    parser.add_argument('--save-dir', default='ckpt_rgbtcc')
    parser.add_argument('--model', default='best_model.pth')
    parser.add_argument('--type-model', default='bl', choices=['bl', 'csrnet'], type=str)
    args = parser.parse_args()

    print('Testing dataset:', args.dataset)
    print('Testing model:', args.type_model)
    test(args)