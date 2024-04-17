import torch
import os
import argparse
from datasets.crowd import Crowd_RGBTCC, Crowd_shanghaiTechRGBD
from models.fusion import fusion_model
from utils.evaluation import eval_game, eval_relative
import warnings
warnings.filterwarnings('ignore')

def test(args):
    if args.dataset == 'ShanghaiTechRGBD':
        datasets = Crowd_shanghaiTechRGBD(os.path.join(args.data_dir, 'test_data'), method='test')
        dataloader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    elif args.dataset == 'RGBTCC':
        datasets = Crowd_RGBTCC(os.path.join(args.data_dir, 'test'), method='test')
        dataloader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    else:
        print("This dataset does not exist")
        raise NotImplementedError
    # load model
    model = fusion_model()
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
            outputs = model(inputs, args.dataset)
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
    args = parser.parse_args()

    print('Testing dataset:', args.dataset)
    test(args)