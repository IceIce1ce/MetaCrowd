import torch
import os
import numpy as np
from datasets.crowd import Crowd_UCF
from models.vgg import vgg19
import argparse
import warnings
warnings.filterwarnings('ignore')

def test(args):
    datasets = Crowd_UCF(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
    model = vgg19()
    model.cuda()
    checkpoint = os.path.join(args.save_dir, 'best_model.pth')
    model.load_state_dict(torch.load(checkpoint, map_location='cuda'))
    epoch_minus = []
    for inputs, count, name in dataloader:
        inputs = inputs.cuda()
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            epoch_minus.append(temp_minu)
    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    print('MAE: {:.4f}, MSE: {:.4f}'.format(mae, mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='datasets/UCF-Train-Val-Test')
    parser.add_argument('--save-dir', default='ckpt_ucf')
    parser.add_argument('--dataset', default='ucf', type=str)
    args = parser.parse_args()

    print('Testing dataset:', args.dataset)
    test(args)