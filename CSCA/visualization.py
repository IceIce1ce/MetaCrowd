import numpy as np
import os
import torch
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
from models.fusion import FusionModel
import warnings
warnings.filterwarnings("ignore")

def image_processing(gt_path):
    RGB_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.407, 0.389, 0.396], std=[0.241, 0.246, 0.242])])
    T_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.492, 0.168, 0.430], std=[0.317, 0.174, 0.191])])
    rgb_path = gt_path.replace('GT', 'RGB').replace('npy', 'jpg')
    t_path = gt_path.replace('GT', 'T').replace('npy', 'jpg')
    RGB = cv2.imread(rgb_path)[..., ::-1].copy() # [480, 640, 3]
    T = cv2.imread(t_path)[..., ::-1].copy() # [480, 640, 3]
    keypoints = np.load(gt_path)
    gt = keypoints # [99, 2]
    k = np.zeros((T.shape[0], T.shape[1])) # [480, 640]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < T.shape[0] and int(gt[i][0]) < T.shape[1]: # y < h and x < w:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    target = k # [480, 640]
    RGB = RGB_transform(RGB) # [3, 480, 640]
    T = T_transform(T) # [3, 480, 640]
    name = os.path.basename(gt_path).split('.')[0]
    input = [RGB, T]
    return input, target, name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processGT', default='datasets/bayes-RGBT-CC/train/1162_GT.npy')
    parser.add_argument('--dataset', default='RGBTCC', type=str, choices=['RGBTCC', 'ShanghaiTechRGBD'])
    parser.add_argument('--save-dir', default='ckpt_rgbtcc/')
    args = parser.parse_args()

    print('Visualizing dataset:', args.dataset)
    input, target, name = image_processing(args.processGT)
    input[0] = torch.unsqueeze(input[0], 0).cuda() # [1, 3, 480, 640]
    input[1] = torch.unsqueeze(input[1], 0).cuda() # [1, 3, 480, 640]
    model = FusionModel()
    model.cuda()
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint)
    model.eval()
    output = model(input, args.dataset) # [1, 1, 60, 80]
    output = output.cpu().detach().numpy() # [1, 1, 60, 80]
    pre_count = output.sum()
    target_num = target.sum()
    output = output[0][0] # [60, 80]
    H, W = target.shape # 480, 640
    ratio = H / output.shape[0] # 8.0
    output = cv2.resize(output, (W, H), interpolation=cv2.INTER_CUBIC) / (ratio * ratio)
    plt.imshow(output, cmap=cm.jet)
    plt.axis('off')
    plt.savefig('vis.jpg', bbox_inches='tight', pad_inches=0.0)
    print('Predicted count:', pre_count)
    print('Target count:', target_num)