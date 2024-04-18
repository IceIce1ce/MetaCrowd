from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio

def find_dis(point):
    square = np.sum(point * points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2 * np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size # 2572, 1928
    mat_path = im_path.replace('.jpg', '_ann.mat')
    points = loadmat(mat_path)['annPoints'].astype(np.float32) # [144, 2]
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h) # keep points inside rgb.shape
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size) # 1928, 2572, 1.0
    im = np.array(im) # [1928, 2572, 3]
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC) # [2048, 3059, 3]
        points = points * rr
    return Image.fromarray(im), points

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_dir', default='datasets/UCF-QNRF')
    parser.add_argument('--data_dir', default='datasets/UCF-Train-Val-Test')
    args = parser.parse_args()

    print('Processing dataset:', args.original_dir.split('/')[-1])
    min_size = 512
    max_size = 2048
    for phase in ['Train', 'Test']:
        sub_dir = os.path.join(args.original_dir, phase)
        if phase == 'Train':
            sub_phase_list = ['train', 'val']
            for sub_phase in sub_phase_list:
                sub_save_dir = os.path.join(args.data_dir, sub_phase)
                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
                with open('{}.txt'.format(sub_phase)) as f:
                    for i in f:
                        im_path = os.path.join(sub_dir, i.strip())
                        name = os.path.basename(im_path)
                        im, points = generate_data(im_path) # [2572, 1928], [143, 2]
                        if sub_phase == 'train':
                            dis = find_dis(points) # [143, 1]
                            points = np.concatenate((points, dis), axis=1) # [143, 3]
                        im_save_path = os.path.join(sub_save_dir, name)
                        im.save(im_save_path)
                        gd_save_path = im_save_path.replace('jpg', 'npy')
                        np.save(gd_save_path, points)
        else:
            sub_save_dir = os.path.join(args.data_dir, 'test')
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
            im_list = glob(os.path.join(sub_dir, '*jpg'))
            for im_path in im_list:
                name = os.path.basename(im_path)
                im, points = generate_data(im_path)
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)