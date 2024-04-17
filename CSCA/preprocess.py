import numpy as np
import os
from glob import glob
import cv2
import json
from argparse import ArgumentParser
from natsort import natsorted

def generate_data(label_path):
    rgb_path = label_path.replace('GT', 'RGB').replace('json', 'jpg') # 1162_GT.json -> 1162_RGB.jpg
    t_path = label_path.replace('GT', 'T').replace('json', 'jpg') # 1162_GT.json -> 1162_T.json
    rgb = cv2.imread(rgb_path)[..., ::-1].copy() # rgb image: [480, 640, 3]
    t = cv2.imread(t_path)[..., ::-1].copy() # thermal image: [480, 640, 3]
    im_h, im_w, _ = rgb.shape # 480, 640
    with open(label_path, 'r') as f:
        label_file = json.load(f)
    points = np.asarray(label_file['points']) # head coordinate (x, y): [99, 2]
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h) # keep points inside rgb.shape
    points = points[idx_mask] # [99, 2]
    return rgb, t, points

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--type_dataset', default='RGBT-CC', type=str)
    args = parser.parse_args()

    if args.type_dataset == 'RGBT-CC':
        root_path = 'datasets/RGBT-CC'
        save_path = 'datasets/bayes-RGBT-CC'
    else:
        print('This dataset does not support')
        raise NotImplementedError
    for phase in ['train', 'val', 'test']:
        sub_dir = os.path.join(root_path, phase)
        sub_save_dir = os.path.join(save_path, phase)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        gt_list = natsorted(glob(os.path.join(sub_dir, '*json')))
        for gt_path in gt_list:
            name = os.path.basename(gt_path)
            rgb, t, points = generate_data(gt_path) # rgb images, thermal images and head points
            im_save_path = os.path.join(sub_save_dir, name)
            rgb_save_path = im_save_path.replace('GT', 'RGB').replace('json', 'jpg')
            t_save_path = im_save_path.replace('GT', 'T').replace('json', 'jpg')
            cv2.imwrite(rgb_save_path, rgb)
            cv2.imwrite(t_save_path, t)
            gd_save_path = im_save_path.replace('json', 'npy')
            np.save(gd_save_path, points)