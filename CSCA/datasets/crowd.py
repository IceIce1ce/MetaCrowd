import torch.utils.data as data
import os
from glob import glob
import torch
from torchvision import transforms
import random
import numpy as np
import cv2
import scipy.io

def random_crop(im_h, im_w, crop_h, crop_w): # 480, 480, 256, 256
    res_h = im_h - crop_h # 224
    res_w = im_w - crop_w # 224
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

class Crowd_RGBTCC(data.Dataset):
    def __init__(self, root_path, crop_size=256, downsample_ratio=8, method='train'):
        self.root_path = root_path
        self.gt_list = sorted(glob(os.path.join(self.root_path, '*.npy')))
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")
        self.method = method
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio # 32
        self.RGB_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.407, 0.389, 0.396], std=[0.241, 0.246, 0.242])])
        self.T_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.492, 0.168, 0.430], std=[0.317, 0.174, 0.191])])

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, item):
        gt_path = self.gt_list[item]
        rgb_path = gt_path.replace('GT', 'RGB').replace('npy', 'jpg')
        t_path = gt_path.replace('GT', 'T').replace('npy', 'jpg')
        RGB = cv2.imread(rgb_path)[..., ::-1].copy() # [480, 640, 3]
        T = cv2.imread(t_path)[..., ::-1].copy() # [480, 640, 3]
        if self.method == 'train':
            keypoints = np.load(gt_path) # head points
            return self.train_transform(RGB, T, keypoints)
        elif self.method == 'val' or self.method == 'test':
            keypoints = np.load(gt_path) # head points
            gt = keypoints
            k = np.zeros((T.shape[0], T.shape[1])) # [480, 640]
            for i in range(0, len(gt)):
                if int(gt[i][1]) < T.shape[0] and int(gt[i][0]) < T.shape[1]: # y_coord < w and x_coord < h
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            target = k # [480, 640]
            RGB = self.RGB_transform(RGB) # [3, 480, 640]
            T = self.T_transform(T) # [3, 480, 640]
            name = os.path.basename(gt_path).split('.')[0]
            input = [RGB, T]
            return input, target, name
        else:
            raise Exception("Not implement")

    def train_transform(self, RGB, T, keypoints):
        ht, wd, _ = RGB.shape # 480, 640
        st_size = 1.0 * min(wd, ht) # 480
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        RGB = RGB[i: i + h, j: j + w, :] # [256, 256, 3]
        T = T[i: i + h, j: j + w, :] # [256, 256, 3]
        keypoints = keypoints - [j, i]
        idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
        keypoints = keypoints[idx_mask]
        RGB = self.RGB_transform(RGB) # [3, 256, 256]
        T = self.T_transform(T) # [3, 256, 256]
        input = [RGB, T]
        return input, torch.from_numpy(keypoints.copy()).float(), st_size

class Crowd_shanghaiTechRGBD(data.Dataset):
    def __init__(self, root_path, crop_size=1024, downsample_ratio=8, method='train'):
        self.root_path = root_path
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")
        if method == 'train':
            self.gt_list = sorted(glob(os.path.join(self.root_path, 'train_gt', '*.mat')))
        elif method == 'test':
            self.gt_list = sorted(glob(os.path.join(self.root_path, 'test_bbox_anno', '*.mat')))
        elif method == 'val':
            self.root_path = self.root_path.replace('val_data', 'test_data')
            self.gt_list = sorted(glob(os.path.join(self.root_path, 'test_bbox_anno', '*.mat')))
        else:
            raise Exception("Not implement")
        self.method = method
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        self.RGB_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.407, 0.389, 0.396], std=[0.241, 0.246, 0.242]), ])
        self.depth_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.441], std=[0.329])])

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, item):
        gt_path = self.gt_list[item]
        if self.method == 'train':
            rgb_path = gt_path.replace('train_gt', 'train_img').replace('GT', 'IMG').replace('mat', 'png')
            d_path = gt_path.replace('train_gt', 'train_depth').replace('GT', 'DEPTH').replace('mat', 'npy')
            RGB = cv2.imread(rgb_path)[..., ::-1].copy()
            Depth = np.load(d_path)
            keypoints = scipy.io.loadmat(gt_path)['point']
            return self.train_transform(RGB, Depth, keypoints)
        elif self.method == 'val' or self.method == 'test':
            rgb_path = gt_path.replace('test_bbox_anno', 'test_img').replace('BBOX', 'IMG').replace('mat', 'png')
            d_path = gt_path.replace('test_bbox_anno', 'test_depth').replace('BBOX', 'DEPTH').replace('mat', 'npy')
            RGB = cv2.imread(rgb_path)[..., ::-1].copy()
            Depth = np.load(d_path)
            bbox = scipy.io.loadmat(gt_path)['bbox']
            gt = np.zeros((bbox.shape[0], 2))
            for i in range(bbox.shape[0]):
                gt[i][0] = int((bbox[i][0] + bbox[i][2]) / 2)
                gt[i][1] = int((bbox[i][1] + bbox[i][3]) / 2)
            k = np.zeros((Depth.shape[0], Depth.shape[1]))
            for i in range(0, len(gt)):
                if int(gt[i][1]) < Depth.shape[0] and int(gt[i][0]) < Depth.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            target = k
            RGB = self.RGB_transform(RGB)
            Depth = self.depth_transform(Depth)
            name = os.path.basename(gt_path).split('.')[0]
            input = [RGB, Depth]
            return input, target, name
        else:
            raise Exception("Not implement")

    def train_transform(self, RGB, Depth, keypoints):
        ht, wd, _ = RGB.shape
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        RGB = RGB[i: i + h, j:j + w, :]
        Depth = Depth[i: i + h, j: j + w]
        keypoints = keypoints - [j, i]
        idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
        keypoints = keypoints[idx_mask]
        RGB = self.RGB_transform(RGB)
        Depth = self.depth_transform(Depth)
        input = [RGB, Depth]
        return input, torch.from_numpy(keypoints.copy()).float(), st_size