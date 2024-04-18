from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np

def random_crop(im_h, im_w, crop_h, crop_w): # 480, 480, 256, 256
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right - inner_left, 0.0) * np.maximum(inner_down - inner_up, 0.0)
    return inner_area

class Crowd_UCF(data.Dataset):
    def __init__(self, root_path, crop_size, downsample_ratio, is_gray=False, method='train'):
        self.root_path = root_path
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.method = method
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        if is_gray:
            self.trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        else:
            self.trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gt_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB') # [991, 618]
        if self.method == 'train':
            keypoints = np.load(gt_path) # [252, 3]
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gt_path) # [175, 2]
            img = self.trans(img) # [3, 512, 766]
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        wd, ht = img.size # 827, 512
        st_size = min(wd, ht) # 512
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w) # [512, 512]
        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0) # 921
        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0 # [644, 2]
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0 # [379, 2]
        bbox = np.concatenate((points_left_up, points_right_down), axis=1) # [542, 4]
        inner_area = cal_innner_area(j, i, j + w, i + h, bbox) # [343]
        origin_area = nearest_dis * nearest_dis # [597]
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0) # [4096]
        mask = (ratio >= 0.3) # [4096]
        target = ratio[mask] # [345]
        keypoints = keypoints[mask] # [264, 3]
        keypoints = keypoints[:, :2] - [j, i] # change coordinate
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(target.copy()).float(), st_size # [3, 512, 512], [0, 2], [0], 2048