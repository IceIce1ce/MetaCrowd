import torch.nn as nn
import torch
from torch.nn import functional as F

class BL_IADM(nn.Module):
    def __init__(self, ratio=0.6):
        super(BL_IADM, self).__init__()
        c1 = int(64 * ratio)
        c2 = int(128 * ratio)
        c3 = int(256 * ratio)
        c4 = int(512 * ratio)
        self.block1 = Block([c1, c1, 'M'], in_channels=3, first_block=True)
        self.block2 = Block([c2, c2, 'M'], in_channels=c1)
        self.block3 = Block([c3, c3, c3, c3, 'M'], in_channels=c2)
        self.block4 = Block([c4, c4, c4, c4, 'M'], in_channels=c3)
        self.block5 = Block([c4, c4, c4, c4], in_channels=c4)
        self.reg_layer = nn.Sequential(nn.Conv2d(c4, c3, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c3, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 1, 1))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, RGBT):
        RGB = RGBT[0] # [1, 3, 256, 256]
        T = RGBT[1] # [1, 3, 256, 256]
        RGB, T, shared = self.block1(RGB, T, None) # [1, 38, 128, 128]
        RGB, T, shared = self.block2(RGB, T, shared) # [1, 76, 64, 64]
        RGB, T, shared = self.block3(RGB, T, shared) # [1, 153, 32, 32]
        RGB, T, shared = self.block4(RGB, T, shared) # [1, 307, 16, 16]
        _, _, shared = self.block5(RGB, T, shared) # [1, 307, 16, 16]
        x = shared # [1, 307, 16, 16]
        x = F.upsample_bilinear(x, scale_factor=2) # [1, 307, 32, 32]
        x = self.reg_layer(x) # [1, 1, 32, 32]
        return torch.abs(x)

class Block(nn.Module):
    def __init__(self, cfg, in_channels, first_block=False, dilation_rate=1):
        super(Block, self).__init__()
        self.seen = 0
        self.first_block = first_block
        self.d_rate = dilation_rate
        self.rgb_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
        self.t_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
        if first_block is False:
            self.shared_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
        channels = cfg[0]
        self.rgb_msc = MSC(channels)
        self.t_msc = MSC(channels)
        if first_block is False:
            self.shared_fuse_msc = MSC(channels)
        self.shared_distribute_msc = MSC(channels)
        self.rgb_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.t_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.rgb_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.t_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, RGB, T, shared): # [1, 3, 256, 256], [1, 3, 256, 256]
        RGB = self.rgb_conv(RGB)
        T = self.t_conv(T)
        if self.first_block:
            shared = torch.zeros(RGB.shape).cuda()
        else:
            shared = self.shared_conv(shared)
        new_RGB, new_T, new_shared = self.fuse(RGB, T, shared)
        return new_RGB, new_T, new_shared

    def fuse(self, RGB, T, shared):
        RGB_m = self.rgb_msc(RGB)
        T_m = self.t_msc(T)
        if self.first_block:
            shared_m = shared # init zero for first block
        else:
            shared_m = self.shared_fuse_msc(shared) # get share from first block for second block
        # Information Aggregation module
        rgb_s = self.rgb_fuse_1x1conv(RGB_m - shared_m) # Conv1_2
        rgb_fuse_gate = torch.sigmoid(rgb_s) # fuse rgb and share in Figure 4a
        t_s = self.t_fuse_1x1conv(T_m - shared_m) # Conv1_2
        t_fuse_gate = torch.sigmoid(t_s) # fuse thermal and share in Figure 4a
        new_shared = shared + (RGB_m - shared_m) * rgb_fuse_gate + (T_m - shared_m) * t_fuse_gate # Eq 3
        # Information Distribution module
        new_shared_m = self.shared_distribute_msc(new_shared)
        s_rgb = self.rgb_distribute_1x1conv(new_shared_m - RGB_m) # Conv1_2
        rgb_distribute_gate = torch.sigmoid(s_rgb) # fuse rgb and share in Figure 4b
        s_t = self.t_distribute_1x1conv(new_shared_m - T_m) # Conv1_2
        t_distribute_gate = torch.sigmoid(s_t) # fuse thermal and share in Figure 4b
        new_RGB = RGB + (new_shared_m - RGB_m) * rgb_distribute_gate # Eq 4
        new_T = T + (new_shared_m - T_m) * t_distribute_gate # Eq 4
        return new_RGB, new_T, new_shared

class MSC(nn.Module):
    def __init__(self, channels):
        super(MSC, self).__init__()
        self.channels = channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv = nn.Sequential(nn.Conv2d(3 * channels, channels, kernel_size=1), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = nn.functional.interpolate(self.pool1(x), x.shape[2:])
        x2 = nn.functional.interpolate(self.pool2(x), x.shape[2:])
        concat = torch.cat([x, x1, x2], 1)
        fusion = self.conv(concat)
        return fusion

def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)