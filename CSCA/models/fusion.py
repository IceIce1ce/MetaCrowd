import torch.nn as nn
import torch
from torch.nn import functional as F

class FusionModel(nn.Module):
    def __init__(self, ratio=0.6):
        super(FusionModel, self).__init__()
        c1 = int(64 * ratio) # 38
        c2 = int(128 * ratio) # 76
        c3 = int(256 * ratio) # 153
        c4 = int(512 * ratio) # 307
        self.block1_depth = Block([c1, c1, 'M'], in_channels=3, L=4, first_block=True, D_in_channels=True)
        self.block1 = Block([c1, c1, 'M'], in_channels=3, L=4, first_block=True, D_in_channels=False)
        self.block2 = Block([c2, c2, 'M'], in_channels=c1, L=3)
        self.block3 = Block([c3, c3, c3, c3, 'M'], in_channels=c2, L=2)
        self.block4 = Block([c4, c4, c4, c4, 'M'], in_channels=c3, L=1)
        self.block5 = Block([c4, c4, c4, c4], in_channels=c4, L=1)
        self.reg_layer = nn.Sequential(nn.Conv2d(c4, c3, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(c3, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 1, 1))
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

    def forward(self, RGBT, dataset):
        RGB = RGBT[0] # [1, 3, 256, 256]
        T = RGBT[1] # [1, 3, 256, 256]
        if dataset == 'ShanghaiTechRGBD':
            RGB, T, shared = self.block1_depth(RGB, T)
        else:
            RGB, T, shared = self.block1(RGB, T) # [1, 38, 128, 128], [1, 38, 128, 128], [1, 38, 128, 128]
        RGB, T, shared = self.block2(RGB, T) # [1, 76, 64, 64], [1, 76, 64, 64], [1, 76, 64, 64]
        RGB, T, shared = self.block3(RGB, T) # [1, 153, 32, 32], [1, 153, 32, 32], [1, 153, 32, 32]
        RGB, T, shared = self.block4(RGB, T) # [1, 307, 16, 16], [1, 307, 16, 16], [1, 307, 16, 16]
        _, _, shared = self.block5(RGB, T) # [1, 307, 16, 16], [1, 307, 16, 16], [1, 307, 16, 16]
        x = shared # [1, 307, 16, 16]
        x = F.upsample_bilinear(x, scale_factor=2) # [1, 307, 32, 32]
        x = self.reg_layer(x) # [1, 1, 32, 32]
        return torch.abs(x)

class Block(nn.Module):
    def __init__(self, cfg, in_channels, L, first_block=False, dilation_rate=1, D_in_channels=False):
        super(Block, self).__init__()
        self.seen = 0
        self.first_block = first_block
        self.d_rate = dilation_rate
        self.L = L

        if first_block:
            if D_in_channels:
                t_in_channels = 1
            else:
                t_in_channels = in_channels
        else:
            t_in_channels = in_channels

        self.rgb_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
        self.t_conv = make_layers(cfg, in_channels=t_in_channels, d_rate=self.d_rate)
        if first_block is False:
            self.shared_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
        channels = cfg[0] # 38, 38, 76, 153, 307, 307
        self.out_channels = channels // 2 # 19, 19, 38, 76, 153, 153
        self.rgb_msc = MSC(channels)
        self.t_msc = MSC(channels)

        self.RGB_key = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0), nn.Dropout(0.5), nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.RGB_query = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0), nn.Dropout(0.5), nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.RGB_value = nn.Conv2d(in_channels=channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0)
        self.RGB_W = nn.Conv2d(in_channels=self.out_channels, out_channels=channels, kernel_size=1, stride=1, padding=0)

        self.T_key = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0), nn.Dropout(0.5), nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.T_query = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0), nn.Dropout(0.5), nn.BatchNorm2d(self.out_channels), nn.ReLU())
        self.T_value = nn.Conv2d(in_channels=channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0)
        self.T_W = nn.Conv2d(in_channels=self.out_channels, out_channels=channels, kernel_size=1, stride=1, padding=0)

        self.gate_RGB = nn.Conv2d(channels * 2, 1, kernel_size=1, bias=True)
        self.gate_T = nn.Conv2d(channels * 2, 1, kernel_size=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, RGB, T): # [1, 3, 256, 256], [1, 3, 256, 256]
        RGB = self.rgb_conv(RGB) # [1.38, 128, 128]
        T = self.t_conv(T) # [1, 38, 128, 128]
        new_RGB, new_T, new_shared = self.fuse(RGB, T) # [1, 38, 128, 128], [1, 38, 128, 128], [1, 38, 128, 128]
        return new_RGB, new_T, new_shared

    def fuse(self, RGB, T): # [1, 38, 128, 128], [1, 38, 128, 128]
        RGB_m = self.rgb_msc(RGB) # [1, 38, 128, 128]
        T_m = self.t_msc(T) # [1, 38, 128, 128]
        # SCA Block
        adapt_channels = 2 ** self.L * self.out_channels # 304
        batch_size = RGB_m.size(0) # 1
        rgb_query = self.RGB_query(RGB_m).view(batch_size, adapt_channels, -1).permute(0, 2, 1) # [1, 19, 128, 128] -> [1, 1024, 304]
        rgb_key = self.RGB_key(RGB_m).view(batch_size, adapt_channels, -1) # [1, 19, 128, 128] -> [1, 304, 1024]
        rgb_value = self.RGB_value(RGB_m).view(batch_size, adapt_channels, -1).permute(0, 2, 1) # [1, 19, 128, 128] -> [1, 1024, 304]

        batch_size = T_m.size(0) # 1
        T_query = self.T_query(T_m).view(batch_size, adapt_channels, -1).permute(0, 2, 1) # [1, 19, 128, 128] -> [1, 1024, 304]
        T_key = self.T_key(T_m).view(batch_size, adapt_channels, -1) # [1, 19, 128, 128] -> [1, 304, 1024]
        T_value = self.T_value(T_m).view(batch_size, adapt_channels, -1).permute(0, 2, 1) # [1, 19, 128, 128] -> [1, 1024, 304]

        RGB_sim_map = torch.matmul(T_query, rgb_key) # [1, 1024, 1024]
        RGB_sim_map = (adapt_channels ** -.5) * RGB_sim_map # [1, 1024, 1024]
        RGB_sim_map = F.softmax(RGB_sim_map, dim=-1) # [1, 1024, 1024]
        RGB_context = torch.matmul(RGB_sim_map, rgb_value) # [1, 1024, 304]
        RGB_context = RGB_context.permute(0, 2, 1).contiguous() # [1, 304, 1024]
        RGB_context = RGB_context.view(batch_size, self.out_channels,  *RGB_m.size()[2:]) # [1, 19, 128, 128]
        RGB_context = self.RGB_W(RGB_context) # [1, 38, 128, 128]

        T_sim_map = torch.matmul(rgb_query, T_key) # [1, 1024, 1024]
        T_sim_map = (adapt_channels ** -.5) * T_sim_map # [1, 1024, 1024]
        T_sim_map = F.softmax(T_sim_map, dim=-1) # [1, 1024, 1024]
        T_context = torch.matmul(T_sim_map, T_value) # [1, 1024, 304]
        T_context = T_context.permute(0, 2, 1).contiguous() # [1, 304, 1024]
        T_context = T_context.view(batch_size, self.out_channels, *T_m.size()[2:]) # [1, 19, 128, 128]
        T_context = self.T_W(T_context) # [1, 38, 128, 128]
        # CFA Block
        cat_fea = torch.cat([T_context, RGB_context], dim=1) # [1, 76, 128, 128]
        attention_vector_RGB = self.gate_RGB(cat_fea) # [1, 1, 128, 128]
        attention_vector_T = self.gate_T(cat_fea) # [1, 1, 128, 128]
        attention_vector = torch.cat([attention_vector_RGB, attention_vector_T], dim=1) # [1, 2, 128, 128]
        attention_vector = self.softmax(attention_vector)
        attention_vector_RGB, attention_vector_T = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :] # [1, 1, 128, 128], [1, 1, 128, 128]
        new_shared = RGB * attention_vector_RGB + T * attention_vector_T # [1, 38, 128, 128]
        new_RGB = (RGB + new_shared) / 2 # [1, 38, 128, 128]
        new_T = (T + new_shared) / 2 # [1, 38, 128, 128]
        new_RGB = self.relu1(new_RGB) # [1, 38, 128, 128]
        new_T = self.relu2(new_T) # [1, 38, 128, 128]
        return new_RGB, new_T, new_shared

class MSC(nn.Module):
    def __init__(self, channels):
        super(MSC, self).__init__()
        self.channels = channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv = nn.Sequential(nn.Conv2d(3 * channels, channels, kernel_size=1), nn.ReLU(inplace=True))

    def forward(self, x): # [1, 38, 128, 128]
        x1 = nn.functional.interpolate(self.pool1(x), x.shape[2:]) # [1, 38, 128, 128]
        x2 = nn.functional.interpolate(self.pool2(x), x.shape[2:]) # [1, 38, 128, 128]
        concat = torch.cat([x, x1, x2], 1) # [1, 114, 128, 128]
        fusion = self.conv(concat) # [1, 38, 128, 128]
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