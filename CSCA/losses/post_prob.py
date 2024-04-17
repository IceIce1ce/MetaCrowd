import torch
from torch.nn import Module

class Post_Prob(Module):
    def __init__(self, sigma, c_size, stride, background_ratio, use_background): # 8, 256, 8, 0.15, True
        super(Post_Prob, self).__init__()
        assert c_size % stride == 0
        self.sigma = sigma # 8
        self.bg_ratio = background_ratio # 0.15
        self.cood = torch.arange(0, c_size, step=stride, dtype=torch.float32).cuda() + stride / 2 # [32]
        self.cood.unsqueeze_(0) # [1, 32]
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background

    def forward(self, points, st_sizes):
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0)
        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1) # [3] -> [3, 1]
            y = all_points[:, 1].unsqueeze_(1) # [3] -> [3, 1]
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood # [3, 32]
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood # [3, 32]
            y_dis.unsqueeze_(2) # [3, 32, 1]
            x_dis.unsqueeze_(1) # [3, 1, 32]
            dis = y_dis + x_dis # [3, 32, 32]
            dis = dis.view((dis.size(0), -1)) # [3, 1024]
            dis_list = torch.split(dis, num_points_per_image)
            prob_list = []
            for dis, st_size in zip(dis_list, st_sizes):
                if len(dis) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0) # [1, 1024]
                        d = st_size * self.bg_ratio # 72
                        bg_dis = (d - torch.sqrt(min_dis))**2 # [1, 1024]
                        dis = torch.cat([dis, bg_dis], 0) # [4, 1024]
                    dis = -dis / (2.0 * self.sigma ** 2) # [4, 1024]
                    prob = self.softmax(dis) # [4, 1024]
                else:
                    prob = None
                prob_list.append(prob)
        else:
            prob_list = []
            for _ in range(len(points)):
                prob_list.append(None)
        return prob_list