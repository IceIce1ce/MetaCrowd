from torch.nn.modules import Module
import torch

class Bay_Loss(Module):
    def __init__(self, use_background):
        super(Bay_Loss, self).__init__()
        self.use_bg = use_background # True

    def forward(self, prob_list, target_list, pre_density): # [436, 4096], [435] [1, 1, 64, 64]
        loss = 0
        for idx, prob in enumerate(prob_list):
            if prob is None: # image contains no annotation points
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32).cuda() # [0.]
            else:
                N = len(prob) # 4
                if self.use_bg:
                    target = torch.zeros((N,), dtype=torch.float32).cuda() # [1., 1., 1., 1.]
                    target[:-1] = target_list[idx]
                else:
                    target = target_list[idx]
                pre_count = torch.sum(pre_density[idx].view((1, -1)) * prob, dim=1) # [1, 1024] -> 4
            loss += torch.sum(torch.abs(target - pre_count))
        loss = loss / len(prob_list)
        return loss