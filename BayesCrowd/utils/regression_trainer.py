import os
import sys
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from  models.vgg import vgg19
from datasets.crowd import Crowd_UCF
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes

class RegTrainer():
    def __init__(self, args):
        self.args = args
        self.save_dir = args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def setup(self):
        args = self.args
        self.downsample_ratio = args.downsample_ratio # 8
        # init training dataset
        self.datasets = {x: Crowd_UCF(os.path.join(args.data_dir, x), args.crop_size, args.downsample_ratio, args.is_gray, x) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x], collate_fn=(train_collate if x == 'train' else default_collate), batch_size=(args.batch_size if x == 'train' else 1),
                            shuffle=(True if x == 'train' else False), num_workers=args.num_workers, pin_memory=(True if x == 'train' else False)) for x in ['train', 'val']}
        # init model and optimizer
        self.model = vgg19()
        self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.start_epoch = 0
        # resume training
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, map_location='cuda')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, map_location='cuda'))
        self.post_prob = Post_Prob(args.sigma, args.crop_size, args.downsample_ratio, args.background_ratio, args.use_background)
        self.criterion = Bay_Loss(args.use_background)
        # init metric
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            self.epoch = epoch
            self.train_epoch()
            # if epoch % 1 == 0 and epoch >= args.val_start:
            if epoch % 1 == 0:
                self.val_epoch()

    def train_epoch(self):
        print('Start training')
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        self.model.train()
        for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            inputs = inputs.cuda() # [1, 3, 512, 512]
            st_sizes = st_sizes.cuda()
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.cuda() for p in points]
            targets = [t.cuda() for t in targets]
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs) # [1, 1, 64, 64]
                prob_list = self.post_prob(points, st_sizes)
                loss = self.criterion(prob_list, targets, outputs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)
        print('Epoch: {}, Loss: {:.4f}, MSE: {:.4f}, MAE: {:.4f}'.format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg()))
        # save at each epoch just for using resume training
        # model_state_dic = self.model.state_dict()
        # save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        # torch.save({'epoch': self.epoch, 'optimizer_state_dict': self.optimizer.state_dict(), 'model_state_dict': model_state_dic}, save_path)

    def val_epoch(self):
        print('Start validating')
        self.model.eval()
        epoch_res = []
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.cuda() # [1, 3, 1875, 2500]
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs) # [1, 1, 234, 312]
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        print('Epoch: {}, MSE: {:.4f}, MAE: {:.4f}'.format(self.epoch, mse, mae))
        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            print("Best MSE: {:.4f}, MAE: {:.4f} at epoch {}".format(self.best_mse, self.best_mae, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))