from utils.evaluation import eval_game, eval_relative
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datasets.crowd import Crowd_RGBTCC
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
from models.BL_IADM import BL_IADM
from models.CSRNet_IADM import CSRNet_IADM

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
    if type(transposed_batch[0][0]) == list:
        rgb_list = [item[0] for item in transposed_batch[0]] # [3, 256, 256]
        t_list = [item[1] for item in transposed_batch[0]] # [3, 256, 256]
        rgb = torch.stack(rgb_list, 0) # [1, 3, 256, 256]
        t = torch.stack(t_list, 0) # [1, 3, 256, 256]
        images = [rgb, t]
    else:
        images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1] # [51, 2]
    st_sizes = torch.FloatTensor(transposed_batch[2]) # [1] -> 480.
    return images, points, st_sizes

class RegTrainer():
    def __init__(self, args):
        self.args = args
        self.save_dir = args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def setup(self):
        args = self.args
        self.downsample_ratio = args.downsample_ratio # 8
        self.dataset = args.dataset # RGBT-CC
        # init training dataset
        if args.dataset == 'RGBTCC':
            self.datasets = {x: Crowd_RGBTCC(os.path.join(args.data_dir, x), args.crop_size, args.downsample_ratio, x) for x in ['train', 'val', 'test']}
            self.dataloaders = {x: DataLoader(self.datasets[x], collate_fn=(train_collate if x == 'train' else default_collate), batch_size=(args.batch_size if x == 'train' else 1),
                                shuffle=(True if x == 'train' else False), num_workers=args.num_workers, pin_memory=(True if x == 'train' else False)) for x in ['train', 'val', 'test']}
        # init model and optimizer
        if args.type_model == 'bl':
            self.model = BL_IADM()
        elif args.type_model == 'csrnet':
            self.model = CSRNet_IADM()
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
        # init loss
        self.post_prob = Post_Prob(args.sigma, args.crop_size, args.downsample_ratio, args.background_ratio, args.use_background)
        self.criterion = Bay_Loss(args.use_background)
        # init metric
        self.best_game0 = np.inf
        self.best_game3 = np.inf
        self.best_rmse = np.inf
        self.best_count = 0

    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            self.epoch = epoch
            self.train_eopch()
            if epoch % 1 == 0 and epoch >= args.val_start:
                game0_is_best, game3_is_best = self.val_epoch()
            if epoch >= args.val_start and (game0_is_best or game3_is_best):
                self.test_epoch()

    def train_eopch(self):
        print('Start training')
        epoch_loss = AverageMeter()
        epoch_game = AverageMeter()
        epoch_mse = AverageMeter()
        self.model.train()
        for step, (inputs, points, st_sizes) in enumerate(self.dataloaders['train']):
            if type(inputs) == list:
                inputs[0] = inputs[0].cuda() # [[1, 3, 256, 256]
                inputs[1] = inputs[1].cuda() # [1, 3, 256, 256]
            else:
                inputs = inputs.cuda()
            st_sizes = st_sizes.cuda() # 480
            gd_count = np.array([len(p) for p in points], dtype=np.float32) # [1]
            points = [p.cuda() for p in points]
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs) # [1, 1, 32, 32]
                prob_list = self.post_prob(points, st_sizes) # [4, 1024]
                loss = self.criterion(prob_list, outputs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if type(inputs) == list:
                    N = inputs[0].size(0) # 1
                else:
                    N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_game.update(np.mean(abs(res)), N)
        print('Epoch: {}, Loss: {:.4f}, GAME0: {:.4f}. MSE: {:.4f}'.format(self.epoch, epoch_loss.get_avg(), epoch_game.get_avg(), np.sqrt(epoch_mse.get_avg())))
        # save at each epoch just for using reume training
        # model_state_dic = self.model.state_dict()
        # save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        # torch.save({'epoch': self.epoch, 'optimizer_state_dict': self.optimizer.state_dict(), 'model_state_dict': model_state_dic}, save_path)

    def val_epoch(self):
        print('Start validating')
        self.model.eval()
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        total_relative_error = 0
        for inputs, target, name in self.dataloaders['val']: # target: [1, 480, 640]
            if type(inputs) == list:
                inputs[0] = inputs[0].cuda() # [1, 3, 480, 640]
                inputs[1] = inputs[1].cuda() # [1, 3, 480, 640]
            else:
                inputs = inputs.cuda()
            if type(inputs) == list:
                assert inputs[0].size(0) == 1
            else:
                assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs) # [1, 1, 60, 80]
                for L in range(4):
                    abs_error, square_error = eval_game(outputs, target, L)
                    game[L] += abs_error
                    mse[L] += square_error
                relative_error = eval_relative(outputs, target)
                total_relative_error += relative_error
        N = len(self.dataloaders['val']) # 200
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        total_relative_error = total_relative_error / N
        print('Epoch: {}, GAME0: {:.4f}, GAME1: {:.4f}, GAME2: {:.4f}, GAME3: {:.4f}, MSE: {:.4f}, Relative Error: {:.4f}'.format(self.epoch, game[0], game[1], game[2], game[3], mse[0], total_relative_error))
        model_state_dic = self.model.state_dict()
        game0_is_best = game[0] < self.best_game0
        game3_is_best = game[3] < self.best_game3
        if game[0] < self.best_game0 or game[3] < self.best_game3:
            self.best_game3 = min(game[3], self.best_game3)
            self.best_game0 = min(game[0], self.best_game0)
            print("Best val GAME0: {:.4f}, GAME3: {:.4f} at epoch {}".format(self.best_game0, self.best_game3, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))
        return game0_is_best, game3_is_best

    def test_epoch(self):
        print('Start testing')
        self.model.eval()
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        total_relative_error = 0
        for inputs, target, name in self.dataloaders['test']:
            if type(inputs) == list:
                inputs[0] = inputs[0].cuda() # [1, 3, 480, 640]
                inputs[1] = inputs[1].cuda() # [1, 3, 480, 640]
            else:
                inputs = inputs.cuda() # [1, 3, 480, 640]
            if type(inputs) == list:
                assert inputs[0].size(0) == 1
            else:
                assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                for L in range(4):
                    abs_error, square_error = eval_game(outputs, target, L)
                    game[L] += abs_error
                    mse[L] += square_error
                relative_error = eval_relative(outputs, target)
                total_relative_error += relative_error
        N = len(self.dataloaders['test'])
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        total_relative_error = total_relative_error / N
        print('Epoch: {}, GAME0: {:.4f}, GAME1: {:.4f}, GAME2: {:.4f}, GAME3: {:.4f}, MSE: {:.4f}, Relative Error: {:.4f}'.format(self.epoch, game[0], game[1], game[2], game[3], mse[0], total_relative_error))
        print()