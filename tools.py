from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import numpy as np
import warnings
import math


class CosineAnnealingLRWarmup(CosineAnnealingLR):

    def __init__(self, *args, **kwargs):
        self.warmup_iter = kwargs['warmup_iter']
        self.cur_iter = 0
        self.warmup_ratio = kwargs['warmup_ratio']
        self.init_lr = None
        del kwargs['warmup_iter']
        del kwargs['warmup_ratio']
        super(CosineAnnealingLRWarmup, self).__init__(*args, **kwargs)
        self.init_lr = [group['lr'] for group in self.optimizer.param_groups]

    def iter_step(self):
        self.cur_iter += 1
        if self.cur_iter <= self.warmup_iter and self.init_lr:
            values = [lr * (self.warmup_ratio + (1 - self.warmup_ratio) * (self.cur_iter / self.warmup_iter))
                      for lr in self.init_lr]
            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, 0)

            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - init_lr * self.eta_min) + init_lr * self.eta_min
                for init_lr, group in zip(self.init_lr, self.optimizer.param_groups)]


def transform_input(x):
    split_num = 4
    x = x.T.tolist()
    x.sort(key=lambda x: x[0])
    total = len(x)
    result = []
    split_size = int(total / split_num)
    for i in range(split_num):
        x_sorted = x[i * split_size:(i + 1) * split_size]

        x_sorted.sort(key=lambda x: x[1])
        total_x = len(x_sorted)
        split_size_x = int(total_x / split_num)
        result_x = []
        for j in range(split_num):
            sorted_y = x_sorted[i * split_size_x:(i + 1) * split_size_x]
            sorted_y.sort(key=lambda x: x[2])
            total_y = len(sorted_y)
            split_size_y = int(total_y / split_num)
            result_y = []
            for k in range(split_num):
                sorted_z = sorted_y[i * split_size_y:(i + 1) * split_size_y]
                result_y.append(sorted_z)
            result_x.append(result_y)
        result.append(result_x)

    re_ts = torch.FloatTensor(result).flatten(3).flatten(0, 2)

    return re_ts
