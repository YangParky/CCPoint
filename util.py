import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.autograd as autograd


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def get_time():
    time_now = time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(int(round(time.time()*1000))/1000))

    return time_now


class IOStream():
    def __init__(self, path, rank=-1):
        self.rank = rank
        if self.rank == 0:
            self.f = open(path, 'a')

    def cprint(self, text):
        if self.rank == 0:
            print(text)
            self.f.write(text+'\n')
            self.f.flush()

    def close(self):
        if self.rank == 0:
            self.f.close()


class contrastive_loss(nn.Module):
    def __init__(self, T=1.0):
        super().__init__()
        self.T = T

    def forward(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

        # gather all targets
        k = concat_all_gather(k)

        # Einstein sum is more intuitive
        logits = torch.einsum('nc, mc -> nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()

        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)


class AverageMeter(object):
    """Computes and stores the average and current value"""
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
        self.avg = self.sum / self.count