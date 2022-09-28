import torch
from torch import nn
import numpy as np


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class NCESoftmaxLoss_reduce(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss_reduce, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class NCESoftmaxLoss_sam_threshold(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss_sam_threshold, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, threshold):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        count = 0
        loss_t = 0
        for i in range(loss.cpu().detach().numpy().shape[0]):
            if loss.cpu().detach().numpy()[i] > threshold:
                count = count + 1
                if loss_t == 0:
                    loss_t = loss[i]
                else:
                    loss_t = loss_t + loss[i]
        if count == 0:
            count = 1
            loss_t = loss[0]
        return loss_t / count, count


class NCESoftmaxLossNS(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLossNS, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        # positives on the diagonal
        label = torch.arange(bsz).cuda().long()
        loss = self.criterion(x, label)
        return loss
