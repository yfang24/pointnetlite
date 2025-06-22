import torch
import torch.nn as nn
import torch.nn.functional as F

class DGCNNLoss(nn.Module):
    def __init__(self, smoothing=True, eps=0.2):
        super().__init__()
        self.smoothing = smoothing
        self.eps = eps

    def forward(self, pred, target):
        target = target.contiguous().view(-1)

        if self.smoothing:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, target, reduction='mean')

        return loss
