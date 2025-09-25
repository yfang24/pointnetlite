import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance

#=====
# Using PyTorch3D funcs
#=====
class ChamferDistance(nn.Module):
    def __init__(self, norm: int = 2):
        super().__init__()
        self.norm = norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss, _ = chamfer_distance(pred, target, norm=self.norm)
        return loss

class ChamferDistanceL2(ChamferDistance):
    def __init__(self):
        super().__init__(norm=2)


class ChamferDistanceL1(ChamferDistance):
    def __init__(self):
        super().__init__(norm=1)

'''
#=====
# Using my PyTorch funcs
#=====
def square_distance(src, dst):
    # torch.cdist computes L2 distances; we square them
    dist = torch.cdist(src, dst, p=2) ** 2   # (B, N, M)
    return dist

def chamfer_distance_l2(src, dst):
    """
    src, dst: (B, N, 3)
    return: mean of (min_dist(src->dst) + min_dist(dst->src))
    """
    dist = square_distance(src, dst)                           # (B, N, M)
    dist_src2dst = torch.min(dist, dim=2)[0]                   # (B, N)
    dist_dst2src = torch.min(dist, dim=1)[0]                   # (B, M)
    loss = dist_src2dst.mean(dim=1) + dist_dst2src.mean(dim=1)
    return loss.mean()

def chamfer_distance_l1(src, dst):
    """
    Same as above, but with L1 (rooted) distance.
    """
    dist = torch.sqrt(square_distance(src, dst) + 1e-6)        # (B, N, M)
    dist_src2dst = torch.min(dist, dim=2)[0]                   # (B, N)
    dist_dst2src = torch.min(dist, dim=1)[0]                   # (B, M)
    loss = dist_src2dst.mean(dim=1) + dist_dst2src.mean(dim=1)
    return loss.mean()

class ChamferDistanceL2(nn.Module):
    def forward(self, pred, target):
        return chamfer_distance_l2(pred, target)

class ChamferDistanceL1(nn.Module):
    def forward(self, pred, target):
        return chamfer_distance_l1(pred, target)
'''