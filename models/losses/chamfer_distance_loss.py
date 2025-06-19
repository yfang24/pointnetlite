import torch
import torch.nn as nn

def square_distance(src, dst):
    """
    Calculate squared distances between each pair of points.
    src: (B, N, C)
    dst: (B, M, C)
    return: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))  # (B, N, M)
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(2)    # (B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)    # (B, 1, M)
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
