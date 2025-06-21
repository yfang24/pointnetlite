import torch
import torch.nn as nn
import torch.nn.functional as F
from pcd_utils import sample_and_group

class PointNet2Encoder(nn.Module):
    def __init__(self, num_group=512, group_size=32, mlp_channels=[64, 128, 256]):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.mlp = nn.Sequential()
        last_dim = 3  # since sample_and_group returns (B, G, M, 3)
        for i, out_dim in enumerate(mlp_channels):
            self.mlp.add_module(f"conv{i}", nn.Conv2d(last_dim, out_dim, 1))
            self.mlp.add_module(f"bn{i}", nn.BatchNorm2d(out_dim))
            self.mlp.add_module(f"relu{i}", nn.ReLU())
            last_dim = out_dim

    def forward(self, points):
        # points: (B, N, 3)
        B, N, _ = points.shape
        neighborhoods, centers = sample_and_group(points, self.num_group, self.group_size)  # (B, G, M, 3), (B, G, 3)
        x = neighborhoods.permute(0, 3, 2, 1)  # (B, 3, M, G)
        x = self.mlp(x)                        # (B, C, M, G)
        x = torch.max(x, 2)[0]                 # (B, C, G)
        x = torch.max(x, 2)[0]                 # (B, C)
        return x
