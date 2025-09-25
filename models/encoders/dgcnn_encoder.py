import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pcd_utils import knn_group, group_points
from models.modules.builders import build_shared_mlp

class DGCNNEncoder(nn.Module):
    def __init__(self, k=20, in_dim=3, embed_dim=1024, hidden_dims=[64, 64, 128, 256]):
        super().__init__()
        self.k = k
        self.embed_dim = embed_dim

        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = build_shared_mlp([in_dim * 2, hidden_dims[0]], conv_dim=2, conv_bias=False, act=act, final_act=True)
        self.conv2 = build_shared_mlp([hidden_dims[0] * 2, hidden_dims[1]], conv_dim=2, conv_bias=False, act=act, final_act=True)
        self.conv3 = build_shared_mlp([hidden_dims[1] * 2, hidden_dims[2]], conv_dim=2, conv_bias=False, act=act, final_act=True)
        self.conv4 = build_shared_mlp([hidden_dims[2] * 2, hidden_dims[3]], conv_dim=2, conv_bias=False, act=act, final_act=True)
        self.conv5 = build_shared_mlp([hidden_dims[3] * 2, embed_dim], conv_dim=1, conv_bias=False, act=act, final_act=True)

    def _edge_conv(self, x, conv, k=None):  # x: (B, N, C)
        if k is None:
            k = self.k
        idx = knn_group(x, x, k)                  # (B, N, k); for each point, gather its knn neighbors
        grouped = group_points(x, idx)            # (B, N, k, C)
        centers = x.unsqueeze(2)                  # (B, N, 1, C)
        edge = torch.cat((grouped - centers, centers.expand_as(grouped)), dim=-1)  # (B, N, k, 2C); for each grouped point, set its group coords + group center as features
        edge = edge.permute(0, 3, 1, 2).contiguous()  # (B, 2C, N, k)
        out = conv(edge).max(dim=-1)[0]           # (B, C_out, N)
        return out.transpose(1, 2)             # (B, N, C_out)
        
    def forward(self, x):  # (B, N, 3)
        B, _, _ = x.shape
        
        x1 = self._edge_conv(x, self.conv1)   # (B, N, C1)
        x2 = self._edge_conv(x1, self.conv2)  # (B, N, C2)
        x3 = self._edge_conv(x2, self.conv3)  # (B, N, C3)
        x4 = self._edge_conv(x3, self.conv4)  # (B, N, C4)

        x_cat = torch.cat((x1, x2, x3, x4), dim=-1)   # (B, N, C_total)

        x_cat = x_cat.transpose(1, 2)                 # (B, C_total, N)
        x_feat = self.conv5(x_cat)                    # (B, embed_dim, N)
        
        x_max = F.adaptive_max_pool1d(x_feat, 1).view(B, -1)  # (B, embed_dim)
        x_avg = F.adaptive_avg_pool1d(x_feat, 1).view(B, -1)  # (B, embed_dim)

        x_global = torch.cat((x_max, x_avg), dim=1)   # (B, embed_dim*2)
        return x_global
