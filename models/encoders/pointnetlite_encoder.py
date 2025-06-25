import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.builders import build_shared_mlp

class PointNetLiteEncoder(nn.Module):
    def __init__(self, in_dim=3, embed_dim=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, embed_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(embed_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, 3, N)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, dim=2)
        return x  # (B, 1024)

class PointNetLiteEncoder(nn.Module):
    def __init__(self, in_dim=3, hidden_dims=[64, 128], embed_dim=1024, conv_dim=1, act=nn.ReLU(inplace=True), return_all=False):
        """
        conv_dim (int): default=1; set to 2 if working with grouped points (B, G, S, C)
        return_all (bool): If True, return global_feat (B, embed_dim), local_feat (B, N, embed_dim)
        """
        super().__init__()
        self.return_all = return_all
        self.mlp = build_shared_mlp([in_dim] + hidden_dims + [embed_dim], conv_dim=1, act=act)

    def forward(self, x):
        """
        x.shape = (B, N, in_dim)
        """
        x = x.permute(0, 2, 1)  # (B, in_dim, N)
        feat = self.mlp(x)      # (B, embed_dim, N)
        global_feat = torch.max(feat, dim=2)[0]  # (B, embed_dim)

        if self.return_all:
            local_feat = feat.permute(0, 2, 1)  # (B, N, embed_dim)
            return global_feat, local_feat
        return global_feat
