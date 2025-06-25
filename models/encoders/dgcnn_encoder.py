import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pcd_utils import knn_group, group_points


class DGCNNEncoder(nn.Module):
    def __init__(self, k=20, in_dim=3, embed_dim=1024):
        super().__init__()
        self.k = k
        self.embed_dim = embed_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(0.2)
        )

    def _get_graph_feature(self, x, k):
        """
        Args:
            x: (B, N, C)
        Returns:
            edge_features: (B, 6, N, k)
        """
        idx = knn_group(x, x, k)               # (B, N, k); for each point, gather its knn neighbors
        grouped = group_points(x, idx)          # (B, N, k, 3)
        centers = x.unsqueeze(2)                # (B, N, 1, 3)
        edge = torch.cat((grouped - centers, centers.expand_as(grouped)), dim=-1)  # (B, N, k, 6) # for each grouped point, set its group coords + group center as features
        return edge.permute(0, 3, 1, 2).contiguous()  # (B, 6, N, k)

    def forward(self, x):
        """
        Args:
            x: (B, N, 3)
        Returns:
            global feature: (B, embed_dim*2)
        """
        B, _, _ = x.shape
        
        x1 = self.conv1(self._get_graph_feature(x, self.k)).max(dim=-1)[0]
        x2 = self.conv2(self._get_graph_feature(x1.transpose(1, 2), self.k)).max(dim=-1)[0]
        x3 = self.conv3(self._get_graph_feature(x2.transpose(1, 2), self.k)).max(dim=-1)[0]
        x4 = self.conv4(self._get_graph_feature(x3.transpose(1, 2), self.k)).max(dim=-1)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        x_feat = self.conv5(x_cat)                 # (B, embed_dim, N)
        
        x1 = F.adaptive_max_pool1d(x_feat, 1).view(B, -1)  # (B, embed_dim)
        x2 = F.adaptive_avg_pool1d(x_feat, 1).view(B, -1)
        x_global = torch.cat((x1, x2), dim=1)
        return x_global  # (B, embed_dim*2)
