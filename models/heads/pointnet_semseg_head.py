import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetSemSegHead(nn.Module):
    def __init__(self, embed_dim=1088, out_dim=13):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, out_dim, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.out_dim = out_dim

    def forward(self, x):  # x: (B, C, N)
        B, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()  # (B, N, out_dim)
        return x.view(-1, self.out_dim)
