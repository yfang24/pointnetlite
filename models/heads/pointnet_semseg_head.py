import torch
import torch.nn as nn

from models.modules.builders import build_fc_layers

class PointNetSemSegHead(nn.Module):
    def __init__(self, embed_dim=1024 + 64, hidden_dims=[512, 256, 128], out_dim=13, dropout=0.):
        super().__init__()
        self.fc = build_fc_layers([embed_dim] + hidden_dims + [out_dim], linear_bias=True, act=nn.ReLU(inplace=True), dropout=dropout)

    def forward(self, x):  # x: (B, C, N)
        B, _, N = x.size()
        x = self.fc(x)
        x = x.transpose(2, 1).contiguous()  # (B, N, out_dim)
        return x.view(-1, self.out_dim)
