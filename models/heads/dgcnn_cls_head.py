import torch
import torch.nn as nn

from models.modules.builders import build_fc_layers

class DGCNNClsHead(nn.Module):
    def __init__(self, embed_dim=1024, hidden_dims=[512, 256], out_dim=40, dropout=0.5):
        super().__init__()
        self.fc = build_fc_layers([embed_dim*2] + hidden_dims + [out_dim], linear_bias=False, act=nn.LeakyReLU(negative_slope=0.2, inplace=True)), dropout=dropout)
        
    def forward(self, x):
        return self.fc(x)
