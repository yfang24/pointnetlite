import torch
import torch.nn as nn

from models.modules.builders import build_fc_layers

class DGCNNClsHead(nn.Module):
    def __init__(self, embed_dim=1024, hidden_dims=[512, 256], out_dim=40, dropout=0.5):
        super().__init__()

        self.fc = build_fc_layers([embed_dim*2] + hidden_dims, 
                                  act=nn.LeakyReLU(negative_slope=0.2, inplace=True)), 
                                  dropout=dropout, final_bn=False, final_act=False)
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.fc3(x)
        return x
