import torch
import torch.nn as nn

from models.modules.builders import build_fc_layers

class PointNetSemSegHead(nn.Module):
    def __init__(self, embed_dim=1024, local_dim=64, hidden_dims=[512, 256, 128], out_dim=13, dropout=0.):
        super().__init__()
        self.out_dim = out_dim
        self.fc = build_fc_layers([embed_dim + local_dim] + hidden_dims + [out_dim], linear_bias=True, act=nn.ReLU(inplace=True), dropout=dropout)

    def forward(self, x):
        global_feat, local_feat = x   # (B, embed_dim), (B, N, local_dim)
        
        B, N, _ = local_feat.shape
        global_expanded = global_feat.unsqueeze(1).expand(-1, N, -1)  # (B, N, embed_dim)
        x = torch.cat([global_expanded, local_feat], dim=-1)         # (B, N, embed_dim + local_dim)
        
        x = x.permute(0, 2, 1)                                        # (B, C, N)
        x = self.fc(x)                                                # (B, out_dim, N)
        x = x.transpose(2, 1).contiguous()                            # (B, N, out_dim)
        return x.view(-1, self.out_dim)                               # (B * N, out_dim)       
