import torch
import torch.nn as nn

from models.modules.builders import build_shared_mlp, build_fc_layers

# A Spatial Transformer Network (STN) that learns a kxk transformation matrix.
class STNk(nn.Module):
    def __init__(self, k=64, act=nn.ReLU(inplace=True), dropout=0.0):
        super().__init__()
        self.k = k

        self.mlp = build_shared_mlp([k, 64, 128, 1024], conv_dim=1, act=act, final_act=True)
        self.fc = build_fc_layers([1024, 512, 256, k * k], act=act, dropout=dropout)

    def forward(self, x):  # x: (B, k, N)
        B = x.size(0)
              
        x = self.mlp(x)                          # (B, 1024, N)
        x = torch.max(x, dim=2)[0]              # (B, 1024)
        x = self.fc(x)                           # (B, k*k)

        # An identity matrix is added to ensure the transformation matrix starts close to the identity.
        iden = torch.eye(self.k, device=x.device).flatten().unsqueeze(0).repeat(B, 1)  # (B, k*k)
        x = x + iden
        return x.view(B, self.k, self.k)

class PointNetEncoder(nn.Module):
    def __init__(self, in_dim=3, embed_dim=1024, hidden_dims=[64, 128], feature_transform=True, return_all=False):
        """
        return_all (bool): If True, return global_feat (B, embed_dim), trans_feat, local_feat (B, N, embed_dim)
        feature_transform (bool): Whether to apply a second STN on 64-dim features
        """
        super().__init__()
        self.return_all = return_all
        self.feature_transform = feature_transform

        # Input transformation
        self.input_stn = STNk(in_dim)
        
        # Shared MLP: input -> 64
        self.mlp1 = build_shared_mlp([in_dim, hidden_dims[0]], conv_dim=1, final_act=True)

        # Feature transformation
        if self.feature_transform:
            self.feature_stn = STNk(k=hidden_dims[0])
        
        # Shared MLP: 64 -> 128 -> embed_dim
        self.mlp2 = build_shared_mlp(hidden_dims + [embed_dim], conv_dim=1, final_act=False)
          
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, D, N) 
        
        # Input transform
        trans_input = self.input_stn(x)
        x = torch.bmm(trans_input, x)

        x = self.mlp1(x)  # (B, 64, N)

        # Feature transform (optional)
        if self.feature_transform:
            trans_feat = self.feature_stn(x)
            x = torch.bmm(trans_feat, x)
        else:
            trans_feat = None

        local_feat = x.permute(0, 2, 1)  # (B, N, 64)
        
        x = self.mlp2(x)  # (B, embed_dim, N)
        global_feat = torch.max(x, dim=2)[0]  # (B, embed_dim)

        if self.return_all:
            return global_feat, trans_feat, local_feat
        else:
            return global_feat, trans_feat
