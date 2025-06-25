import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.builders import build_shared_mlp, build_fc_layers

# A Spatial Transformer Network (STN) that learns a kxk transformation matrix.
class STNkd(nn.Module):
    def __init__(self, k=64, act=nn.ReLU(inplace=True), dropout=0.0):
        super().__init__()
        self.k = k

        self.mlp = build_shared_mlp([k, 64, 128, 1024], conv_dim=1, act=act)
        self.fc = build_fc_layers([1024, 512, 256, k * k], act=act, final_act=False, dropout=dropout)

    def forward(self, x):  # x: (B, N, k)
        B = x.size(0)
        
        x = x.permute(0, 2, 1)  # (B, k, N)        
        x = self.mlp(x)                          # (B, 1024, N)
        x = torch.max(x, dim=2)[0]              # (B, 1024)
        x = self.fc(x)                           # (B, k*k)

        # An identity matrix is added to ensure the transformation matrix starts close to the identity.
        iden = torch.eye(self.k, device=x.device).flatten().unsqueeze(0).repeat(B, 1)  # (B, k*k)
        x = x + iden
        return x.view(B, self.k, self.k)

""" 
encode point cloud, output local/global features, which are fed into MLP for classification (see pointnet_cls).
if global_feat=True, only global feat is returned; otherwise, returns a concat of global and local features
"""
class PointNetEncoder(nn.Module):
    def __init__(self, in_dim=3, embed_dim=1024, hidden_dims=[64, 128], global_feat=True, feature_transform=True):
        super().__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.input_stn = STNk(in_dim)
        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        if feature_transform:
            self.feature_stn = STNk(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, embed_dim, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(embed_dim)

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        B, N, D = x.size()
        x = x.permute(0, 2, 1)
        trans = self.stn(x)
        x = torch.bmm(trans, x)

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = torch.bmm(trans_feat, x)
        else:
            trans_feat = None

        point_feat = x  # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # (B, 1024, N)
        x = torch.max(x, 2)[0]  # (B, 1024)

        if self.global_feat:
            return x, trans_feat
        else:
            x = x.view(B, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, point_feat], 1), trans_feat
