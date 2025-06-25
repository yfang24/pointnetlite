import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.builders import build_shared_mlp, build_fc_layers

# A Spatial Transformer Network (STN) that learns a kxk transformation matrix.
class STNkd(nn.Module):
    def __init__(self, k=64, act=nn.ReLU(inplace=True), dropout=0.0):
        super().__init__()
        self.k = k

        self.mlp = build_shared_mlp([k, 64, 128, 1024], conv_dim=1, act=act, final_act=True)
        self.fc = build_fc_layers([1024, 512, 256, k * k], act=act, dropout=dropout, final_bn=False, final_act=False)

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


class PointNetEncoder(nn.Module):
    def __init__(self, in_dim=3, embed_dim=1024, hidden_dims=[64, 128], return_all=False, feature_transform=True):
        """
        return_all (bool): If True, return global_feat (B, embed_dim), local_feat (B, N, embed_dim)
        feature_transform (bool): Whether to apply a second STN on 64-dim features
        """
        super().__init__()
        self.return_all = return_all
        self.feature_transform = feature_transform

        # Input transformation
        self.input_stn = STNk(in_dim)
        
        # Shared MLP: input -> 64
        self.mlp1 = build_shared_mlp([in_dim, hidden_dims[0]], conv_dim=1)

        # Feature transformation
        if self.feature_transform:
            self.feature_stn = STNkd(k=64)
        

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
