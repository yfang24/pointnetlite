import torch
import torch.nn as nn
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self, channel=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x): # (B, N, 3)
        B = x.size(0)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(B, -1)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, dtype=torch.float32).view(1, 9).repeat(B, 1).to(x.device)
        x = x + iden
        return x.view(-1, 3, 3)

# A Spatial Transformer Network (STN) that learns a kxk transformation matrix.
class STNkd(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        B = x.size(0)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(B, -1)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # An identity matrix is added to ensure the transformation matrix starts close to the identity.
        iden = torch.eye(self.k, dtype=torch.float32).flatten().view(1, self.k * self.k).repeat(B, 1).to(x.device)
        x = x + iden
        return x.view(-1, self.k, self.k)

'''    
encode point cloud, output local/global features, which are fed into MLP for classification (see pointnet_cls).
if global_feat=True, only global feat is returned; otherwise, returns a concat of global and local features
'''
class PointNetEncoder(nn.Module):
    def __init__(self, in_dim=3, embed_dim=1024, global_feat=True, feature_transform=True):
        super().__init__()
        self.stn = STN3d(in_dim)
        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, embed_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(embed_dim)

        self.feature_transform = feature_transform
        self.global_feat = global_feat

        if feature_transform:
            self.fstn = STNkd(k=64)

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
