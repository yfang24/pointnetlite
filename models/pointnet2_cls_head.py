import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet2ClsHead(nn.Module):
    def __init__(self, in_dim=1024, hidden_dims=[512, 256], out_dim=40, dropout=0.4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], out_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.drop1(self.fc1(x))))
        x = F.relu(self.bn2(self.drop2(self.fc2(x))))
        x = self.fc3(x)
        return x
