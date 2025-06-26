import torch
import torch.nn as nn

class DGCNNClsHead(nn.Module):
    def __init__(self, embed_dim=1024, hidden_dims=[512, 256], out_dim=40, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim*2, hidden_dims[0], bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dp1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1], bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dp2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dims[1], out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.fc3(x)
        return x
