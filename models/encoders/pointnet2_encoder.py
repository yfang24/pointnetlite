import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pcd_utils import sample_and_group_ball

class PointNet2Encoder(nn.Module):
    def __init__(self, use_msg=False):
        super().__init__()
        self.use_msg = use_msg # multi-scale grouping

        # Layer configurations
        self.sa1_params = {
            "num_group": 512,
            "group_size": 32,
            "mlps": [[64, 64, 128]] if not use_msg else [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        }

        self.sa2_params = {
            "num_group": 128,
            "group_size": 64,
            "mlps": [[128, 128, 256]] if not use_msg else [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        }

        self.sa3_mlp = [256, 512, 1024] # no grouping; a global mlp to aggregate the whole pc

        self.sa1_mlps = self._build_mlp_blocks(self.sa1_params["mlps"])
        self.sa2_mlps = self._build_mlp_blocks(self.sa2_params["mlps"])

        # for sa3, input is the concatenation of sa2 features from all msg branches
        self.sa3_mlp_layers = self._build_mlp_layers(sum(m[-1] for m in self.sa2_params["mlps"]), self.sa3_mlp)

    def _build_mlp_blocks(self, mlp_list):
        blocks = nn.ModuleList()
        for mlp in mlp_list:
            blocks.append(self._build_mlp_layers(3, mlp))  # Input = groups (B, 3, S, G)
        return blocks

    def _build_mlp_layers(self, in_dim, mlp_channels):
        layers = []
        last_dim = in_dim
        for out_dim in mlp_channels:
            layers.append(nn.Conv2d(last_dim, out_dim, 1)) # shared mlp over local groups
            layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.ReLU())
            last_dim = out_dim
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, N, 3) point cloud

        Returns:
            (B, 1024) global feature vector
        """
        B, N, _ = x.shape

        # SA1
        neighborhoods, centers = sample_and_group(x, self.sa1_params["num_group"], self.sa1_params["group_size"])
        features = self._apply_mlp_blocks(neighborhoods, self.sa1_mlps)

        # SA2
        neighborhoods, centers = sample_and_group(centers, self.sa2_params["num_group"], self.sa2_params["group_size"])
        features = self._apply_mlp_blocks(neighborhoods, self.sa2_mlps)

        # SA3 (global)
        x = features.unsqueeze(-1)  # (B, 256, G, 1)
        x = self.sa3_mlp_layers(x)
        x = torch.max(x, dim=2)[0]  # (B, 1024, 1)
        x = x.squeeze(-1)
        return x  # (B, 1024)

    def _apply_mlp_blocks(self, neighborhoods, mlp_blocks):
        """
        neighborhoods: (B, G, S, 3)
        mlp_blocks: list of shared MLPs
        returns: (B, C, G)
        """
        feats_list = []
        x = neighborhoods.permute(0, 3, 2, 1)  # (B, 3, S, G)
        for mlp in mlp_blocks:
            f = mlp(x)              # (B, C, S, G)
            f = torch.max(f, dim=2)[0]  # (B, C, G); max-pooled over groups
            feats_list.append(f)
        return torch.cat(feats_list, dim=1)  # (B, sum(C), G)
