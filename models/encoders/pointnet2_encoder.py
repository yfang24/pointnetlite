import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pcd_utils import sample_and_group_ball, farthest_point_sample_gpu_batch, ball_query, group_points

class PointNet2Encoder(nn.Module):
    def __init__(self, use_msg=False):
        super().__init__()
        self.use_msg = use_msg # multi-scale grouping

        # Layer configurations
        self.sa1_params = {
            "num_group": 512,
            "group_size": 32 if not use_msg else [16, 32, 128],
            "radius": 0.2 if not use_msg else [0.1, 0.2, 0.4],
            "mlps": [[64, 64, 128]] if not use_msg else [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        }
        
        self.sa2_params = {
            "num_group": 128,
            "group_size": 64 if not use_msg else [32, 64, 128],
            "radius": 0.4 if not use_msg else [0.2, 0.4, 0.8],
            "mlps": [[128, 128, 256]] if not use_msg else [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        }

        self.sa3_mlp = [256, 512, 1024] # no grouping; a global mlp to aggregate the whole pc

        self.sa1_mlps = self._build_mlp_blocks(self.sa1_params["mlps"])
        self.sa2_mlps = self._build_mlp_blocks(self.sa2_params["mlps"])

        # for sa3, input is the concatenation of sa2 features from all msg branches
        self.sa3_mlp_layers = self._build_mlp_layers(sum(m[-1] for m in self.sa2_params["mlps"]), self.sa3_mlp)

    def _build_mlp_blocks(self, mlp_list):
        return nn.ModuleList([self._build_mlp_layers(3, mlp) for mlp in mlp_list])
    
    def _build_mlp_layers(self, in_dim, mlp_channels):
        layers = []
        last_dim = in_dim
        for out_dim in mlp_channels:
            layers += [nn.Conv2d(last_dim, out_dim, 1), nn.BatchNorm2d(out_dim), nn.ReLU()] # conv2d: shared mlp over local groups
            last_dim = out_dim
        return nn.Sequential(*layers)

    def _apply_mlp_blocks(self, neighborhoods, mlp_blocks):
        """
        neighborhoods: (B, G, S, 3)
        mlp_blocks: list of shared MLPs
        returns: (B, C, G)
        """
        x = neighborhoods.permute(0, 3, 2, 1)  # (B, 3, S, G)
        return torch.cat([mlp(x).max(dim=2)[0] for mlp in mlp_blocks], dim=1)  # (B, C_total, G); max-pooled over groups and concatenated across scales

    def _sa_layer(self, input_points, num_group, radius, group_size, mlp_blocks):
        centers = farthest_point_sample_gpu_batch(input_points, num_group)    
        if self.use_msg:
            feats_list = []
            for r, k, mlp in zip(radius, group_size, mlp_blocks):
                idx = ball_query(centers, input_points, r, k)
                grouped = group_points(input_points, idx) - centers.unsqueeze(2)
                f = mlp(grouped.permute(0, 3, 2, 1))  # (B, C, k, G)
                f = torch.max(f, dim=2)[0]            # (B, C, G)
                feats_list.append(f)
            features = torch.cat(feats_list, dim=1)   # (B, sum(C), G)
        else:
            neighborhoods, centers = sample_and_group_ball(input_points, num_group, group_size, radius)
            features = self._apply_mlp_blocks(neighborhoods, mlp_blocks)  # (B, C, G)    
        return features, centers

    def forward(self, x):
        B, N, _ = x.shape

        # SA1
        features, centers = self._sa_layer(
            x,
            self.sa1_params["num_group"],
            self.sa1_params["radius"],
            self.sa1_params["group_size"],
            self.sa1_mlps
        )

        # SA2
        features, centers = self._sa_layer(
            centers,
            self.sa2_params["num_group"],
            self.sa2_params["radius"],
            self.sa2_params["group_size"],
            self.sa2_mlps
        )
        
        # SA3 (global)
        x = features.unsqueeze(-1)  # (B, 256, G, 1)
        x = self.sa3_mlp_layers(x)
        x = torch.max(x, dim=2)[0]  # (B, 1024, 1)
        x = x.squeeze(-1)
        return x  # (B, 1024)

    
