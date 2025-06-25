import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pcd_utils import fps, ball_group, group_points

class PointNet2Encoder(nn.Module):
    def __init__(self, in_dim=3, use_msg=False):
        super().__init__()

        # mlp_layer: conv2d + bn2d + relu
        # mlp: stack of mlp_layers
        # mlp_block: list of mlps
        
        # Layer configurations
        self.sa1_params = {
            "num_group": 512,
            "group_size": [32] if not use_msg else [16, 32, 128],
            "radius": [0.2] if not use_msg else [0.1, 0.2, 0.4],
            "mlps": [[64, 64, 128]] if not use_msg else [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        }
        
        self.sa2_params = {
            "num_group": 128,
            "group_size": [64] if not use_msg else [32, 64, 128],
            "radius": [0.4] if not use_msg else [0.2, 0.4, 0.8],
            "mlps": [[128, 128, 256]] if not use_msg else [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        }

        # last sa: no grouping; a global mlp to aggregate the whole pc
        self.sa3_params = {
            "num_group": 1,
            "group_size": [-1],
            "radius": [-1],
            "mlps": [[256, 512, 1024]]
        }

        # for msg, sa layer input is the concatenation of features from all msg branches of earlier sa layer
        self.sa1_mlps = [self._build_mlp(in_dim, sa1_mlp_channels) for sa1_mlp_channels in self.sa1_params["mlps"]]
        self.sa2_mlps = [self._build_mlp(in_dim + sum(m[-1] for m in self.sa1_params["mlps"]), sa2_mlp_channels) for sa2_mlp_channels in self.sa2_params["mlps"]]
        self.sa3_mlps = [self._build_mlp(in_dim + sum(m[-1] for m in self.sa2_params["mlps"]), sa3_mlp_channels) for sa3_mlp_channels in self.sa3_params["mlps"]]
    
    def _build_mlp(self, in_dim, mlp_channels):
        layers = []
        last_dim = in_dim
        for out_dim in mlp_channels:
            layers += [nn.Conv2d(last_dim, out_dim, 1), nn.BatchNorm2d(out_dim), nn.ReLU()] # conv2d: shared mlp over local groups
            last_dim = out_dim
        return nn.Sequential(*layers)
        
    def _apply_mlp(self, x, mlp):
        """
        x: (B, G, S, C_in)
        returns: (B, G, C_out)
        """
        x = neighborhoods.permute(0, 3, 2, 1)  # (B, C_in, S, G)
        x = mlp(x).max(dim=2)[0]               # (B, C_out, S, G), max over S -> (B, C_out, G) 
        return x.permute(0, 2, 1)              # (B, G, C_out)

    def _single_scale_group(self, input_points, input_feats, centers, radius, group_size, mlp):
        """
        input_points: (B, G, 3)
        input_feats: (B, G, D)
        centers: (B, G, 3)
        """
        if radius == -1:   # for last sa (sa3); treat whole pc as one group: num_group G = 1, group_size k = G
            neighborhood_points = input_points.unsqueeze(1)    # (B, 1, G, 3)
        else:
            idx = ball_group(input_points, centers, radius, group_size)  # (B, G, k)
            neighborhood_points = group_points(input_points, idx) - centers.unsqueeze(2)  # (B, G, k, 3)    
        if input_feats is not None:
            if radius == -1:
                grouped_feats = input_feats.unsqueeze(1)    # (B, 1, G, D)
            else:
                grouped_feats = group_points(input_feats, idx)  # (B, G, k, D)
            neighborhood_feats = torch.cat([neighborhood_points, grouped_feats], dim=-1)  # (B, G, k, 3+D); (B, 1, G, 3+D) for sa3
        else:   # for sa1
            neighborhood_feats = neighborhood_points  # (B, G, k, 3)    
        return self._apply_mlp(neighborhood_feats, mlp)  # group_features; (B, G, C); (B, 1, C) for sa3

    def _sa_layer(self, input_points, input_feats, num_group, radius, group_size, mlps):
        centers = fps(input_points, num_group)    # (B, G, 3)
        feats_list = []
        for r, k, mlp in zip(radius, group_size, mlps):
            f = self._single_scale_group(input_points, input_feats, centers, r, k, mlp)  # (B, G, C)
            feats_list.append(f)
        features = torch.cat(feats_list, dim=2)  # (B, G, sum(C))
        return centers, features

    def forward(self, x):
        B, N, _ = x.shape

        # SA1
        centers1, features1 = self._sa_layer(
            x,
            None,
            self.sa1_params["num_group"],
            self.sa1_params["radius"],
            self.sa1_params["group_size"],
            self.sa1_mlps
        )

        # SA2
        centers2, features2 = self._sa_layer(
            centers1,
            features1,
            self.sa2_params["num_group"],
            self.sa2_params["radius"],
            self.sa2_params["group_size"],
            self.sa2_mlps
        )
        
        # SA3 (global)    
        _, features3 = self._sa_layer(
            centers2,
            features2,
            self.sa3_params["num_group"],
            self.sa3_params["radius"],
            self.sa3_params["group_size"],
            self.sa3_mlps
        )
        
        x = features3.squeeze(-1)
        return x  # (B, 1024)

    
