import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pcd_utils import fps, ball_group, group_points

class PointNet2Encoder(nn.Module):
    def __init__(self, in_dim=3, use_msg=False):
        super().__init__()
        self.use_msg = use_msg

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
        self.sa3_mlp_channels = [256, 512, 1024]

        # for msg, sa layer input is the concatenation of features from all msg branches of earlier sa layer
        self.sa1_mlps = [self._build_mlp(in_dim, mlp) for mlp in self.sa1_params["mlps"]]
        self.sa2_mlps = [self._build_mlp(in_dim + sum(m[-1] for m in self.sa1_params["mlps"]), mlp) for mlp in self.sa2_params["mlps"]]
        self.sa3_mlp = self._build_mlp(in_dim + sum(m[-1] for m in self.sa2_params["mlps"]), self.sa3_mlp_channels)

    # mlp_layer: conv2d + bn2d + relu
    # mlp: stack of mlp_layers
    def _build_mlp(self, in_dim, mlp_channels):
        layers = []
        for out_dim in mlp_channels:
            layers += [nn.Conv2d(in_dim, out_dim, 1), nn.BatchNorm2d(out_dim), nn.ReLU()] # conv2d: shared mlp over local groups
            in_dim = out_dim
        return nn.Sequential(*layers)
        
    def _apply_mlp(self, x, mlp):   # x: neighborhood_points; (B, G, k, C_in)
        x = neighborhoods.permute(0, 3, 2, 1)  # (B, C_in, k, G)
        x = mlp(x).max(dim=2)[0]               # (B, C_out, k, G), max over neighborhood k -> (B, C_out, G) 
        return x.permute(0, 2, 1)              # (B, G, C_out)

    def _sa_layer(self, points, feats, params, mlps):
        centers = fps(points, params["num_group"])  # (B, G, 3)
        feat_list = []
        for r, k, mlp in zip(params["radius"], params["group_size"], mlps):
            idx = ball_group(points, centers, r, k)  # (B, G, k)
            neighbor_pts = group_points(points, idx) - centers.unsqueeze(2)  # (B, G, k, 3)
            if feats is not None:
                neighbor_feats = group_points(feats, idx)  # (B, G, k, D)
                group_feats = torch.cat([neighbor_pts, neighbor_feats], dim=-1)  # (B, G, k, 3+D)
            else:
                group_feats = neighbor_pts
            feat_list.append(self._apply_mlp(group_feats, mlp))  # (B, G, C)
        return centers, torch.cat(feat_list, dim=2)  # (B, G, sum(C))

    def _global_sa(self, points, feats, mlp):   # treat whole pc as one group, i.e., G = 1, k = N
        points = points.unsqueeze(1)  # (B, 1, N, 3)
        if feats is not None:
            feats = feats.unsqueeze(1)  # (B, 1, N, D)
            x = torch.cat([points, feats], dim=-1)  # (B, 1, N, 3+D)
        else:
            x = points
        return self._apply_mlp(x, mlp).squeeze(1)  # (B, 1, C) -> (B, C)

    def forward(self, x):
        # SA1
        c1, f1 = self._sa_layer(x, None, self.sa1_params, self.sa1_mlps)
        # SA2
        c2, f2 = self._sa_layer(c1, f1, self.sa2_params, self.sa2_mlps)
        # SA3
        return self._global_sa(c2, f2, self.sa3_mlp)  # (B, 1024)
