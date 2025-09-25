import torch
import torch.nn as nn

from utils.pcd_utils import fps, knn_group, group_points
from models.modules.builders import build_fc_layers, build_shared_mlp


class PosE(nn.Module):
    """Fourier-style positional encoding"""
    def __init__(self, out_dim, alpha=1000, beta=100):
        super().__init__()
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):  # (B, G, S, 3=in_dim)
        B, G, S, in_dim = xyz.shape

        # each in_channel is encoded into 2 feats (sin and cos); concatenated to make up for out_dim
        assert self.out_dim % (2 * in_dim) == 0, \
            f"out_dim ({self.out_dim}) must be divisible by 2*in_dim ({2*in_dim})"
        feat_dim = self.out_dim // (2 * in_dim)

        # frequency bases
        freq = torch.arange(feat_dim, device=xyz.device).float()
        denom = torch.pow(self.alpha, freq / feat_dim)  # (feat_dim,)

        # sinusoidal expansion (coords projected onto bases)
        z = (self.beta * xyz.unsqueeze(-1)) / denom  # (B, G, S, 3, F)
        pos = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)  # (B, G, S, 3, 2F)
        return pos.reshape(B, G, S, self.out_dim)  # (B, G, S, D)


class LGA(nn.Module):
    """Local Geometry Aggregation"""
    def __init__(self, in_dim, out_dim, num_group, group_size, blocks=1, alpha=1000, beta=100):
        super().__init__()
        self.num_group, self.group_size = num_group, group_size

        # linear projection for expanded group feats
        self.linear = build_shared_mlp([in_dim * 2, out_dim], conv_dim=2, final_act=True)

        # positional encoding
        self.pos_embed = PosE(out_dim, alpha, beta)

        # residual blocks
        self.blocks = nn.ModuleList([
            build_shared_mlp([out_dim, out_dim // 2, out_dim], conv_dim=2)
            for _ in range(blocks)
        ])
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, feats, dataset="sim"):
        """
        points: (B, N, 3)
        feats:  (B, N, D)
        """
        B, N, _ = points.shape
        G, S = self.num_group, self.group_size

        # --- sampling + grouping ---
        centers = fps(points, G)                   # (B, G, 3)
        idx = knn_group(points, centers, S)        # (B, G, S)
        neighbor_pts = group_points(points, idx)   # (B, G, S, 3)
        neighbor_feats = group_points(feats, idx)  # (B, G, S, D)

        # --- normalize neighbor coords ---
        if dataset == "sim":   # e.g., ModelNet; standard z-score normalization
            mean, std = centers.unsqueeze(2), torch.std(neighbor_pts - centers.unsqueeze(2))
            neighbor_pts = (neighbor_pts - mean) / (std + 1e-5)
        else:  # e.g., ScanObjectNN; normalize into unit sphere
            neighbor_pts = (neighbor_pts - centers.unsqueeze(2)) / (
                neighbor_pts.abs().max(dim=2, keepdim=True)[0] + 1e-5
            )
        
        # center features
        center_idx = knn_group(points, centers, k=1)   # (B, G, 1)
        center_feats = group_points(feats, center_idx).squeeze(2)  # (B, G, D)       

        # --- expand group features ---
        group_feats = torch.cat(
            [neighbor_feats, center_feats.unsqueeze(2).expand(-1, -1, S, -1)], dim=-1
        )  # (B, G, S, 2D)

        # --- linear projection ---
        group_feats = group_feats.permute(0, 3, 2, 1)   # (B, 2D, S, G)
        group_feats = self.linear(group_feats).permute(0, 3, 2, 1)    # (B, G, S, D)

        # --- positional encoding ---
        group_pos = self.pos_embed(neighbor_pts)    # (B, G, S, D)
        group_feats = (group_feats + group_pos) * group_pos  # inject + weighting      

        # --- residual convs ---
        group_feats = group_feats.permute(0, 3, 1, 2)  # (B, D, G, S)
        for block in self.blocks:
            group_feats = self.act(block(group_feats) + group_feats)   # (B, D, G, S)

        # --- pooling over neighbors ---
        group_feats = group_feats.max(-1)[0] + group_feats.mean(-1)  # MaxPool + AvgPool
        return centers, group_feats.permute(0, 2, 1)  # (B, G, 3), (B, G, D)


class PointPNEncoder(nn.Module):
    """Parametric Encoder with hierarchical Local Geometry Aggregation (LGA)."""
    def __init__(self, in_dim=3, embed_dim=36, num_points=1024, alpha=1000, beta=100, dataset="sim"):
        super().__init__()
        self.dataset = dataset

        # initial point embedding
        self.raw_embed = build_shared_mlp([in_dim, embed_dim], conv_dim=1, final_act=True)

        #  build stages
        self.stage_params = [
            (embed_dim, embed_dim * 2, num_points // 2, 40, 2),
            (embed_dim * 2, embed_dim * 4, num_points // 4, 40, 1),
            (embed_dim * 4, embed_dim * 8, num_points // 8, 40, 1),
            (embed_dim * 8, embed_dim * 8, num_points // 16, 40, 1),
        ]
        self.stages = nn.ModuleList([
            LGA(in_dim, out_dim, num_group, group_size, blocks, alpha, beta)
            for in_dim, out_dim, num_group, group_size, blocks in self.stage_params
        ])
    
    def forward(self, xyz):  # (B, N, 3)
        feats = self.raw_embed(xyz.transpose(1, 2)).transpose(1, 2)  # (B, N, D)

        for stage in self.stages:
            xyz, feats = stage(xyz, feats, self.dataset)  # (B, G, D)

        global_feats = feats.max(1)[0] + feats.mean(1)
        return global_feats  # (B, D)
        