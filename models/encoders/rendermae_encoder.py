import torch
import torch.nn as nn

from utils.pcd_utils import fps, knn_group, group_points
from models.modules.transformer_modules import TransformerEncoder
from models.modules.builders import build_shared_mlp

class PointEncoderMLP(nn.Module):
    def __init__(self, in_dim=3, embed_dim=384):
        super().__init__()
        self.mlp = build_shared_mlp([in_dim, 128, 256, embed_dim], conv_dim=1)

    def forward(self, x):
        # x: (B, N, 3)
        x = x.transpose(1, 2)        # (B, 3, N)
        x = self.mlp(x)              # (B, D, N)
        return x.transpose(1, 2)     # (B, N, D)


class RenderMAEEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=12, drop_path=0.1, num_heads=6, noaug=False):  # noaug: whether do masking--False for pretrain, True for finetune
        super().__init__()
        self.noaug = noaug

        self.point_encoder = PointEncoderMLP(in_dim=3, embed_dim=embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            drop_path=dpr,
            num_heads=num_heads,
        )
        self.norm = nn.LayerNorm(embed_dim)

    # def forward(self, vis_pts):
    def forward(self, x):
        vis_pts, mask_pts, reflected_pts = x
        
        """
        Args:
            vis_pts: (B, N, 3)
        Returns:
            vis_token: (B, N, D) - encoded visible tokens
        """
        # vis_embed = self.point_encoder(vis_pts)        # (B, N, D)

        centers = fps(vis_pts, M)  # (B, M, 3)
        grouped_pts = group_points(vis_pts, centers, K=32)  # (B, M, K, 3)
        vis_embed = self.point_encoder(grouped_pts)  # (B, M, D)

        #------------------
        vis_pos = self.pos_embed(vis_pts)             # (B, N, D)

        vis_token = self.blocks(vis_embed, vis_pos)          # (B, N, D)
        vis_token = self.norm(vis_token)

        if self.noaug:
            return vis_token
        return vis_token, vis_pts, mask_pts, reflected_pts
