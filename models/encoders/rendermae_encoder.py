import torch
import torch.nn as nn

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
    def __init__(self, embed_dim=384, depth=12, drop_path=0.1, num_heads=6):
        super().__init__()

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
            drop_path_rate=dpr,
            num_heads=num_heads,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, vis_pts):
        """
        Args:
            vis_pts: (B, N, 3)
        Returns:
            x_vis: (B, N, D) - encoded visible tokens
            mask_pos: (B, N, D) - positional encoding of reflected (masked) points
        """
        feat_vis = self.point_encoder(vis_pts)        # (B, N, D)
        pos_vis = self.pos_embed(vis_pts)             # (B, N, D)

        x_vis = self.blocks(feat_vis, pos_vis)          # (B, N, D)
        x_vis = self.norm(x_vis)
        return x_vis
