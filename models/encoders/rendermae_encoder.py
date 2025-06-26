import torch
import torch.nn as nn

from utils.pcd_utils import fps, knn_group, group_points
from models.modules.transformer_modules import TransformerEncoder
from models.modules.builders import build_shared_mlp

class PointEncoderMLP(nn.Module):
    def __init__(self, in_dim=3, embed_dim=384):
        super().__init__()
        self.mlp = build_shared_mlp([in_dim] + [64, 128] + [embed_dim], conv_dim=2, act=nn.ReLU(inplace=True), final_act=False)
        
    def forward(self, x):  # (B, G, N, 3)
        x = x.permute(0, 3, 2, 1)  # (B, C_in, k, G)
        x = self.mlp(x).max(dim=2)[0]  # (B, C_out, k, G), max over neighborhood k -> (B, C_out, G) 
        x = x.permute(0, 2, 1)  # (B, G, C_out)
        return x


class RenderMAEEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=12, drop_path=0.1, num_heads=6, 
                 group_size=32, num_group=64, noaug=False):
        super().__init__()
        self.group_size = group_size
        self.num_group = num_group
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

    def forward(self, vis_pts):    
        """
        Args:
            vis_pts: (B, N, 3)
        Returns:
            vis_token: (B, G, D) - encoded visible tokens
        """
        vis_centers = fps(vis_pts, self.num_group)  # (B, G, 3)
        vis_groups = group_points(vis_pts, idx=knn_group(vis_pts, vis_centers, self.group_size))  - vis_centers.unsqueeze(2) # (B, G, S, 3)
        
        vis_embed = self.point_encoder(vis_groups)        # (B, G, D)
        vis_pos = self.pos_embed(vis_centers)             # (B, G, D)

        vis_token = self.blocks(vis_embed, vis_pos)          # (B, G, D)
        vis_token = self.norm(vis_token)

        if self.noaug:
            return vis_token
        return vis_token, vis_centers
        
