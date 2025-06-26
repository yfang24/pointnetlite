import torch
import torch.nn as nn

from models.modules.transformer_modules import TransformerEncoder
from models.modules.builders import build_shared_mlp
from utils.pcd_utils import fps, knn_group, group_points

class RenderMAEDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, drop_path=0.1, num_heads=6, group_size=32):
        super().__init__()
        self.group_size = group_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

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
        
        # self.rec_head = build_shared_mlp([embed_dim, 256, 128, out_dim], conv_dim=1)
        self.rec_head = nn.Sequential(
            nn.Conv1d(embed_dim, 3 * group_size, 1)  # Predict (S x 3) coordinates per group
        )

    def forward(self, vis_token, vis_centers, mask_pts):
        B, G, D = vis_token.shape
        S = self.group_size
        
        mask_centers = fps(mask_pts, G)  # (B, G, 3)
        mask_group = group_points(mask_pts, idx=knn_group(mask_pts, mask_centers, S))  - mask_centers.unsqueeze(2) # (B, G, S, 3)
        mask_group = mask_group.reshape(-1, S, 3)  # (B*G, S, 3)
        
        vis_pos = self.pos_embed(vis_centers)        
        mask_pos = self.pos_embed(mask_centers)
        full_pos = torch.cat([vis_pos, mask_pos], dim=1)  # (B, 2G, D)

        mask_token = self.mask_token.expand(B, G, D)
        full_token = torch.cat([vis_token, mask_token], dim=1)

        full_embed = self.blocks(full_token, full_pos)   # (B, 2G, D)
        full_embed = self.norm(full_embed)

        pred_embed = full_embed[:, -G:, :]  # (B, G, D)
        pred_embed = pred_embed.transpose(1, 2)
        pred_group = self.rec_head(pred_embed).transpose(1, 2)  # (B, G, S*3)
        pred_group = pred_group.reshape(-1, S, 3)  # (B*G, S, 3)
        
        return pred_group, mask_group
