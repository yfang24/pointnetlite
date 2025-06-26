import torch
import torch.nn as nn

from models.modules.transformer_modules import TransformerEncoder
from models.modules.builders import build_shared_mlp

class RenderMAEDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, drop_path=0.1, num_heads=6, out_dim=3):
        super().__init__()

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
            drop_path_rate=dpr,
            num_heads=num_heads,
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.rec_head = build_shared_mlp([embed_dim, 256, 128, out_dim], conv_dim=1)

    # def forward(self, vis_token, vis_pts, reflected_pts):
    def forward(self, x):
        vis_token, vis_pts, mask_pts, reflected_pts = x
        
        """
        Args:
            vis_token: (B, N_visible, D) - encoded visible tokens
            vis_pts: (B, N_visible, 3) - input visible points (for position encoding)
            reflected_pts: (B, N_mask, 3) - reflected visible points (for mask position encoding)
        Returns:
            pred_pts: (B, N_mask, 3) - reconstructed masked points
        """
        B, N_mask, _ = reflected_pts.shape
        D = vis_token.shape[-1]

        vis_pos = self.pos_embed(vis_pts)             # (B, N_visible, D)
        mask_pos = self.pos_embed(reflected_pts)      # (B, N_mask, D)        
        mask_token = self.mask_token.expand(B, N_mask, D)  # (B, N_mask, D)
        
        x_full = torch.cat([vis_token, mask_token], dim=1)   # (B, N_all, D)
        pos_full = torch.cat([vis_pos, mask_pos], dim=1)     # (B, N_all, D)

        x = self.blocks(x_full, pos_full)                    # (B, N_all, D)
        x = self.norm(x)

        pred_mask = x[:, -N_mask:, :]                        # (B, N_mask, D)
        pred_mask = pred_mask.transpose(1, 2)                # (B, D, N_mask)
        pred_pts = self.rec_head(pred_mask).transpose(1, 2)  # (B, N_mask, 3)

        # return pred_pts
        return pred_pts, mask_pts
