import torch
import torch.nn as nn

from models.modules.transformer_modules import TransformerEncoder

class PointMAEDecoder(nn.Module):
    def __init__(self, embed_dim=384, group_size=32, drop_path=0.1, depth=4, num_heads=6):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.decoder = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            drop_path=dpr,
            num_heads=num_heads,
        )

        self.reconstruction_head = nn.Sequential(
            nn.Conv1d(embed_dim, 3 * group_size, 1)  # Predict (S x 3) coordinates per group
        )

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x):
        """
        x: encoder output; a tuple of
            x_vis: (B, G_visible, D)        
            mask: (B, G) boolean mask
            neighborhood: (B, G, S, 3) - ground truth grouped points
            center: (B, G, 3)
        ------------------------------------------
        returns:
            - rec_points: (B * G_masked, S, 3)
            - gt_points: (B * G_masked, S, 3)
        """
        x_vis, mask, neighborhood, center = x        
        B, _, S, _ = neighborhood.shape       
        mask_len = int(mask.sum(dim=1)[0].item())   # (B,)

        # Mask token
        x_mask = self.mask_token.expand(B, mask_len, -1)    # (B, G_masked, D)

        # Concat visible + masked
        x_group = torch.cat([x_vis, x_mask], dim=1)   # (B, G, D)

        # Decode
        group_pos = self.pos_embed(center)  # (B, G, D)
        group_token = self.decoder(x_group, group_pos)    # (B, G, D)
        rec_token = group_token[:, -mask_len:]    # (B, G_mask, D); the last mask_len elements correspond to x_mask
        
        # Predict grouped points
        rec_token = rec_token.transpose(1, 2)                                      # (B, D, G_masked)
        rec_points = self.reconstruction_head(rec_token).transpose(1, 2)             # (B, G_masked, S*3)
        rec_points = rec_points.reshape(-1, S, 3)              # (B * G_masked, S, 3)

        # Ground truth masked points
        gt_points = neighborhood[mask].reshape(-1, S, 3)                    # (B * G_masked, S, 3)

        return rec_points, gt_points
