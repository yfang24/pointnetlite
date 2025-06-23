import torch
import torch.nn as nn

from models.modules.transformer_modules import TransformerDecoder

class PointMAEDecoder(nn.Module):
    def __init__(self, trans_dim=384, group_size=32, drop_path=0.1, depth=4, num_heads=6):
        super().__init__()
        self.group_size = group_size
        self.trans_dim = trans_dim

        self.mask_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.decoder = TransformerDecoder(
            embed_dim=trans_dim,
            depth=depth,
            drop_path=dpr,
            num_heads=num_heads,
        )

        self.reconstruction_head = nn.Sequential(
            nn.Conv1d(trans_dim, 3 * group_size, 1)  # Predict (S x 3) coordinates per group
        )

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x):
        """
        x: encoder output; a tuple of
            x_vis: (B, G_visible, C)        
            mask: (B, G) boolean mask
            neighborhood: (B, G, S, 3) - ground truth grouped points
            center: (B, G, 3)
        ------------------------------------------
        returns:
            - rebuilt_points: (B * G_masked, S, 3)
            - gt_points: (B * G_masked, S, 3)
        """
        x_vis, mask, neighborhood, center = x
        
        B, G, S, _ = neighborhood.shape
        C = self.trans_dim

        # Positional embeddings
        pos_vis = self.decoder_pos_embed(center[~mask].reshape(B, -1, 3))  # (B, G_visible, C)
        pos_mask = self.decoder_pos_embed(center[mask].reshape(B, -1, 3))  # (B, G_masked, C)

        # Mask token
        mask_token = self.mask_token.expand(B, pos_mask.shape[1], -1)      # (B, G_masked, C)

        # Concat visible + masked
        x_full = torch.cat([x_vis, mask_token], dim=1)                     # (B, G, C)
        pos_full = torch.cat([pos_vis, pos_mask], dim=1)                   # (B, G, C)

        # Decode
        x_rec = self.decoder(x_full, pos_full, pos_mask.shape[1])          # (B, G_masked, C)

        # Predict grouped points
        x_rec = x_rec.transpose(1, 2)                                       # (B, C, G_masked)
        pred = self.reconstruction_head(x_rec).transpose(1, 2)             # (B, G_masked, S*3)
        rebuild_points = pred.reshape(-1, self.group_size, 3)              # (B * G_masked, S, 3)

        # Ground truth masked points
        gt_points = neighborhood[mask].reshape(-1, S, 3)                    # (B * G_masked, S, 3)

        return rebuild_points, gt_points
