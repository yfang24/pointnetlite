import torch
import torch.nn as nn

from utils.pcd_utils import pointmae_group
from models.modules.transformer_modules import TransformerEncoder

# encode point cloud to 1024dims using pointnet-like encoder
class PointGroupEncoder(nn.Module):
    def __init__(self, encoder_channel=1024):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1)
        )

    def forward(self, x):
        B, G, N, _ = x.shape                                       # (B, G, N, 3)
        x = point_groups.view(B * G, N, 3).transpose(2, 1)         # (BG, 3, N)
        x = self.first_conv(x)                                     # (BG, 256, N)
        x_global = torch.max(x, dim=2, keepdim=True)[0]            # (BG, 256, 1)
        x = torch.cat([x_global.expand(-1, -1, N), x], dim=1)      # (BG, 512, N)
        x = self.second_conv(x)                                    # (BG, C, N)
        x = torch.max(x, dim=2)[0]                                 # (BG, C)
        return x.view(B, G, self.encoder_channel)                  # (B, G, C)
        
        
class PointMAEEncoder(nn.Module):
    def __init__(self, encoder_dims=384, trans_dim=384, depth=12, drop_path_rate=0.1,
                 num_heads=6, mask_ratio=0.6, mask_type='rand', group_size=32, num_group=64):
        super().__init__()        
        self.group_size = group_size
        self.num_group = num_group
        self.mask_ratio = mask_ratio
        self.trans_dim = trans_dim
        self.mask_type = mask_type
        
        self.encoder = PointGroupEncoder(encoder_channel=encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = TransformerEncoder(
            embed_dim=trans_dim,
            depth=depth,
            drop_path_rate=dpr,
            num_heads=num_heads,
        )
        self.norm = nn.LayerNorm(trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_rand(self, center, noaug=False):
        B, G, _ = center.shape  # (B, G, 3)
        if noaug or self.mask_ratio == 0: # skip the mask
            return torch.zeros(B, G, dtype=torch.bool, device=center.device)
    
        num_mask = int(self.mask_ratio * G)
        mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
        for i in range(B):
            perm = torch.randperm(G, device=center.device)
            mask[i, perm[:num_mask]] = True
        return mask # (B, G)-bool

    def _mask_center_block(self, center, noaug=False):
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(B, G, dtype=torch.bool, device=center.device)
    
        mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
        for i in range(B):
            ref_idx = torch.randint(0, G, (1,), device=center.device)
            ref_point = center[i, ref_idx, :]  # (1, 3)
            dists = torch.norm(center[i] - ref_point, dim=-1)  # (G,)
            sorted_idx = torch.argsort(dists)
            num_mask = int(self.mask_ratio * G)
            mask[i, sorted_idx[:num_mask]] = True
        return mask  # (B, G)
   
    def forward(self, point_cloud, noaug=False): # (B, N, 3)
        neighborhood, center = pointmae_group(point_cloud, self.num_group, self.group_size)  # (B, G, S, 3), (B, G, 3)

        if self.mask_type == 'rand':
            mask = self._mask_center_rand(center, noaug)
        else:
            mask = self._mask_center_block(center, noaug)

        group_tokens = self.encoder(neighborhood)                # (B, G, C)
        B, G, C = group_tokens.shape
        x_vis = group_tokens[~mask].reshape(B, -1, C)            # (B, G_visible, C)

        pos = self.pos_embed(center[~mask].reshape(B, -1, 3))    # (B, G_visible, trans_dim)
        x_vis = self.blocks(x_vis, pos)                          # (B, G_visible, trans_dim)
        x_vis = self.norm(x_vis)

        return x_vis, mask, neighborhood, center