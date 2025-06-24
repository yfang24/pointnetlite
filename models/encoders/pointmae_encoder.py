import torch
import torch.nn as nn

from utils.pcd_utils import sample_and_group
from models.modules.transformer_modules import TransformerEncoder

# encode point cloud to 1024dims using pointnet-like encoder
class PointGroupEncoder(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
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
            nn.Conv1d(512, embed_dim, 1)
        )

    def forward(self, x):
        B, G, N, _ = x.shape                                       # (B, G, N, 3)
        x = x.view(B * G, N, 3).transpose(2, 1)                    # (BG, 3, N)
        x = self.first_conv(x)                                     # (BG, 256, N)
        x_global = torch.max(x, dim=2, keepdim=True)[0]            # (BG, 256, 1)
        x = torch.cat([x_global.expand(-1, -1, N), x], dim=1)      # (BG, 512, N)
        x = self.second_conv(x)                                    # (BG, D, N)
        x = torch.max(x, dim=2)[0]                                 # (BG, D)
        return x.view(B, G, self.embed_dim)                  # (B, G, D)
        
        
class PointMAEEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=12, drop_path=0.1, num_heads=6, 
                 mask_ratio=0.6, mask_type='rand', group_size=32, num_group=64, 
                 noaug=False):  # noaug: whether do masking--False for pretrain, True for finetune
        super().__init__()        
        self.group_size = group_size
        self.num_group = num_group
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.mask_type = mask_type
        self.noaug = noaug
        
        self.encoder = PointGroupEncoder(embed_dim=embed_dim)
                     
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

    # for each batch, randomly select num_mask centers
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

    # for each batch, randomly select a ref center and other closest (num_mask-1) centers
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
   
    def forward(self, point_cloud): # (B, N, 3)
        neighborhood, center = sample_and_group(point_cloud, self.num_group, self.group_size)  # (B, G, S, 3), (B, G, 3)

        noaug = self.noaug
        if self.mask_type == 'rand':
            mask = self._mask_center_rand(center, noaug)
        else:
            mask = self._mask_center_block(center, noaug)

        group_tokens = self.encoder(neighborhood)                # (B, G, D)
        B, G, D = group_tokens.shape
        x_vis = group_tokens[~mask].reshape(B, -1, D)            # (B, G_visible, D)

        pos = self.pos_embed(center[~mask].reshape(B, -1, 3))    # (B, G_visible, D)
        x_vis = self.blocks(x_vis, pos)                          # (B, G_visible, D)
        x_vis = self.norm(x_vis)

        if noaug:
            return x_vis
        return x_vis, mask, neighborhood, center
