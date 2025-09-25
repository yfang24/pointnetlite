import torch
import torch.nn as nn

from utils.pcd_utils import fps, knn_group, group_points
from models.modules.transformer_modules import TransformerEncoder
from models.modules.builders import build_shared_mlp

# class PointGroupEncoder(nn.Module):  # pointnetlite
#     def __init__(self, in_dim=3, embed_dim=1024):
#         super().__init__()
#         self.mlp = build_shared_mlp([in_dim] + [64, 128] + [embed_dim], conv_dim=2, act=nn.ReLU(inplace=True), final_act=False)
        
#     def forward(self, x):  # (B, G, N, 3)
#         x = x.permute(0, 3, 2, 1)  # (B, C_in, k, G)
#         x = self.mlp(x).max(dim=2)[0]  # (B, C_out, k, G), max over neighborhood k -> (B, C_out, G) 
#         x = x.permute(0, 2, 1)  # (B, G, C_out)
#         return x

class PointGroupEncoder(nn.Module): # original pointmae's
    def __init__(self, in_dim=3, embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_dim, 128, 1),
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

class PointM2AEEncoder(nn.Module):
    def __init__(self, embed_dims=[256, 512], depths=[2, 4], drop_path=0.1, num_heads=8,
                 mask_ratio=0.6, group_sizes=[32, 16], num_groups=[64, 32], local_radius=[0.2, 0.4],
                 noaug=False):  # noaug: whether do masking--False for pretrain, True for finetune
        super().__init__()
        self.group_sizes = group_sizes
        self.num_groups = num_groups
        self.local_radius = local_radius
        self.mask_ratio = mask_ratio
        self.noaug = noaug

        self.num_stages = len(embed_dims)

        self.encoders = nn.ModuleList([
            PointGroupEncoder(in_dim=3 if i == 0 else embed_dims[i - 1], embed_dim=embed_dims[i])
            for i in range(self.num_stages)
        ])

        self.pos_embeds = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, embed_dims[i]),
                nn.GELU(),
                nn.Linear(embed_dims[i], embed_dims[i])
            ) for i in range(self.num_stages)
        ])

        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.transformers = nn.ModuleList()
        start = 0
        for i in range(self.num_stages):
            self.transformers.append(TransformerEncoder(
                embed_dim=embed_dims[i],
                depth=depths[i],
                drop_path=dpr[start:start + depths[i]],
                num_heads=num_heads
            ))
            start += depths[i]

        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dims[i]) for i in range(self.num_stages)
        ])

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

    def _local_att_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)  # (B, G, G)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius  # (B, G, G) — True = masked out
        return mask, dist

    def forward(self, x):  # (B, N, 3)
        neighborhoods, centers, idxs = [], [], []
        pc = x
        for i in range(self.num_stages):
            center = fps(pc, self.num_groups[i])  # (B, G, 3)
            idx = knn_group(pc, center, self.group_sizes[i])  # (B, G, S)
            neighborhood = group_points(pc, idx)  # (B, G, S, 3)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)
            pc = center  # next level input
        
        if self.noaug:
            # Finetune mode — no masking
            x = None
            xyz_dist = None
            for i in range(self.num_stages):
                if i == 0:
                    x = self.encoders[i](neighborhoods[i])
                else:
                    B, G1, _ = x.shape
                    B, G2, K2, _ = neighborhoods[i].shape
                    feat = x.view(B * G1, -1)[idxs[i].view(-1)].view(B, G2, K2, -1)
                    x = self.encoders[i](feat)

                pos = self.pos_embeds[i](centers[i])
                attn_mask = None
                if self.local_radius[i] > 0:
                    attn_mask, xyz_dist = self._local_att_mask(centers[i], self.local_radius[i], xyz_dist)

                x = self.transformers[i](x, pos, attn_mask)
                x = self.norms[i](x)
            return x  # (B, G, D)
        else:
            # Pretrain mode — with masking
            bool_masked_pos = []
            bool_masked_pos.append(self._mask_center_rand(centers[-1]))  # final scale

            for i in reversed(range(self.num_stages - 1)):
                B, G, K, _ = neighborhoods[i + 1].shape
                idx = idxs[i + 1].reshape(B * G, -1)
                idx_masked = (~bool_masked_pos[-1].reshape(-1, 1)) * idx
                idx_masked = idx_masked.reshape(-1).long()
                masked_pos = torch.ones(B * centers[i].shape[1], device=centers[i].device)
                masked_pos.scatter_(0, idx_masked, 0)
                bool_masked_pos.append(masked_pos.view(B, -1).bool())

            bool_masked_pos.reverse()
            x_vis_list = []
            mask_vis_list = []
            xyz_dist = None

            for i in range(self.num_stages):
                if i == 0:
                    x_full = self.encoders[i](neighborhoods[i])
                else:
                    B, G1, _ = x.shape
                    B, G2, K2, _ = neighborhoods[i].shape
                    x_feat = x.reshape(B * G1, -1)[idxs[i].reshape(-1)].reshape(B, G2, K2, -1)
                    x_full = self.encoders[i](x_feat)

                bool_vis_pos = ~bool_masked_pos[i]
                vis_lens = bool_vis_pos.sum(dim=1)
                max_len = vis_lens.max()
                D = x_full.size(-1)

                x_vis = torch.zeros(B, max_len, D, device=x_full.device)
                center_vis = torch.zeros(B, max_len, 3, device=x_full.device)
                mask_vis = torch.ones(B, max_len, max_len, device=x_full.device)

                for bz in range(B):
                    vis_tokens = x_full[bz][bool_vis_pos[bz]]
                    vis_centers = centers[i][bz][bool_vis_pos[bz]]
                    x_vis[bz, :vis_lens[bz]] = vis_tokens
                    center_vis[bz, :vis_lens[bz]] = vis_centers
                    mask_vis[bz, :vis_lens[bz], :vis_lens[bz]] = 0

                pos = self.pos_embeds[i](center_vis)

                if self.local_radius[i] > 0:
                    attn_mask, xyz_dist = self._local_att_mask(center_vis, self.local_radius[i], xyz_dist)
                    mask_vis_att = mask_vis * attn_mask
                else:
                    mask_vis_att = mask_vis

                x = self.transformers[i](x_vis, pos, mask_vis_att)
                x_vis_list.append(self.norms[i](x))
                mask_vis_list.append(~mask_vis[:, :, 0].bool())

                if i < self.num_stages - 1:
                    x_full[bool_vis_pos] = x[~mask_vis[:, :, 0].bool()]
                    x = x_full

            return x_vis_list, mask_vis_list, bool_masked_pos, neighborhoods, centers