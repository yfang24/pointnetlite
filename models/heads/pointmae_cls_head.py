import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from models.modules.builders import build_fc_layers

class PointMAEClsHead(nn.Module):
    def __init__(self, embed_dim=384, out_dim=40):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))  # [cls] token: serve as a global aggregator — attention layers are trained so that information from all tokens flows into this token, which then represents the whole object/scene.
        self.cls_pos = nn.Parameter(torch.randn(1, embed_dim))
        
        self.cls_head = build_fc_layers([embed_dim * 2, 256, 256, out_dim], linear_bias=True, act=nn.ReLU(inplace=True), dropout=0.5)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x): # (B, G, embed_dim); group token embeddings
        cls_feature = (self.cls_token + self.cls_pos).expand(x.size(0), -1)  # (B, embed_dim); cls_token_embedding = cls_token (B, embed_dim) + its pos; parametric, trainable global summary (depends on attention, relies on transformer dynamics)
        global_feature = x.max(dim=1)[0]    # (B, embed_dim); maxpool over groups; deterministic, a hard pooled global summary computed directly from the token embeddings (without relying on CLS attention)

        feat = torch.cat([cls_feature, global_feature], dim=1)  # (B, 2*embed_dim)
        return self.cls_head(feat)                  # (B, out_dim)
