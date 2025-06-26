import torch
import torch.nn as nn
from timm.layers import DropPath

from models.modules.builders import build_fc_layers

# feed forward layer in transformer block
class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, drop=0.):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim

        
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))                                       # (B, N, hidden)
        x = self.drop(self.fc2(x))                                                 # (B, N, out)
        return x


# multi-head self-attention; heads' attn_out are concatenated and merged via a linear proj
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                          # (B, H, N, D_head=D//H)
        attn = (q @ k.transpose(-2, -1)) * self.scale             # (B, H, N, N)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)           # (B, H, N, D_head) -> (B, N, D)
        return self.proj_drop(self.proj(x))                       # (B, N, D)


# x -> attn module: layernorm+attn -> residual -> ff module: laynorm+mlp -> residual -> output
# droppath: attn/ff module might be skipped
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))                          # (B, N, D)
        x = x + self.drop_path(self.mlp(self.norm2(x)))                          # (B, N, D)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                  drop, attn_drop, drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x, pos):  # (B, N, D); point group embedding + positional encoding
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x
