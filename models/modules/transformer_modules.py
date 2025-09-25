import torch
import torch.nn as nn
from timm.layers import DropPath

class FeedForward(nn.Module):
    '''
    feed forward layer in transformer block:
        FFN(x) = Dropout(W_2(Activation(W_1x)))
    '''
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

#===========
# Encoder
#===========
class SelfAttention(nn.Module):
    '''
    multi-head self-attention:
        heads' attn_out are concatenated and merged via a linear proj
    '''
    def __init__(self, embed_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)     # (3, B, H, N, D_head)
        q, k, v = qkv[0], qkv[1], qkv[2]                          # (B, H, N, D_head=D//H)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale             # (B, H, N, N)

        if attn_mask is not None:
            # attn_mask shape: (N, N) or (B, 1, N, N)
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))   # -inf ensures after softmax, the probability becomes zero
        
        attn = self.attn_drop(attn.softmax(dim=-1))
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)           # (B, H, N, D_head) -> (B, N, D)
        return self.proj_drop(self.proj(x))                       # (B, N, D)


class EncoderBlock(nn.Module):
    '''
    x -> attn module: layernorm + attn -> residual -> ffn module: laynorm + ffn -> residual -> output
    '''
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.):
        '''
        droppath: attn/ff module might be skipped
        '''
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)        

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, int(embed_dim * mlp_ratio), drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # Self-attention
        x = x + self.drop_path(self.attn(self.norm1(x)))          # (B, N, D)
        # Feed-forward
        x = x + self.drop_path(self.ffn(self.norm2(x)))           # (B, N, D)
        return x


class TransformerEncoder(nn.Module):
    '''
    Transformer encoder: a stack of Transformer encoder blocks
    '''
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                  drop, attn_drop, drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)
            ])
        self.norm = nn.LayerNorm(embed_dim)
            
    def forward(self, x, pos=0):  # (B, N, D); token (point group) embedding + positional encoding
        # most common in ViTs, BERT
        # x = x + pos
        # for block in self.blocks:
        #     x = block(x)

        # used in some vision/3D transformers, like pointmae, where position is reinforced at each layer
        for block in self.blocks:
            x = block(x + pos)

        return self.norm(x)

#===========
# Decoder
#===========
class CrossAttention(nn.Module):
    """
    Multi-head cross-attention:
        Queries from one sequence (decoder), keys/values from another (encoder).
    """
    def __init__(self, embed_dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_input, kv_input):
        """
        q_input: (B, Nq, D)  queries (decoder tokens)
        kv_input: (B, Nk, D) keys/values (encoder output)
        """
        B, Nq, D = q_input.shape
        Nk = kv_input.size(1)

        # Linear projections
        q = self.q_proj(q_input).reshape(B, Nq, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)  # (B,H,Nq,Dh)
        k = self.k_proj(kv_input).reshape(B, Nk, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)  # (B,H,Nk,Dh)
        v = self.v_proj(kv_input).reshape(B, Nk, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)  # (B,H,Nk,Dh)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale              # (B, H, Nq, Nk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, D)         # (B, Nq, D)

        return self.proj_drop(self.proj(out))

class DecoderBlock(nn.Module):
    """
    x -> self-attn module: layernorm + masked self-attn -> residual
      -> cross-attn module: layernorm + cross-attn with encoder -> residual
      -> ffn module: laynorm + ffn -> residual -> output
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        # 1. Masked self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = SelfAttention(embed_dim, num_heads, qkv_bias, qk_scale,
                                       attn_drop=attn_drop, proj_drop=drop)

        # 2. Cross-attention (decoder queries attend to encoder outputs (keys/values))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(embed_dim, num_heads, qkv_bias, qk_scale,
                                         attn_drop=attn_drop, proj_drop=drop)
        
        # 3. Feed-forward network
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, int(embed_dim * mlp_ratio), drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, memory, attn_mask=None):
        """
        x: (B, N_dec, D)   decoder input
        memory: (B, N_enc, D) encoder outputs
        attn_mask: optional mask for causal attention
            attn_mask =  causal_mask(N_dec).to(x.device)
        """
        x = x + self.drop_path(self.self_attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop_path(self.cross_attn(self.norm2(x), memory))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x
    
    def causal_mask(size):
        '''
        for autoregressive decoding (e.g., GPT, seq2seq text decoders), so that token i cannot "peek" at tokens j > i (avoid seeing future tokens)
        '''
        # lower-triangular mask (1 = keep, 0 = block)
        mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
        return mask

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder: a stack of Transformer decoder blocks
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                         drop, attn_drop, drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, memory, pos_dec=0, pos_enc=0, attn_mask=None):
        """
        x: (B, N_dec, D)   decoder input tokens
        memory: (B, N_enc, D) encoder output tokens
        """
        x = x + pos_dec
        memory = memory + pos_enc

        for block in self.blocks:
            x = block(x, memory, attn_mask=attn_mask)

        return self.norm(x)
