import torch
import torch.nn as nn
import torch.nn.functional as F
from .rotary import RotaryEmbedding


class AttentionWithRelativePosition(nn.Module):
    def __init__(self, dim, heads=8, max_relative_distance=128):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.max_relative_distance = max_relative_distance

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.relative_position_embedding = nn.Embedding(2 * max_relative_distance - 1, heads)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, attention_mask=None):
        B, T, L, D = x.shape
        h = self.heads

        x_flat = x.view(B * T, L, D)
        qkv = self.to_qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B * T, L, h, self.dim_head)
        k = k.view(B * T, L, h, self.dim_head)
        v = v.view(B * T, L, h, self.dim_head)

        attn_weights = torch.einsum('blhd,bshe->bhls', q, k) * self.scale

        range_vec = torch.arange(L, device=x.device)
        relative_positions = range_vec[:, None] - range_vec[None, :]
        relative_positions = torch.clamp(
            relative_positions,
            min=-self.max_relative_distance + 1,
            max=self.max_relative_distance - 1
        )
        relative_positions += self.max_relative_distance - 1

        relative_bias = self.relative_position_embedding(relative_positions)
        relative_bias = relative_bias.permute(2, 0, 1)
        attn_weights += relative_bias.unsqueeze(0)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(B, T, L).reshape(B * T, 1, 1, L)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        out = torch.einsum('bhls,bshe->blhd', attn_weights, v)
        out = out.reshape(B * T, L, D)
        out = self.to_out(out)
        out = out.view(B, T, L, D)

        return out


class AttentionWithRotary(nn.Module):
    def __init__(self, dim, heads=8, qk_layernorm=True):
        super().__init__()
        self.heads = heads
        self.dim_heads = dim // heads
        self.scale = self.dim_heads ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.norm_qkv = nn.LayerNorm(dim)

        if qk_layernorm:
            self.norm_q = nn.LayerNorm(dim)
            self.norm_k = nn.LayerNorm(dim)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        self.rotary = RotaryEmbedding(self.dim_heads)

    def _apply_rotary(self, q, k):
        q = q.unflatten(-1, (self.heads, self.dim_heads))
        k = k.unflatten(-1, (self.heads, self.dim_heads))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x, attention_mask=None):
        B, T, L, D = x.shape
        h = self.heads

        x_flat = x.view(B * T, L, D)
        qkv = self.to_qkv(self.norm_qkv(x_flat))
        q, k, v = qkv.chunk(3, dim=-1)

        q, k = self.norm_q(q), self.norm_k(k)
        q, k = self._apply_rotary(q, k)

        q = q.view(B * T, L, h, self.dim_heads)
        k = k.view(B * T, L, h, self.dim_heads)
        v = v.view(B * T, L, h, self.dim_heads)

        attn_weights = torch.einsum('blhd,bshd->bhls', q, k) * self.scale

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(B, T, L).reshape(B * T, 1, 1, L)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        out = torch.einsum('bhls,bshd->blhd', attn_weights, v)
        out = out.reshape(B * T, L, D)
        out = self.to_out(out)
        out = out.view(B, T, L, D)

        return out
