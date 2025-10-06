import functools
import math
import einops
import torch
import torch.nn.functional as F
from torch import nn

from esm.layers.rotary import RotaryEmbedding
from esm.layers.blocks import swiglu_correction_fn, gelu_ln_ffn, swiglu_ln_ffn


class SwiGLUFFNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        gate, lin = x.chunk(2, dim=-1)
        x = F.silu(gate) * lin

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class SwiGLUFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc_in(x)
        gate, lin = x.chunk(2, dim=-1)
        x = F.silu(gate) * lin

        x = self.fc_out(x)
        return x


class TransformerStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        scale_residue: bool = True,
        bias: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",
        expansion_ratio: float = 8 / 3,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    residue_scaling_factor=(
                        math.sqrt(n_layers / 36) if scale_residue else 1.0
                    ),
                    expansion_ratio=expansion_ratio,
                    bias=bias,
                    qk_layernorm=qk_layernorm,
                    ffn_type=ffn_type,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, sequence_id, attn_mask)

        return self.norm(x), x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x, seq_id=None, attention_mask=None):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)

        query_BLD, key_BLD = self.q_ln(query_BLD), self.k_ln(key_BLD)
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        n_heads = self.n_heads
        reshaper = functools.partial(
            einops.rearrange, pattern="b s (h d) -> b h s d", h=n_heads
        )
        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )

        if seq_id is not None:
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1).expand(-1, n_heads, -1, -1)

            if attention_mask is not None:
                attention_mask_BLL = attention_mask.unsqueeze(-1) == attention_mask.unsqueeze(-2)
                attention_mask_BHLL = attention_mask_BLL.unsqueeze(1).expand(-1, n_heads, -1, -1)
                mask_BHLL = mask_BHLL & attention_mask_BHLL

            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, mask_BHLL
            )
        else:
            if attention_mask is not None:
                attention_mask_BLL = attention_mask.unsqueeze(-1) == attention_mask.unsqueeze(-2)
                attention_mask_BHLL = attention_mask_BLL.unsqueeze(1).expand(-1, n_heads, -1, -1)
            else:
                attention_mask_BHLL = None

            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, attention_mask_BHLL
            )

        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
        return self.out_proj(context_BLD)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_model, n_heads, bias, qk_layernorm=qk_layernorm
        )

        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.scaling_factor = residue_scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        r1 = self.attn(x, sequence_id, attn_mask)
        x = x + r1 / self.scaling_factor

        r2 = self.ffn(x)
        x = x + r2 / self.scaling_factor

        return x