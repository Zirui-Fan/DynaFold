import torch
import torch.nn as nn
import os
import logging
import torch.nn.functional as F

from .attention import AttentionWithRotary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeqEmbedder(nn.Module):
    def __init__(self, aa_nums, seq_dim, latent_dim):
        super(SeqEmbedder, self).__init__()
        self.aa_nums = aa_nums
        self.aa_embed = nn.Embedding(aa_nums, latent_dim)
        self.seq_embed = nn.Linear(seq_dim, latent_dim)
        self.token_embed = nn.Linear(seq_dim, latent_dim)
        self.norm_token = nn.LayerNorm(seq_dim)
        self.norm_seq = nn.LayerNorm(seq_dim)

    def forward(self, aa_types, seq_rep):
        seq_embed = self.seq_embed(self.norm_seq(seq_rep.mean(1, keepdim=True)))
        token_embed = self.token_embed(self.norm_token(seq_rep))
        aa_embed = self.aa_embed(aa_types)
        seq_rep = seq_embed + token_embed + aa_embed
        return seq_rep


class TimeStepEmbedding(nn.Module):
    def __init__(self, embed_dim, time_step_dim=1):
        super(TimeStepEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_step_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 9 * embed_dim)
        )

    def forward(self, time_step):
        params = self.mlp(time_step.view(time_step.shape[0], 1, 1, 1))
        at, bt, gt, al, bl, gl, am, bm, gm = torch.chunk(params, 9, dim=-1)
        return at, bt, gt, al, bl, gl, am, bm, gm


class CondTrajEmbedder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(CondTrajEmbedder, self).__init__()
        self.condition_norm = nn.LayerNorm(input_dim)
        self.frame_norm = nn.LayerNorm(input_dim)

        self.cond_traj_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.Linear(latent_dim // 2, latent_dim)
        )

        self.cond_frame_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.Linear(latent_dim // 2, latent_dim)
        )

        self.mask_embed = nn.Embedding(2, latent_dim)

    def forward(self, condition_frames, condition_mask):
        condition_mask_expanded = condition_mask.unsqueeze(-1).unsqueeze(-1)
        condition_frames_masked = condition_frames * condition_mask_expanded

        cond_frame_proj = self.cond_frame_proj(self.frame_norm(condition_frames_masked))
        cond_frames_proj = (cond_frame_proj * condition_mask_expanded).sum(dim=1) / condition_mask_expanded.sum(dim=1)

        cond_traj_proj = (
            self.cond_traj_proj(self.condition_norm(condition_frames_masked)) +
            self.mask_embed(condition_mask.to(torch.int)).unsqueeze(2)
        )

        return cond_frames_proj, cond_traj_proj


class GEGLUMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(GEGLUMLP, self).__init__()
        self.proj_in = nn.Linear(embed_dim, hidden_dim * 2)
        self.gelu = nn.GELU()
        self.proj_out = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x_proj = self.proj_in(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        x_glu = self.gelu(x1) * x2
        out = self.proj_out(x_glu)
        return out


class SpatioTemporalTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim=None,
        spatial_attn_cls=AttentionWithRotary,
        temporal_attn_cls=AttentionWithRotary,
        use_temporal_attn=True
    ):
        super(SpatioTemporalTransformerLayer, self).__init__()
        self.use_temporal_attn = use_temporal_attn

        if mlp_dim is None:
            mlp_dim = embed_dim * 2

        self.spatial_attn = spatial_attn_cls(embed_dim, num_heads)

        if self.use_temporal_attn:
            self.temporal_attn = temporal_attn_cls(embed_dim, num_heads)

        self.mlp = GEGLUMLP(embed_dim, mlp_dim)
        self.prot_mlp = GEGLUMLP(embed_dim, embed_dim)
        self.integrate = nn.Linear(embed_dim * 2, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)

        self.time_adaln = TimeStepEmbedding(embed_dim)

    def check_for_nan(self, tensor, operation_name):
        if torch.isnan(tensor).any():
            nan_indices = torch.nonzero(torch.isnan(tensor), as_tuple=False)
            if nan_indices.numel() > 0:
                first_nan = nan_indices[0]
                logger.error(f"NaN detected after {operation_name} at index {tuple(first_nan.tolist())}")
            else:
                logger.error(f"NaN detected after {operation_name}, but could not locate the index.")
            raise ValueError(f"NaN detected after {operation_name}")

    def forward(self, x, t, prot_rep, attn_mask=None, temporal_mask=None):
        at, bt, gt, al, bl, gl, am, bm, gm = self.time_adaln(t)

        x = x + gl * self.spatial_attn(al * self.norm1(x) + bl)

        if self.use_temporal_attn:
            x = x + gt * self.temporal_attn(at * self.norm2(x.transpose(1, 2)) + bt).transpose(1, 2)

        integrate = self.integrate(torch.cat((prot_rep, x.mean(dim=1)), dim=-1))
        prot_rep = prot_rep + self.prot_mlp(self.norm3(integrate + prot_rep))

        x = x + gm * self.mlp(am * self.norm4(x + prot_rep.unsqueeze(1)) + bm)

        return x, prot_rep
