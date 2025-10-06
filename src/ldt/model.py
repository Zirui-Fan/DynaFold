import torch
import torch.nn as nn
import logging
from .modules import SpatioTemporalTransformerLayer, GEGLUMLP, SeqEmbedder, CondTrajEmbedder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LatentEDM(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        seq_dim,
        num_heads,
        mlp_dim,
        num_transformer_blocks,
        aa_nums=21,
        max_temp=20,
        use_temporal_attn=True,
        use_condition_frame=False,
    ):
        super(LatentEDM, self).__init__()

        self.use_temporal_attn = use_temporal_attn
        self.use_condition_frame = use_condition_frame
        self.seq_embedder = SeqEmbedder(aa_nums, seq_dim, latent_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.Linear(latent_dim // 2, latent_dim),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.Linear(latent_dim // 2, input_dim),
        )

        self.norm_in = nn.LayerNorm(input_dim)
        self.norm_out = nn.LayerNorm(latent_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                SpatioTemporalTransformerLayer(
                    embed_dim=latent_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    use_temporal_attn=use_temporal_attn,
                )
                for _ in range(num_transformer_blocks)
            ]
        )
        if use_condition_frame:
            self.cond_traj_embedder = CondTrajEmbedder(input_dim, latent_dim)

    def forward(
        self,
        x,
        cond_x,
        sigma,
        aa_types,
        esm2_seq_rep,
        attn_mask=None,
        condition_mask=None,
        temporal_mask=None,
    ):
        if torch.isnan(x).any():
            raise ValueError("Input tensor x contains NaN values.")

        x = self.input_proj(self.norm_in(x))
        B, T, L, D = x.shape

        if torch.isnan(esm2_seq_rep).any():
            raise ValueError("Warning: Input tensor sequences contains NaN values.")

        prot_rep = self.seq_embedder(aa_types, esm2_seq_rep)

        if condition_mask is None:
            condition_mask = torch.zeros((B, T), dtype=torch.bool, device=x.device)

        if self.use_condition_frame:
            cond_frames_rep, cond_traj_rep = self.cond_traj_embedder(cond_x, condition_mask)
            prot_rep = prot_rep + cond_frames_rep

            if self.use_temporal_attn:
                x = x + cond_traj_rep

        x = x + prot_rep.unsqueeze(1)

        for block in self.transformer_blocks:
            x, prot_rep = block(x, sigma, prot_rep, attn_mask, temporal_mask)

        x = self.output_proj(self.norm_out(x))

        return x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
