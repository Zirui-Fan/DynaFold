import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import TransformerStack, SwiGLUFFNEncoder, SwiGLUFFN
from esm.utils.misc import knn_graph
from esm.utils.structure.affine3d import Affine3D, build_affine3d_from_coordinates
from esm.models.vqvae import node_gather, RelativePositionEmbedding, GeometricEncoderStack


class StructureEncoder(nn.Module):
    def __init__(self, d_model, n_heads, v_heads, n_layers, d_latent, bn=False):
        super().__init__()

        self.transformer = GeometricEncoderStack(d_model, n_heads, v_heads, n_layers)
        self.relative_positional_embedding = RelativePositionEmbedding(32, d_model, init_std=0.02)
        self.bn = bn
        self.knn = 16
        self.layernorm = nn.LayerNorm(d_model)

        if self.bn:
            self.batchnorm = nn.BatchNorm1d(d_latent)

        self.ffn = SwiGLUFFNEncoder(d_model, d_model // 2, d_latent)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode_local_structure(
        self,
        coords: torch.Tensor,
        affine: Affine3D,
        attention_mask: torch.Tensor,
        sequence_id: torch.Tensor | None,
        affine_mask: torch.Tensor,
        residue_index: torch.Tensor | None = None,
    ):
        assert coords.size(-1) == 3 and coords.size(-2) == 3

        with torch.no_grad():
            knn_edges, _ = self.find_knn_edges(
                coords,
                ~attention_mask,
                coord_mask=affine_mask,
                sequence_id=sequence_id,
                knn=self.knn,
            )

            B, L, E = knn_edges.shape
            affine_tensor = affine.tensor
            T_D = affine_tensor.size(-1)

            knn_affine_tensor = node_gather(affine_tensor, knn_edges)
            knn_affine_tensor = knn_affine_tensor.view(-1, E, T_D).contiguous()
            affine = Affine3D.from_tensor(knn_affine_tensor)

            knn_sequence_id = (
                node_gather(sequence_id.unsqueeze(-1), knn_edges).view(-1, E)
                if sequence_id is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            )

            knn_affine_mask = node_gather(affine_mask.unsqueeze(-1), knn_edges).view(-1, E)
            knn_chain_id = torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)

            if residue_index is None:
                res_idxs = knn_edges.view(-1, E)
            else:
                res_idxs = node_gather(residue_index.unsqueeze(-1), knn_edges).view(-1, E)

        z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs)

        z, _ = self.transformer.forward(
            x=z,
            sequence_id=knn_sequence_id,
            affine=affine,
            affine_mask=knn_affine_mask,
            chain_id=knn_chain_id,
        )

        z = z.view(B, L, E, -1)
        z = z[:, :, 0, :]

        return z

    @staticmethod
    def find_knn_edges(
        coords,
        padding_mask,
        coord_mask,
        sequence_id: torch.Tensor | None = None,
        knn: int | None = None,
    ) -> tuple:
        assert knn is not None

        coords = coords.clone()
        coords[~coord_mask] = 0

        if sequence_id is None:
            sequence_id = torch.zeros(
                (coords.shape[0], coords.shape[1]), device=coords.device
            ).long()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            ca = coords[..., 1, :]
            edges, edge_mask = knn_graph(
                ca,
                coord_mask,
                padding_mask,
                sequence_id,
                no_knn=knn,
            )

        return edges, edge_mask

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self,
        coords: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        residue_index: torch.Tensor | None = None,
    ):
        coords = coords[..., :3, :]

        affine, affine_mask = build_affine3d_from_coordinates(coords=coords)

        if attention_mask is None:
            attention_mask = torch.ones_like(affine_mask, dtype=torch.bool)

        attention_mask = attention_mask.bool()

        if sequence_id is None:
            sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)

        z = self.encode_local_structure(
            coords=coords,
            affine=affine,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            affine_mask=affine_mask,
            residue_index=residue_index,
        )

        z = self.layernorm(z)
        z = z.masked_fill(~affine_mask.unsqueeze(2), 0)

        mu, logvar = self.ffn(z)

        if self.bn:
            mu = mu.permute(0, 2, 1)
            mu = self.batchnorm(mu)
            mu = mu.permute(0, 2, 1)

        logvar = torch.clamp(logvar, max=5.0)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class StructureAtomDecoder(nn.Module):
    def __init__(
        self,
        d_latent,
        d_model,
        n_heads,
        n_layers,
        predict_torsion_angles=False
    ):
        super().__init__()

        self.decoder_channels = d_model
        self.input_norm = nn.LayerNorm(d_latent)

        self.input_proj = nn.Sequential(
            SwiGLUFFN(d_latent, d_model // 4, d_model // 4),
            nn.LayerNorm(d_model // 4),
            SwiGLUFFN(d_model // 4, d_model, d_model)
        )

        self.decoder_stack = TransformerStack(
            d_model, n_heads, n_layers, scale_residue=False,
        )

        self.atom14_project = nn.Sequential(
            SwiGLUFFN(d_model, d_model, d_model // 4),
            nn.LayerNorm(d_model // 4),
            SwiGLUFFN(d_model // 4, d_model // 4, 42)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def decode(
        self,
        structure_latent: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        sequence: torch.Tensor | None = None,
    ):
        B, L, _ = structure_latent.shape

        if attention_mask is None:
            attention_mask = torch.ones((B, L), dtype=torch.bool).to(structure_latent.device)

        attention_mask = attention_mask.bool()

        x = self.input_proj(self.input_norm(structure_latent))

        x, _ = self.decoder_stack(
            x, sequence_id, attention_mask
        )

        xyz_pred = self.atom14_project(x).reshape(B, L, 14, 3)

        return xyz_pred, None, None