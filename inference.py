
from __future__ import annotations

import argparse
import gzip
import logging
import time
import gc
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
from tqdm import tqdm

import esm2
from src.vae.model import StructureEncoder, StructureAtomDecoder
from src.ldt.model import LatentEDM 
from src.ldt.diffusion import edm_sampler
from src.utils.pdb_utils import coords_alignment, write_trajectory_pdb
from src.utils.residue_constants import index_to_restype, one_to_three
from src.utils.protein_items import collect_proteins

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
LOGGER = logging.getLogger("latent_sampler")


def build_parser():
    p = argparse.ArgumentParser("Latent diffusion sampler")

    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--esm_model_path", required=True)
    p.add_argument("--encoder_ckpt", required=True)
    p.add_argument("--decoder_ckpt", required=True)  
    p.add_argument("--ldt_ckpt", required=True)
    p.add_argument("--latent_dim", type=int, default=12)
    p.add_argument("--scaling", type=float, default=15.0, help="Data scaling during diffusion")
    p.add_argument("--num_steps", type=int, default=100, help="Reverse diffusion steps")
    p.add_argument("--T", type=int, default=201, help="Number of Frames or structures")
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--num_repeat", type=int, default=1)
    p.add_argument("--forward_times", type=int, default=1)
    p.add_argument("--batch_decoder", type=int, default=50)
    p.add_argument("--batch_edm", type=int, default=100)
    p.add_argument("--temp_attn", action="store_true")
    p.add_argument("--use_cond", action="store_true")
    p.add_argument("--seed", type=int, default=4396)

    return p


def main():
    args = build_parser().parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    LOGGER.info(f"随机种子已设置为: {args.seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / f"log_time"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"log_time/T{args.T}_S{args.num_steps}_N{args.sigma_max}_{datetime.now():%Y%m%d_%H%M%S}.tsv"
    log_path.write_text("ID\tEDM\tVAE\tTotal\tLength\tSteps\n")

    LOGGER.info("Loading ESM‑2 …")
    esm_model, alphabet = esm2.pretrained.load_model_and_alphabet(args.esm_model_path)
    esm_model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    structure_encoder = None
    if args.use_cond or args.sigma_max <= 2:
        if not args.encoder_ckpt:
            raise ValueError("--encoder_path required in conditional mode")
        LOGGER.info("Loading StructureEncoder …")
        structure_encoder = StructureEncoder(1024, 1, 128, 2, args.latent_dim).to(device)
        structure_encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device, weights_only=True))
        structure_encoder.eval()

    proteins = collect_proteins(args, esm_model, batch_converter, structure_encoder, device)
    LOGGER.info("Collected %d proteins", len(proteins))

    del esm_model, batch_converter
    if structure_encoder:
        del structure_encoder
    torch.cuda.empty_cache()

    edm_arch_params: Tuple[int, int, int, int, int, int] = (
        (args.latent_dim, 384, 1280, 6, 768, 6)
        if args.temp_attn
        else (args.latent_dim, 1024, 1280, 16, 2048, 12)
    )

    LOGGER.info("Loading EDM and VAE …")
    edm_net = LatentEDM(
        *edm_arch_params,
        use_temporal_attn=args.temp_attn,
        use_condition_frame=args.use_cond
    ).to(device)
    edm_net.load_state_dict(torch.load(args.ldt_ckpt, map_location=device, weights_only=True))
    edm_net.eval()

    vae_decoder = StructureAtomDecoder(args.latent_dim, 1024, 16, 12, True).to(device)
    vae_decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location=device, weights_only=True))
    vae_decoder.eval()

    cond_mask = None
    if args.use_cond:
        cond_mask = torch.zeros(args.T, device=device)
        cond_mask[0] = 1
        cond_mask = cond_mask.unsqueeze(0)

    with torch.inference_mode():
        for protein in tqdm(proteins, desc="Proteins"):
            if args.use_cond:
                assert protein.mu0 is not None, "condition ensemble generation mode requires mu0"

            seq_len = len(protein.aa_idx)
            aa_indices = protein.aa_idx.unsqueeze(0).to(device)
            seq_repr_gpu = protein.seq_repr.unsqueeze(0).to(device)
            atom_mask = (
                protein.mask.to(device) if protein.mask is not None else torch.ones(seq_len, 14, device=device)
            )

            for rep in range(1, args.num_repeat + 1):
                if args.temp_attn:
                    t_edm_start = time.time()
                    mu_traj = torch.zeros((1, args.T, seq_len, args.latent_dim), device=device)
                    if args.use_cond:
                        mu_traj[0, 0] = protein.mu0.to(device) / args.scaling

                    all_traj = []
                    for t in range(1, args.forward_times + 1):
                        if args.use_cond and args.sigma_max <= 2:
                            assert protein.mu0 is not None, "partial denoise mode requires mu0"
                            noise = mu_traj.clone()
                            noise[:, :] = protein.mu0.to(device) / args.scaling
                            noise += torch.randn_like(noise) * args.sigma_max
                        else:
                            noise = torch.randn_like(mu_traj) * args.sigma_max

                        latent_traj_parts = edm_sampler(
                            edm_net,
                            noise,
                            mu_traj,
                            aa_indices,
                            seq_repr_gpu,
                            cond_mask,
                            num_steps=args.num_steps,
                            sigma_max=args.sigma_max,
                        ) * args.scaling

                        mu_traj[0, 0] = latent_traj_parts[0, -1]
                        if t > 1:
                            all_traj.append(latent_traj_parts.squeeze(0)[1:])
                        else:
                            all_traj.append(latent_traj_parts.squeeze(0))

                    edm_time = time.time() - t_edm_start
                    latent_traj = torch.cat(all_traj, dim=0)

                else:
                    latent_parts = []
                    t_edm_start = time.time()

                    if args.use_cond:
                        mu_cond = protein.mu0.to(device) / args.scaling
                        mu_cond = mu_cond.reshape(1, 1, seq_len, 12).expand(args.batch_edm, 1, seq_len, 12)
                    else:
                        mu_cond = torch.zeros((args.batch_edm, 1, seq_len, args.latent_dim), device=device)

                    aa_indices = aa_indices.expand(args.batch_edm, seq_len)
                    seq_repr_gpu = seq_repr_gpu.expand(args.batch_edm, seq_len, 1280)

                    for start in range(0, args.T, args.batch_edm):
                        n_frames = min(args.batch_edm, args.T - start)
                        noise = torch.randn((n_frames, 1, seq_len, args.latent_dim), device=device) * args.sigma_max

                        latent_chunk = edm_sampler(
                            edm_net,
                            noise,
                            mu_cond[:n_frames],
                            aa_indices[:n_frames],
                            seq_repr_gpu[:n_frames],
                            num_steps=args.num_steps,
                            sigma_max=args.sigma_max,
                        ) * args.scaling

                        latent_parts.append(latent_chunk.squeeze(1))

                    edm_time = time.time() - t_edm_start
                    latent_traj = torch.cat(latent_parts, dim=0)

                coords_frames = []
                n_frames = latent_traj.shape[0]
                t_vae_start = time.time()

                for start in range(0, n_frames, args.batch_decoder):
                    end = min(start + args.batch_decoder, n_frames)
                    coords_batch, _, _ = vae_decoder.decode(latent_traj[start:end])
                    coords_frames.append(coords_batch)

                vae_time = time.time() - t_vae_start
                coords_pred = torch.cat(coords_frames, dim=0)

                if protein.coords0 is None:
                    coords_ref = coords_pred[0].to(device)
                else:
                    coords_ref = protein.coords0.to(device)

                centroids_pred, rot_mats = [], []
                for t_idx in range(n_frames):
                    centroid_real, centroid_pred, R = coords_alignment(
                        coords_ref, coords_pred[t_idx], atom_mask
                    )
                    centroids_pred.append(centroid_pred)
                    rot_mats.append(R)

                centroids_pred = torch.stack(centroids_pred)
                rot_mats = torch.stack(rot_mats)
                coords_centered = coords_pred - centroids_pred[:, None, None]
                coords_pred = torch.einsum("tlia,tab->tlib", coords_centered, rot_mats) + centroid_real

                residue_names = [one_to_three[index_to_restype[int(i)]] for i in protein.aa_idx]
                pdb_output = args.out_dir / f"{protein.id}_R{rep}.pdb"
                write_trajectory_pdb(coords_pred.detach().cpu(), residue_names, pdb_output)
                LOGGER.info("Saved %s", pdb_output.name)

                with open(log_path, "a") as fp:
                    fp.write(
                        f"{protein.id}_{rep}\t{edm_time:.4f}\t{vae_time:.4f}\t{(edm_time+vae_time):.4f}\t{seq_len}\t{args.num_steps}\n"
                    )

                torch.cuda.empty_cache()
                gc.collect()

    LOGGER.info("All sampling done — log: %s", log_path)


if __name__ == "__main__":
    main()