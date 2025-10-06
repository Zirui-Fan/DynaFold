import torch
import gzip
from pathlib import Path
from typing import List
from src.utils.residue_constants import restype_to_index
from src.utils.pdb_utils import parse_pdb_as_atom14

IDX_TO_AA_LETTER = {v: k for k, v in restype_to_index.items()}

def seq_to_index_tensor(seq: str) -> torch.Tensor:
    return torch.tensor([restype_to_index[c] for c in seq], dtype=torch.long)

def iterate_fasta(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt") as fh:
        header, seq = None, []
        for line in fh:
            if line.startswith(">"):
                if header:
                    yield header, "".join(seq)
                header, seq = line[1:].strip(), []
            else:
                seq.append(line.strip())
        if header:
            yield header, "".join(seq)

def save_tensor(t: torch.Tensor, file: Path):
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t.cpu(), file)

def load_tensor(file: Path):
    return torch.load(file, map_location="cpu")

class ProteinItem:
    def __init__(
        self,
        pid: str,
        seq_repr: torch.Tensor,
        aa_idx: torch.Tensor,
        mu0: torch.Tensor | None,
        coords0: torch.Tensor | None,
        mask: torch.Tensor | None
    ):
        self.id = pid
        self.seq_repr = seq_repr
        self.aa_idx = aa_idx
        self.mu0 = mu0
        self.coords0 = coords0
        self.mask = mask


class ProteinPairItem:
    def __init__(
        self,
        pid: str,
        seq_repr: torch.Tensor,
        aa_idx: torch.Tensor,
        mu0: torch.Tensor,
        muT: torch.Tensor,
        coords0: torch.Tensor,
        coordsT: torch.Tensor,
        mask: torch.Tensor,
    ):
        self.id = pid
        self.seq_repr = seq_repr
        self.aa_idx = aa_idx
        self.mu0 = mu0
        self.muT = muT
        self.coords0 = coords0
        self.coordsT = coordsT
        self.mask = mask


def collect_proteins(args, esm_model, batch_converter, encoder, device) -> List[ProteinItem]:
    cache_root = args.out_dir / "cache"
    cache_root.mkdir(exist_ok=True)
    proteins: List[ProteinItem] = []

    def add_protein(pid: str, seq: str, pdb_path: Path | None):
        sub = cache_root / pid
        seq_repr = mu0 = coords0 = mask = None

        if (sub / "seq_repr.pt").exists():
            seq_repr = load_tensor(sub / "seq_repr.pt")
        else:
            _, _, toks = batch_converter([(pid, seq)])
            with torch.inference_mode():
                out = esm_model(toks.to(device), repr_layers=range(34), return_contacts=False)
            rep = torch.stack([v for _, v in sorted(out["representations"].items())], dim=2)[0, 1:-1].mean(-2)
            seq_repr = rep.cpu()
            save_tensor(seq_repr, sub / "seq_repr.pt")

        if encoder is not None:
            assert pdb_path is not None, "Conditional mode requires PDB"
            if (sub / "mu.pt").exists():
                mu0 = load_tensor(sub / "mu.pt")
                coords0 = load_tensor(sub / "coords.pt")
                mask = load_tensor(sub / "mask.pt")
            else:
                parsed = parse_pdb_as_atom14(str(pdb_path), skip_unknown_residues=True)
                coords_np, mask_np = parsed['atom14_coords'], parsed['atom14_mask']
                coords_t = torch.from_numpy(coords_np).unsqueeze(0).to(device)
                with torch.inference_mode():
                    mu0, _, _ = encoder.encode(coords_t)
                mu0 = mu0[0].cpu()
                coords0 = torch.from_numpy(coords_np).cpu()
                mask = torch.from_numpy(mask_np).cpu()
                save_tensor(mu0, sub / "mu.pt")
                save_tensor(coords0, sub / "coords.pt")
                save_tensor(mask, sub / "mask.pt")

        aa_idx = seq_to_index_tensor(seq)
        proteins.append(ProteinItem(pid, seq_repr, aa_idx, mu0, coords0, mask))

    files = sorted(args.input.iterdir()) if args.input.is_dir() else [args.input]

    for fp in files:
        ext = fp.suffix.lower()
        if ext in (".pdb", ".gz") and not fp.name.endswith((".fasta.gz", ".fa.gz")):
            pid = fp.stem.split('.')[0]
            parsed = parse_pdb_as_atom14(str(fp), skip_unknown_residues=True)
            seq = ''.join(IDX_TO_AA_LETTER[i] for i in parsed['sequence'])
            add_protein(pid, seq, fp)
        elif ext in (".fasta", ".fa", ".fasta.gz", ".fa.gz"):
            for hdr, seq in iterate_fasta(fp):
                LOGGER.info(f'Processing {hdr}, length {len(seq)}')
                add_protein(hdr.split()[0], seq.upper(), None)
        else:
            raise ValueError(f"Unsupported file {fp}")

    return proteins


def collect_protein_pair(args, esm_model, batch_converter, encoder, device) -> ProteinPairItem:
    cache_root = args.out_dir / "cache"
    cache_root.mkdir(exist_ok=True)

    parsed0 = parse_pdb_as_atom14(str(args.pdb0), skip_unknown_residues=True)
    parsedT = parse_pdb_as_atom14(str(args.pdbT), skip_unknown_residues=True)

    seq0 = ''.join(IDX_TO_AA_LETTER[i] for i in parsed0['sequence'])
    seqT = ''.join(IDX_TO_AA_LETTER[i] for i in parsedT['sequence'])

    if seq0 != seqT:
        raise ValueError("The two PDB files must have the same sequence for trajectory prediction.")

    pid = f"{args.pdb0.stem}_to_{args.pdbT.stem}"
    sub = cache_root / pid

    seq_repr = mu0 = muT = coords0 = coordsT = mask = None

    if (sub / "seq_repr.pt").exists():
        seq_repr = load_tensor(sub / "seq_repr.pt")
    else:
        _, _, toks = batch_converter([(pid, seq0)])
        with torch.inference_mode():
            out = esm_model(toks.to(device), repr_layers=range(34), return_contacts=False)
        rep = torch.stack([v for _, v in sorted(out["representations"].items())], dim=2)[0, 1:-1].mean(-2)
        seq_repr = rep.cpu()
        save_tensor(seq_repr, sub / "seq_repr.pt")

    if (sub / "mu0.pt").exists() and (sub / "muT.pt").exists():
        mu0 = load_tensor(sub / "mu0.pt")
        muT = load_tensor(sub / "muT.pt")
        coords0 = load_tensor(sub / "coords0.pt")
        coordsT = load_tensor(sub / "coordsT.pt")
        mask = load_tensor(sub / "mask.pt")
    else:
        coords_np0, mask_np0 = parsed0['atom14_coords'], parsed0['atom14_mask']
        coords_npT, mask_npT = parsedT['atom14_coords'], parsedT['atom14_mask']

        if not (mask_np0 == mask_npT).all():
            raise ValueError("Atom masks of the two structures do not match.")

        coords_t0 = torch.from_numpy(coords_np0).unsqueeze(0).to(device)
        coords_tT = torch.from_numpy(coords_npT).unsqueeze(0).to(device)

        with torch.inference_mode():
            mu0, _, _ = encoder.encode(coords_t0)
            muT, _, _ = encoder.encode(coords_tT)

        mu0 = mu0[0].cpu()
        muT = muT[0].cpu()
        coords0 = torch.from_numpy(coords_np0).cpu()
        coordsT = torch.from_numpy(coords_npT).cpu()
        mask = torch.from_numpy(mask_np0).cpu()

        save_tensor(mu0, sub / "mu0.pt")
        save_tensor(muT, sub / "muT.pt")
        save_tensor(coords0, sub / "coords0.pt")
        save_tensor(coordsT, sub / "coordsT.pt")
        save_tensor(mask, sub / "mask.pt")

    aa_idx = seq_to_index_tensor(seq0)
    return ProteinPairItem(pid, seq_repr, aa_idx, mu0, muT, coords0, coordsT, mask)