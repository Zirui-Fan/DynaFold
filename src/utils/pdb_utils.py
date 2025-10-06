import numpy as np
import torch

from .residue_constants import (
    restype_to_index,
    three_to_one,
    resname_atom14_indices,
    restype_name_to_atom14_names,
)


def parse_pdb_as_atom14(pdb_file_path, skip_unknown_residues=False): 
    atom14_coords_list = []
    atom14_mask_list = []
    seq = []
    chain_ids = []
    res_ids = []
    seq_letters = ''

    current_coords = None
    current_mask = None
    current_res_name = None
    current_chain_id = None
    current_res_id = None

    with open(pdb_file_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21]
                res_id = line[22:27].strip()
                x_str = line[30:38].strip()
                y_str = line[38:46].strip()
                z_str = line[46:54].strip()

                try:
                    x = float(x_str)
                    y = float(y_str)
                    z = float(z_str)
                except ValueError:
                    continue

                if (current_res_id is not None) and (
                    (chain_id != current_chain_id) or (res_id != current_res_id)
                ):
                    atom14_coords_list.append(current_coords)
                    atom14_mask_list.append(current_mask)
                    one_letter = three_to_one.get(current_res_name, 'X')
                    if skip_unknown_residues and one_letter == 'X':
                        raise ValueError(
                            f"Unknown residue '{current_res_name}' encountered at chain {current_chain_id} residue {current_res_id}."
                        )
                    seq.append(restype_to_index[one_letter])
                    chain_ids.append(current_chain_id)
                    res_ids.append(current_res_id)
                    seq_letters = seq_letters + one_letter

                    current_coords = np.full((14, 3), np.nan, dtype=np.float32)
                    current_mask = np.zeros(14, dtype=np.bool_)
                    current_res_name = res_name
                    current_chain_id = chain_id
                    current_res_id = res_id

                if current_coords is None:
                    current_coords = np.full((14, 3), np.nan, dtype=np.float32)
                    current_mask = np.zeros(14, dtype=np.bool_)
                    current_res_name = res_name
                    current_chain_id = chain_id
                    current_res_id = res_id

                atom14_indices = resname_atom14_indices.get(current_res_name, {})
                if atom_name in atom14_indices:
                    atom_index = atom14_indices[atom_name]
                    current_coords[atom_index] = [x, y, z]
                    current_mask[atom_index] = 1

        if current_res_id is not None and current_coords is not None:
            atom14_coords_list.append(current_coords)
            atom14_mask_list.append(current_mask)
            one_letter = three_to_one.get(current_res_name, 'X')
            if skip_unknown_residues and one_letter == 'X':
                raise ValueError(
                    f"Unknown residue '{current_res_name}' encountered at chain {current_chain_id} residue {current_res_id}."
                )
            seq.append(restype_to_index[one_letter])
            chain_ids.append(current_chain_id)
            res_ids.append(current_res_id)
            seq_letters = seq_letters + one_letter

    atom14_coords = np.array(atom14_coords_list)
    atom14_mask = np.array(atom14_mask_list)
    sequence = np.array(seq, dtype=np.int32)
    unique_elements, indices = np.unique(chain_ids, return_inverse=True)
    chain_ids = np.array(indices, dtype=np.int32)

    return {
        'atom14_coords': atom14_coords,
        'atom14_mask': atom14_mask,
        'sequence': sequence,
        'chain_ids': chain_ids,
        'residue_ids': res_ids,
        'seq_letters': seq_letters
    }


def coords_alignment(real_coords, pred_coords, atom_mask=None, sequence_mask=None):
    if len(real_coords.shape) != 3 or real_coords.shape[2] != 3:
        raise ValueError("真实坐标必须是 [L, numatoms, 3] 的形状。")

    if real_coords.shape != pred_coords.shape:
        raise ValueError("真实坐标和预测坐标的形状必须相同。")

    L, numatoms, _ = real_coords.shape

    if atom_mask is not None:
        if atom_mask.shape != (L, numatoms):
            raise ValueError("atom_mask 必须是 [L, numatoms] 的形状，与输入坐标匹配。")
        atom_mask = atom_mask.bool()
        real_coords = real_coords[atom_mask].reshape(-1, 3)
        pred_coords = pred_coords[atom_mask].reshape(-1, 3)

    elif sequence_mask is not None:
        sequence_mask = sequence_mask.bool()
        real_coords = real_coords[sequence_mask].reshape(-1, 3)
        pred_coords = pred_coords[sequence_mask].reshape(-1, 3)

    else:
        real_coords = real_coords.reshape(-1, 3)
        pred_coords = pred_coords.reshape(-1, 3)

    centroid_real = real_coords.mean(dim=0)
    centroid_pred = pred_coords.mean(dim=0)

    real_centered = real_coords - centroid_real
    pred_centered = pred_coords - centroid_pred

    H = pred_centered.t() @ real_centered

    with torch.no_grad(): 
        U, _, V = torch.linalg.svd(H)

    d = torch.det(U @ V)
    if d < 0:
        U[:, -1] *= -1

    R = U @ V

    return centroid_real, centroid_pred, R


def write_trajectory_pdb(coords, three_letters, file_path, chain_id="A"):
    N, L, num_atoms, dim = coords.shape

    assert num_atoms == 14, "Expected 14 atoms per residue, got {}".format(num_atoms)
    assert dim == 3, "Expected 3D coordinates, got {}".format(dim)
    assert len(three_letters) == L, "three_letters length {} must match L {}".format(len(three_letters), L)

    def get_element(atom_name):
        if atom_name.strip():
            return atom_name.strip()[0]
        return ""

    with open(file_path, 'w') as f:
        for frame in range(N):
            atom_count = 1
            f.write(f"MODEL {frame+1}\n")

            for res_i in range(L):
                residue = three_letters[res_i]
                atoms = restype_name_to_atom14_names.get(residue, [''] * 14)
                atom_coords = coords[frame, res_i]

                for atom_j, atom_name in enumerate(atoms):
                    if atom_name:
                        x, y, z = atom_coords[atom_j]
                        element = get_element(atom_name)
                        f.write(
                            f"ATOM  {atom_count:5d}  {atom_name:<3} {residue:>3} {chain_id:>1}{res_i+1:4d}    "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:<2}\n"
                        )
                        atom_count += 1

            f.write("ENDMDL\n")

        f.write("END\n")
