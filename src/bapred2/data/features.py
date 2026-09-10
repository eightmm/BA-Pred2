from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import torch
from rdkit import Chem
from torch_geometric.utils import get_self_loop_attr, scatter, to_edge_index, to_torch_csr_tensor

ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se", "METAL", "OTHER"]
METALS = {
    "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI",
    "CU", "ZN", "Y", "ZR", "NB", "MO", "RU", "RH", "PD", "AG", "CD", "HF", "TA", "W", "RE", "OS", "IR",
    "PT", "AU", "HG", "AL", "GA", "IN", "SN", "PB", "BI", "LA", "CE", "PR", "ND", "SM", "EU", "GD", "TB",
    "DY", "HO", "ER", "TM", "YB", "LU",
}
HYBRIDS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, choices: Iterable) -> list[float]:
    choices = list(choices)
    return [1.0 if value == c else 0.0 for c in choices]


def safe_call(fn, default=0):
    try:
        return fn()
    except Exception:
        return default


def atom_features(atom: Chem.Atom) -> list[float]:
    symbol = atom.GetSymbol()
    symbol_key = "METAL" if symbol.upper() in METALS else (symbol if symbol in ELEMENTS else "OTHER")
    degree = min(int(safe_call(atom.GetDegree, 0)), 6)
    total_h = min(int(safe_call(atom.GetTotalNumHs, 0)), 4)
    formal_charge = max(-3, min(3, int(safe_call(atom.GetFormalCharge, 0)))) / 3.0
    mass = float(safe_call(atom.GetMass, 0.0)) / 200.0
    atomic_num = float(atom.GetAtomicNum()) / 100.0
    feat = []
    feat += one_hot(symbol_key, ELEMENTS)
    feat += one_hot(degree, range(7))
    feat += one_hot(total_h, range(5))
    feat += one_hot(safe_call(atom.GetHybridization, Chem.rdchem.HybridizationType.UNSPECIFIED), HYBRIDS)
    feat += [
        float(safe_call(atom.GetIsAromatic, False)),
        float(safe_call(atom.IsInRing, False)),
        formal_charge,
        mass,
        atomic_num,
    ]
    return feat


def node_feature_tensor(mol: Chem.Mol, atom_indices: list[int] | None = None) -> torch.Tensor:
    indices = atom_indices if atom_indices is not None else list(range(mol.GetNumAtoms()))
    return torch.tensor([atom_features(mol.GetAtomWithIdx(i)) for i in indices], dtype=torch.float32)


def coords_tensor(mol: Chem.Mol, atom_indices: list[int] | None = None) -> torch.Tensor:
    conf = mol.GetConformer()
    indices = atom_indices if atom_indices is not None else list(range(mol.GetNumAtoms()))
    return torch.tensor([list(conf.GetAtomPosition(i)) for i in indices], dtype=torch.float32)


def rbf_distance(distance: torch.Tensor, dim: int = 16, cutoff: float = 8.0) -> torch.Tensor:
    distance = distance.reshape(-1, 1)
    centers = torch.linspace(0.0, cutoff, dim, device=distance.device, dtype=distance.dtype).reshape(1, -1)
    width = cutoff / max(dim - 1, 1)
    gamma = 1.0 / max(width * width, 1e-6)
    return torch.exp(-gamma * (distance - centers) ** 2)


def random_walk_pe(edge_index: torch.Tensor, num_nodes: int, k: int) -> torch.Tensor:
    if num_nodes == 0:
        return torch.zeros((0, k), dtype=torch.float32)
    if edge_index.numel() == 0:
        return torch.zeros((num_nodes, k), dtype=torch.float32)
    row = edge_index[0]
    deg = scatter(torch.ones(row.size(0), dtype=torch.float32), row, dim=0, dim_size=num_nodes, reduce="sum").clamp_min(1.0)
    value = (1.0 / deg)[row]
    adj = to_torch_csr_tensor(edge_index, value, size=(num_nodes, num_nodes))

    def diagonal(sparse_matrix):
        ei, ev = to_edge_index(sparse_matrix)
        return get_self_loop_attr(ei, ev, num_nodes=num_nodes)

    out = adj
    pe = [diagonal(out)]
    for _ in range(k - 1):
        out = out @ adj
        pe.append(diagonal(out))
    return torch.stack(pe, dim=-1).float()


def atom_property_masks(mol: Chem.Mol, atom_indices: list[int] | None = None) -> torch.Tensor:
    """Five stable atom-local chemistry flags: HBA, HBD, cationic, anionic, hydrophobic."""
    indices = atom_indices if atom_indices is not None else list(range(mol.GetNumAtoms()))
    rows = []
    for idx in indices:
        atom = mol.GetAtomWithIdx(idx)
        z = atom.GetAtomicNum()
        charge = int(safe_call(atom.GetFormalCharge, 0))
        total_h = int(safe_call(atom.GetTotalNumHs, 0))
        aromatic = bool(safe_call(atom.GetIsAromatic, False))
        hba = float(z in {7, 8, 9, 15, 16, 17, 35, 53} and charge <= 0)
        hbd = float(z in {7, 8, 16} and total_h > 0 and charge >= 0)
        cationic = float(charge > 0)
        anionic = float(charge < 0)
        hydrophobic = float(z in {6, 9, 16, 17, 35, 53} and charge == 0 and (aromatic or z != 16))
        rows.append([hba, hbd, cationic, anionic, hydrophobic])
    return torch.tensor(rows, dtype=torch.float32)


def local_direction(coords: torch.Tensor, covalent_edge_index: torch.Tensor) -> torch.Tensor:
    """Rotation-equivariant local outward vector used only through invariant cosines."""
    n = coords.size(0)
    nbrs: dict[int, list[int]] = defaultdict(list)
    for s, d in covalent_edge_index.t().tolist():
        nbrs[d].append(s)
    out = torch.zeros_like(coords)
    for i in range(n):
        if nbrs[i]:
            mean_nbr = coords[torch.tensor(nbrs[i], dtype=torch.long)].mean(0)
            out[i] = coords[i] - mean_nbr
    norm = out.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return out / norm


def bonded_edges(mol: Chem.Mol, selected: list[int], coords: torch.Tensor, rbf_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    old_to_new = {old: new for new, old in enumerate(selected)}
    src, dst, feats = [], [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a not in old_to_new or b not in old_to_new:
            continue
        bf = one_hot(bond.GetBondType(), BOND_TYPES) + [
            float(bond.GetIsConjugated()), float(bond.IsInRing()), float(bond.GetIsAromatic())
        ]
        for u_old, v_old in ((a, b), (b, a)):
            u, v = old_to_new[u_old], old_to_new[v_old]
            dist = torch.norm(coords[u] - coords[v]).reshape(1)
            feat = torch.tensor(bf + [1.0, 0.0], dtype=torch.float32)
            feat = torch.cat([feat, rbf_distance(dist, rbf_dim, 8.0).squeeze(0)])
            src.append(u)
            dst.append(v)
            feats.append(feat)
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.stack(feats) if feats else torch.empty((0, 4 + 3 + 2 + rbf_dim), dtype=torch.float32)
    return edge_index, edge_attr


def add_spatial_edges(
    coords: torch.Tensor,
    covalent_edge_index: torch.Tensor,
    covalent_edge_attr: torch.Tensor,
    cutoff: float,
    max_neighbors: int,
    rbf_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = coords.size(0)
    if n <= 1:
        return covalent_edge_index, covalent_edge_attr
    bonded = set(map(tuple, covalent_edge_index.t().tolist()))
    dist = torch.cdist(coords, coords)
    spatial_src, spatial_dst, spatial_feat = [], [], []
    for dst in range(n):
        candidates = torch.where((dist[:, dst] < cutoff) & (dist[:, dst] > 1e-6))[0]
        if candidates.numel() > max_neighbors:
            order = torch.argsort(dist[candidates, dst])[:max_neighbors]
            candidates = candidates[order]
        for src in candidates.tolist():
            if (src, dst) in bonded:
                continue
            base = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
            feat = torch.cat([base, rbf_distance(dist[src, dst].reshape(1), rbf_dim, 8.0).squeeze(0)])
            spatial_src.append(src)
            spatial_dst.append(dst)
            spatial_feat.append(feat)
    if not spatial_src:
        return covalent_edge_index, covalent_edge_attr
    sp_index = torch.tensor([spatial_src, spatial_dst], dtype=torch.long)
    sp_attr = torch.stack(spatial_feat)
    return torch.cat([covalent_edge_index, sp_index], dim=1), torch.cat([covalent_edge_attr, sp_attr], dim=0)


# ---------------------------------------------------------------------------------------------------------------------
# Residue-template protein chemistry (v0.2). PDB files carry no bond orders or charges, so RDKit valence inference on a
# proximity-bonded protein mis-assigns most donor/charge flags; residue + atom-name templates are exact for the 20
# standard residues. Token vocabularies follow RMSD-Pred (22 residue classes, (residue, atom) classes + OXT/METAL/UNK/XXX).
# ---------------------------------------------------------------------------------------------------------------------
STANDARD_RESIDUES = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
RESIDUE_ATOMS = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
}
RES_TOKENS = {r: i for i, r in enumerate(STANDARD_RESIDUES)}
RES_TOKENS["XXX"] = 20
RES_TOKENS["METAL"] = 21
N_RES_TOKENS = 22
ATOM_TOKENS: dict[tuple[str, str], int] = {}
for _res in STANDARD_RESIDUES:
    for _name in RESIDUE_ATOMS[_res]:
        ATOM_TOKENS[(_res, _name)] = len(ATOM_TOKENS)
ATOM_TOKENS[("ANY", "OXT")] = len(ATOM_TOKENS)
ATOM_TOKENS[("METAL", "METAL")] = len(ATOM_TOKENS)
for _el in ("C", "N", "O", "S", "P", "SE"):
    ATOM_TOKENS[("XXX", _el)] = len(ATOM_TOKENS)
ATOM_TOKENS[("UNK", "UNK")] = len(ATOM_TOKENS)
N_ATOM_TOKENS = len(ATOM_TOKENS)  # 176
WATER_RESIDUES = {"HOH", "WAT", "DOD", "H2O", "TIP", "SOL"}

# (HBA, HBD, cationic, anionic, hydrophobic) per side-chain atom; backbone handled generically.
_HBA = {("ASN", "OD1"), ("ASP", "OD1"), ("ASP", "OD2"), ("CYS", "SG"), ("GLN", "OE1"), ("GLU", "OE1"), ("GLU", "OE2"),
        ("HIS", "ND1"), ("HIS", "NE2"), ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH")}
_HBD = {("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("ASN", "ND2"), ("CYS", "SG"), ("GLN", "NE2"), ("HIS", "ND1"),
        ("HIS", "NE2"), ("LYS", "NZ"), ("SER", "OG"), ("THR", "OG1"), ("TRP", "NE1"), ("TYR", "OH")}
_CATION = {("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("LYS", "NZ")}
_ANION = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
_HYDROPHOBIC = {
    ("ALA", "CB"), ("ARG", "CB"), ("ARG", "CG"), ("ASN", "CB"), ("ASP", "CB"), ("CYS", "CB"), ("GLN", "CB"), ("GLN", "CG"),
    ("GLU", "CB"), ("GLU", "CG"), ("HIS", "CB"), ("ILE", "CB"), ("ILE", "CG1"), ("ILE", "CG2"), ("ILE", "CD1"),
    ("LEU", "CB"), ("LEU", "CG"), ("LEU", "CD1"), ("LEU", "CD2"), ("LYS", "CB"), ("LYS", "CG"), ("LYS", "CD"),
    ("MET", "CB"), ("MET", "CG"), ("MET", "SD"), ("MET", "CE"), ("PHE", "CB"), ("PHE", "CG"), ("PHE", "CD1"), ("PHE", "CD2"),
    ("PHE", "CE1"), ("PHE", "CE2"), ("PHE", "CZ"), ("PRO", "CB"), ("PRO", "CG"), ("PRO", "CD"), ("THR", "CG2"),
    ("TRP", "CB"), ("TRP", "CG"), ("TRP", "CE3"), ("TRP", "CZ2"), ("TRP", "CZ3"), ("TRP", "CH2"),
    ("TYR", "CB"), ("TYR", "CG"), ("TYR", "CD1"), ("TYR", "CD2"), ("TYR", "CE1"), ("TYR", "CE2"), ("VAL", "CB"), ("VAL", "CG1"), ("VAL", "CG2"),
}


def _template_flags(res: str, name: str) -> list[float] | None:
    """Returns [hba, hbd, cation, anion, hydrophobic] for a standard-residue atom, None if not templated."""
    if name == "OXT":
        return [1.0, 0.0, 0.0, 1.0, 0.0]
    if res not in RESIDUE_ATOMS or name not in RESIDUE_ATOMS[res]:
        return None
    if name == "N":
        return [0.0, 0.0 if res == "PRO" else 1.0, 0.0, 0.0, 0.0]
    if name == "O":
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    key = (res, name)
    return [float(key in _HBA), float(key in _HBD), float(key in _CATION), float(key in _ANION), float(key in _HYDROPHOBIC)]


def is_water_atom(atom: Chem.Atom) -> bool:
    info = atom.GetPDBResidueInfo()
    return info is not None and info.GetResidueName().strip().upper() in WATER_RESIDUES


def protein_tokens_and_flags(mol: Chem.Mol, atom_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(token_res [N] int64, token_atom [N] int64, flags [N,5]) from PDB residue info; RDKit flags are the fallback."""
    fallback = atom_property_masks(mol, atom_indices)
    res_tok, atom_tok, flags = [], [], []
    for k, idx in enumerate(atom_indices):
        atom = mol.GetAtomWithIdx(idx)
        info = atom.GetPDBResidueInfo()
        symbol = atom.GetSymbol().upper()
        res = info.GetResidueName().strip().upper() if info is not None else ""
        name = info.GetName().strip().upper() if info is not None else ""
        if symbol in METALS:
            res_tok.append(RES_TOKENS["METAL"])
            atom_tok.append(ATOM_TOKENS[("METAL", "METAL")])
            flags.append([0.0, 0.0, 1.0, 0.0, 0.0])
            continue
        tf = _template_flags(res, name) if res in RESIDUE_ATOMS else None
        if res in RESIDUE_ATOMS and (name in RESIDUE_ATOMS[res] or name == "OXT"):
            res_tok.append(RES_TOKENS[res])
            atom_tok.append(ATOM_TOKENS[(res, name)] if name != "OXT" else ATOM_TOKENS[("ANY", "OXT")])
            flags.append(tf)
        else:
            res_tok.append(RES_TOKENS["XXX"])
            atom_tok.append(ATOM_TOKENS.get(("XXX", symbol), ATOM_TOKENS[("UNK", "UNK")]))
            flags.append(fallback[k].tolist())
    return (
        torch.tensor(res_tok, dtype=torch.long),
        torch.tensor(atom_tok, dtype=torch.long),
        torch.tensor(flags, dtype=torch.float32).reshape(-1, 5),
    )
