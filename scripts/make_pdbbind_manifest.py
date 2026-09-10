"""Build a BA-Pred2 manifest from a PDBbind v2020-style directory.

Point ``--root`` at ONE set directory (e.g. ``general-set``); ``refined-set`` and ``v2020-other-PL`` are overlapping
copies of the same complexes, so scanning the parent would collect duplicates.

Split modes:
  random    - shuffled train/val/test by --train-frac/--val-frac (smoke tests, BA-Pred-compatible runs)
  casf2016  - test = CASF-2016 core set (--core-set CoreSet.dat), val = --val-frac (or --val-count) of the rest, train = remainder
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd

AFFINITY_RE = re.compile(r"^(Kd|Ki|IC50)([=<>~]+)(.+)$")


def parse_index(path: Path) -> dict[str, dict]:
    """Columns of INDEX_general_PL_data: code, resolution, release year, -logKd/Ki, Kd/Ki string, // ref (ligand)."""
    records = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            pdbid = parts[0].lower()
            if len(pdbid) != 4 or not pdbid.isalnum():
                continue
            try:
                affinity = float(parts[3])
            except ValueError:
                continue
            rec = {"affinity": affinity}
            try:
                rec["resolution"] = float(parts[1])
            except ValueError:
                rec["resolution"] = None
            try:
                rec["year"] = int(parts[2])
            except ValueError:
                rec["year"] = None
            raw = parts[4] if len(parts) > 4 else ""
            m = AFFINITY_RE.match(raw)
            rec["affinity_type"] = m.group(1) if m else None
            rec["affinity_relation"] = m.group(2) if m else None
            rec["affinity_raw"] = raw or None
            lig = re.search(r"\(([^)]*)\)\s*$", s)
            rec["ligand_name"] = lig.group(1) if lig else None
            records[pdbid] = rec
    return records


def parse_id_list(path: Path) -> set[str]:
    ids = set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            code = s.split()[0].lower()
            if len(code) == 4 and code.isalnum():
                ids.add(code)
    return ids


def index_complex_files(root: Path) -> dict[str, dict[str, Path]]:
    """Map pdbid -> {protein, pocket, sdf, mol2}; skips editor backups and .ipynb_checkpoints."""
    found: dict[str, dict[str, Path]] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        pdbid = entry.name.lower()
        if len(pdbid) != 4:
            continue
        files = {}
        for p in entry.iterdir():
            if not p.is_file() or p.name.endswith("~"):
                continue
            name = p.name.lower()
            if name == f"{pdbid}_protein.pdb":
                files["protein"] = p
            elif name == f"{pdbid}_pocket.pdb":
                files["pocket"] = p
            elif name == f"{pdbid}_ligand.sdf":
                files["sdf"] = p
            elif name == f"{pdbid}_ligand.mol2":
                files["mol2"] = p
        if "protein" in files and ("sdf" in files or "mol2" in files):
            found[pdbid] = files
    return found


def assign_splits(ids: list[str], mode: str, seed: int, train_frac: float, val_frac: float, core_ids: set[str] | None, val_count: int | None = None) -> dict[str, str]:
    rng = random.Random(seed)
    if mode == "random":
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n_val = val_count if val_count is not None else int(len(shuffled) * val_frac)
        n_train = int(len(shuffled) * train_frac) if val_count is None else len(shuffled) - n_val - int(len(shuffled) * (1 - train_frac - val_frac))
        return {pid: ("train" if i < n_train else "val" if i < n_train + n_val else "test") for i, pid in enumerate(shuffled)}
    if mode == "casf2016":
        if not core_ids:
            raise ValueError("--core-set is required for --split-mode casf2016")
        rest = [pid for pid in ids if pid not in core_ids]
        rng.shuffle(rest)
        n_val = val_count if val_count is not None else round(len(rest) * val_frac)
        split = {pid: "test" for pid in ids if pid in core_ids}
        split.update({pid: ("val" if i < n_val else "train") for i, pid in enumerate(rest)})
        return split
    raise ValueError(f"unknown split mode {mode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="one PDBbind set directory, e.g. .../PDBbind/general-set")
    ap.add_argument("--index", required=True, type=Path, help="INDEX_general_PL_data.<year>")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split-mode", choices=["random", "casf2016"], default="random")
    ap.add_argument("--core-set", type=Path, default=None, help="CASF-2016 CoreSet.dat (codes in column 1)")
    ap.add_argument("--refined-dir", type=Path, default=None, help="refined-set directory, used only to tag source_set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--val-count", type=int, default=None, help="absolute validation size; overrides --val-frac")
    args = ap.parse_args()

    labels = parse_index(args.index)
    files = index_complex_files(args.root)
    ids = sorted(set(labels) & set(files))
    missing_files = sorted(set(labels) - set(files))
    core_ids = parse_id_list(args.core_set) if args.core_set else None
    if core_ids is not None:
        absent = sorted(core_ids - set(ids))
        if absent:
            print(f"warning: {len(absent)} core-set codes not found under --root: {absent[:10]}{'...' if len(absent) > 10 else ''}")
    refined_ids = {p.name.lower() for p in args.refined_dir.iterdir() if p.is_dir()} if args.refined_dir else set()
    split = assign_splits(ids, args.split_mode, args.seed, args.train_frac, args.val_frac, core_ids, args.val_count)

    rows = []
    for pdbid in ids:
        f = files[pdbid]
        ligand = f.get("sdf") or f["mol2"]
        alt = f.get("mol2") if "sdf" in f else None
        rec = labels[pdbid]
        rows.append({
            "id": pdbid,
            "protein_path": str(f["protein"].resolve()),
            "pocket_path": str(f["pocket"].resolve()) if "pocket" in f else None,
            "ligand_path": str(ligand.resolve()),
            "ligand_alt_path": str(alt.resolve()) if alt else None,
            "affinity": rec["affinity"],
            "split": split[pdbid],
            "affinity_type": rec["affinity_type"],
            "affinity_relation": rec["affinity_relation"],
            "affinity_raw": rec["affinity_raw"],
            "resolution": rec["resolution"],
            "year": rec["year"],
            "ligand_name": rec["ligand_name"],
            "source_set": "refined" if pdbid in refined_ids else "general",
            "in_core_set": bool(core_ids and pdbid in core_ids),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    counts = df["split"].value_counts().to_dict()
    print(f"wrote {len(df)} complexes -> {args.out} | splits={counts} | index-only (no files)={len(missing_files)}")


if __name__ == "__main__":
    main()
