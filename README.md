# BA-Pred2

Experimental successor to BA-Pred using **recurrent graph inference** for protein-ligand binding-affinity prediction.

> Full architecture, data, training, equivariance, stability, and ablation plan: **[SPEC.md](SPEC.md)**

## Core idea

Instead of stacking independent GNN layers, BA-Pred2 learns one shared binding-inference operator and repeatedly applies it:

1. refine the dynamic protein-ligand interface state,
2. send bidirectional protein <-> ligand messages,
3. propagate the binding-conditioned state inside each molecule,
4. re-inject the static initial node/interface representation every cycle.

The recurrent core uses LayerNorm, gated updates and LayerScale. The old BA-Pred MHA path is intentionally absent.

## Graphs

Protein and ligand are separate PyG `HeteroData` node types.

- protein intra graph: covalent + spatial edges
- ligand intra graph: covalent + spatial edges
- protein -> ligand contact graph: dynamic interface state
- topology anchor: random-walk PE on the covalent graph
- interface features: distance RBF, donor/acceptor-like atom flags, charge terms and local orientation cosines

Default cutoffs are 8 A for pocket extraction, 5 A for protein-ligand contacts, 5 A for protein spatial edges and 4.5 A for ligand spatial edges.

## Raw manifest

```text
id,protein_path,ligand_path,affinity,split
1abc,/data/1abc_protein.pdb,/data/1abc_ligand.sdf,7.21,train
```

`affinity` should be a pKd/pKi-like `-log10(M)` target.

Optional columns: `pocket_path` (a pre-cut pocket PDB, parsed instead of `protein_path` when present), `ligand_alt_path`
(tried when `ligand_path` fails to parse), and any metadata. `pKd` is accepted as an alias of `affinity`.

### Build a PDBbind manifest

```bash
python scripts/make_pdbbind_manifest.py \
  --root /data/PDBbind/general-set \
  --index /data/PDBbind/INDEX_general_PL_data.2020 \
  --out data/pdbbind_v2020_casf2016.csv \
  --split-mode casf2016 \
  --core-set /data/CASF-2016/power_screening/CoreSet.dat \
  --refined-dir /data/PDBbind/refined-set
```

`--root` must point at one set directory (`general-set` contains every indexed complex; `refined-set` and
`v2020-other-PL` are overlapping copies). `--split-mode casf2016` puts the CASF-2016 core set in `test` and draws
`val` (`--val-frac`, default 0.1) from the remainder; `--split-mode random` is the smoke-test default. The manifest also
carries release year, resolution, Kd/Ki/IC50 type and relation (`=`, `<`, `>`, `~`) so temporal or censored-label
filtering can be applied later without re-parsing the index.

## Environment

Python >= 3.12, torch >= 2.14 (the PyPI wheel ships CUDA 13 and supports Blackwell / sm_120), PyG >= 2.8, RDKit >= 2026.3.

```bash
uv sync            # creates .venv with the dev group (pytest, ruff)
uv run pytest -q
```

## Preprocess

```bash
bapred2-preprocess \
  --manifest data/pdbbind_v2020_casf2016.csv \
  --out data/processed/pdbbind_v2020_casf2016 \
  --config configs/bapred2_base.yaml \
  --workers 19
```

Output: cached PyG graphs, `processed_manifest.csv`, `skipped.csv` (complex id + reason) and `preprocess_config.yaml`
(graph settings, versions, counts). Existing graphs are reused unless `--overwrite` is given.

## Train

```bash
bapred2-train \
  --manifest data/processed/processed_manifest.csv \
  --config configs/bapred2_base.yaml \
  --out runs/base \
  --device cuda
```

Training samples recurrent depth from `[2, 3, 4, 6, 8]`. The default objective is Huber loss and the best checkpoint is
selected by validation RMSE. Mixed precision defaults to bfloat16 (`train.amp_dtype`). `history.json` records parameter
count, mean training recycles, epoch wall time and peak GPU memory. For a quick end-to-end check use
`--epochs 2 --limit-train 2000 --limit-val 300`. Checkpoints store the config and feature dims, so evaluation does not
need a matching manifest sample.

## Test-time recycle sweep

```bash
bapred2-eval \
  --manifest data/processed/processed_manifest.csv \
  --checkpoint runs/base/best.pt \
  --split test \
  --recycles 1,2,3,4,6,8,12,16 \
  --out runs/base/recycle_sweep.json
```

The evaluator reports RMSE, MAE, Pearson, Spearman, wall time and the hidden-state update magnitude at each cycle. The important architectural test is whether performance remains stable or improves when inference recurrence is increased at fixed parameter count.

## First ablations

- independent fixed-depth GNN vs shared recurrent block
- covalent-only vs covalent + spatial intra edges
- static-state reinjection on/off
- endpoint interface context on/off
- ligand-only vs ligand + interface readout
- recycle sweep at fixed parameters

The current implementation uses contacts sharing a protein or ligand atom as a batching-friendly interface-context operator. Exact P-P-L / P-L-L topology-aware triangle updates are intentionally left as the next extension rather than being approximated silently.

## Status

Milestone 0 (runnable scalar baseline) is implemented and exercised end-to-end on PDBbind v2020 with the CASF-2016
core set as test split. The SPEC-by-SPEC audit, including what is partial or missing for later milestones, is in
[docs/GAPS.md](docs/GAPS.md); the architecture critique with probe evidence is in [docs/ARCHITECTURE_REVIEW.md](docs/ARCHITECTURE_REVIEW.md). Run reports are rendered with `python scripts/make_report.py --run runs/<name> --out report.html`.
