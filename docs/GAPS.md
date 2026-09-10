# SPEC coverage and gap list

Status of the `bapred2` branch against [SPEC.md](../SPEC.md), audited 2026-09-10 on the CUDA 13 / torch 2.14 stack.
Legend: **done** = implemented and exercised by a test or a real run, **partial** = present but deviates from SPEC,
**missing** = not implemented. "Fixed here" marks items closed in this audit.

## Milestone 0 — runnable scalar baseline

| SPEC | Item | Status | Notes |
|---|---|---|---|
| 2, 9 | Shared recurrent block (interface -> cross -> intra), static anchors `H^0/Q^0` | done | `model/layers.py` |
| 3.3, 9.1 | Bounded interface update | done | `q + s*g*(qhat - q)` is gated interpolation with a LayerScale gain, not additive accumulation |
| 4.1, 4.2 | Covalent + spatial intra edges with explicit relation type | done | `[cov, spatial]` one-hot in `edge_attr[:, 7:9]` |
| 4 | `max_spatial_neighbors` | partial | single value (32) for both molecules; SPEC wants 32 protein / 24 ligand |
| 4.3, 5.4 | Sparse bipartite interface with RBF, donor/acceptor, charge, orientation cosines | done | |
| 5.1 | Node features | partial | missing period/group, radical flag, and the 5 chemistry flags as *node* inputs (they only enter interface edges). Protein bonds come from `proximityBonding`, so protein aromaticity/hybridization/bond-type features are effectively constant |
| 5.2 | Sparse-CSR random-walk PE, `rwpe_dim: 20` | done | |
| 5.3 | Intra-edge RBF cutoff | partial | hard-coded 8.0 A in `features.py`; spatial cutoffs come from config |
| 6 | E(3)-invariant scalar path | done | rigid-transform test added (`tests/test_preprocess.py`), tolerance bounded by PDB/SDF file precision |
| 7 | Equivariant vector state | missing | config block (`model.equivariant`) also absent — Milestone 4 |
| 8 | Prelude (1 non-recurrent local layer per molecule) | done | |
| 9.2 | Sparse interface-triangle update | missing (v0.2+) | shared-endpoint mean context exists (`use_endpoint_context`), true P-P-L / P-L-L motifs do not |
| 10, 22 | Recurrent-state diagnostics | partial | only one averaged `cycle_delta` per cycle; no per-state deltas, norm mean/std, cosine similarity, Dirichlet energy |
| 11 | Readout `[H_L_global, H_interface, H_P_contact, H_L_contact]` | done | sum pooling |
| 12 | Variable-T training, per-batch sampling | done | probs differ from SPEC example (`[.25,.25,.20,.20,.10]` vs `[.20,.25,.25,.20,.10]`) |
| 13 | Huber/MSE/MAE, AdamW + cosine, grad clip, AMP | done | fixed here: bf16 autocast default (`train.amp_dtype`), fp16 keeps GradScaler; loss computed in fp32 |
| 13 | lr / epochs / patience | partial | config has lr 2e-4, 100 epochs, patience 20; SPEC suggests 1e-4 / 200 / 25 |
| 14.1 | Manifest schema `id,protein_path,ligand_path,pKd,split` | done | fixed here: `pKd` accepted as alias of `affinity`; `id` read as string (PDB codes like `1e10`) |
| 14.2 | PDBbind helper, non-random split | done | fixed here: `--split-mode casf2016` (test = CASF-2016 core set), metadata columns (year, resolution, Kd/Ki/IC50, relation, refined/general) |
| 14.3 | 8 A residue-level pocket | done | fixed here: `pocket_path` (PDBbind 10 A pocket file) is used when present; verified identical atom selection on 4 of 4 sampled complexes that have a pocket file |
| 14.4 | Cached graphs + preprocessing metadata | done | fixed here: `preprocess_config.yaml` and `skipped.csv` written next to `processed_manifest.csv`; multiprocess build; `.sdf` -> `.mol2` fallback |
| 18 | Parameter count / recycle count / wall time / peak memory | partial | fixed here: logged per epoch in `history.json`, `wall_ms` per T in `bapred2-eval`; FLOPs not estimated |
| 19 | CLI flow | done | see README |
| 20 | Config structure | partial | file uses `graph:` not `data:`; `seed` lives under `train`; no `interface_triangle` / `equivariant` blocks |
| 21.1-2 | Preprocess tiny synthetic complex, NaN/Inf | done | fixed here |
| 21.3-4 | Forward at several T, backward | done | |
| 21.5 | Rigid-transform invariance | done | fixed here |
| 21.6 | Equivariant rotation test | n/a | no equivariant mode yet |
| 21.7 | Batch > 1 | done | |
| 21.8 | Empty / low-contact interface | partial | preprocessing rejects zero-contact complexes; model tolerates one-contact graphs in a batch (test added). A zero-contact graph inside a batch is untested |
| 21.9 | Checkpoint save/load | done | fixed here: checkpoint stores `config` + `feature_dims`; `bapred2-eval` rebuilds the model without a manifest sample |

## Milestone 1+ (not started)

- **M1** BA-Pred split reproduction, controlled non-recurrent baseline (Stage B), memory/runtime profile.
- **M2** recycle-scaling analysis needs the SPEC 10/22 diagnostics above; `bapred2-eval` already sweeps T and records wall time.
- **M3** shared-endpoint context exists; triangle updates missing.
- **M4** equivariant gating missing entirely.
- **M5** only CASF-2016 core and random splits exist. Protein-cluster, ligand-scaffold, combined OOD and temporal splits are not implemented; the manifest now carries `year`, `affinity_type` and `affinity_relation` so temporal and censored-label filtering can be added without re-parsing the index.

## Data notes (PDBbind v2020 on this machine)

- `general-set` holds all 19443 indexed complexes; `refined-set` (5316) and `v2020-other-PL` (14127) are overlapping copies, so the manifest is built from `general-set` only.
- One entry (`10gs`) has no `*_pocket.pdb`; it falls back to the full protein file. One `.ipynb_checkpoints` directory exists and is skipped.
- 504 labels are censored (`>`, `<`, `~`, `<=`, `>=`); they are kept with the reported value. Decide whether to drop or down-weight them before reporting benchmark numbers.
