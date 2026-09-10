# Architecture review (2026-09-10)

Assessment of the BA-Pred2 model as implemented in `src/bapred2/model/{model,layers}.py` and
`src/bapred2/data/features.py`, with two probes run on real PDBbind graphs (`runs/smoke/best.pt`, 2 epochs):

- **Probe 1** - protein-side chemistry flags on 1658 pocket atoms from 6 complexes.
- **Probe 2** - hidden-state norms per recycle for 16 CASF-core graphs at T = 16.

Probe evidence is from an under-trained checkpoint; the *mechanisms* it exposes are structural.

## What is done well

1. **Static/dynamic separation is real, not nominal.** `h^0`, `q^0`, `z` are computed once and enter every
   update MLP by concatenation (`u = [LN h, M, h^0]`); the raw contact graph is never rewritten. This is the
   core SPEC 3.1 idea and it is implemented consistently across prelude, cross, intra and interface stages.
2. **Bounded, pre-norm updates with LayerScale.** Every stage is `x + s * sigmoid(gate) * MLP(LN inputs)` with
   `s` initialised at 0.1, LayerNorm only, no BatchNorm. The interface update is a gated interpolation
   `q + s*g*(q_hat - q)`. Probe 2 confirms `q` converges: `|Δq|` falls 2.5 → 0.26 over 16 cycles.
3. **Sparse everywhere.** Contacts are an explicit bipartite edge set; cross attention normalises per
   destination over its incoming contacts (`sigmoid / Σ sigmoid`), so cost is linear in contacts and a node
   with 40 contacts is not louder than one with 4. No dense pair tensor anywhere (SPEC 23).
4. **Readout keeps the interface.** `w = sigmoid(W q)` per contact gives a learned contact importance, used
   both to pool `q` and to contact-weight the protein/ligand atom pools. This is the interpretable quantity
   SPEC 25.7 asks for and it costs nothing extra.
5. **Invariance by construction.** Only distances, RBFs and two local-orientation cosines reach the network;
   the rigid-transform test passes at file precision. No coordinate updates, no absolute positions.
6. **Recurrence is trained as an operator.** T is sampled per batch from {2,3,4,6,8}; the evaluator sweeps
   T at fixed weights and records wall time. The scientific question of the project is answerable from
   the existing CLI without new code.
7. **Small, testable surface.** ~220 lines of model code, 8.7 M parameters (60 % in the shared core, 25 % in
   the prelude, 8 % readout), checkpoints carry config + feature dims, 8 tests cover preprocessing,
   invariance, low-contact batches and checkpoint round-trip.

## What is weak, ranked by expected impact on affinity accuracy

### 1. Protein chemistry features are largely wrong (data side, highest impact)

Proteins are parsed with `MolFromPDBFile(proximityBonding=True)` and no residue templates, so RDKit has no
formal charges and guesses implicit hydrogens. Probe 1 on real pockets:

| expectation | flagged |
|---|---|
| charged side-chain atoms (LYS NZ, ARG NH1/NH2/NE, ASP OD1/OD2, GLU OE1/OE2): 64 | 12 cationic/anionic |
| backbone N as H-bond donor (non-PRO): 197 | 68 |
| SER OG, TYR OH, TRP NE1, ARG NH1 as donors | 0 % donor in every case |

So on the protein side the `hbd`, `cationic`, `anionic` flags that feed the interface features `e_PL`
(donor/acceptor compatibility, charge product) are mostly zero or wrong, and `formal_charge`, `total_h`,
`hybridization` node features carry little signal. Aromatic/ring flags are fine (ring perception works on
proximity bonds). This is the single change most likely to move CASF numbers.

**Fix:** derive protein atom chemistry from residue + atom name (a 20-residue template table: donors,
acceptors, charged groups, aromatic rings) instead of RDKit valence inference, and add a residue-type
embedding to protein node features. Keep RDKit only for coordinates and connectivity. Ligand features are
unaffected (SDF/MOL2 carry bonds and charges).

### 2. Node states drift with depth; the interface state converges but atoms do not

Probe 2 (smoke checkpoint, T = 16):

| t | ‖h_P‖ | ‖h_L‖ | ‖q‖ | ‖Δh_L‖ | ‖Δq‖ |
|---|---|---|---|---|---|
| 0 | 15.4 | 15.9 | 16.0 | – | – |
| 8 | 15.4 | 18.7 | 13.2 | 0.91 | 0.51 |
| 16 | 17.1 | 23.3 | 12.2 | 0.83 | 0.26 |

`q` is a convex interpolation and settles. `h_P`, `h_L` are additive residuals (`h + s*g*Δ`) and grow
roughly linearly; the per-cycle step barely shrinks. Two consumers see the raw, growing states:
`interface_weight(q)` and the readout pools (`global_add_pool(hl)`, `p_strength * hp`), whose scale then
depends on T. This is the mechanism behind the smoke sweep degrading past T = 8 (RMSE 2.08 → 2.20 at
T = 16) and it will cap the "test-time recycle scaling" result the project is built to show. Mean
pairwise cosine of `h_L` stays ≈ 0.02, so this is drift, **not** oversmoothing.

**Fix (cheap, in order):** LayerNorm the states entering `interface_weight` and the readout pools; make the
node update an interpolation like `q` (or bound `s` via `sigmoid(param)` so `s*g <= 1`); add per-state
norm/delta logging to `return_aux` so this is visible during training (SPEC 10).

### 3. Readout pools are on different scales

`q_pool` sums over hundreds of contacts, `l_pool` over 10-40 ligand atoms, `p_contact` over hundreds of
pocket atoms weighted by contact strength. One LayerNorm over the 1024-d concatenation normalises the
whole vector, not each block, so the block with the largest magnitude dominates and the balance shifts
with pocket size and T. **Fix:** LayerNorm per pooled block (or mean pooling for `q`), then concat; ablate
attention pooling as SPEC 11 suggests.

### 4. Dropout inside the shared recurrent core

`MLP` hidden layers carry `dropout = 0.1` and the same block runs up to 8 times; noise compounds through
the recursion and differs between T = 2 and T = 8 batches. Prefer dropout in prelude/readout only, or
`<= 0.05` inside the core. Untested claim, easy ablation.

### 5. Endpoint context is not a triangle update

`use_endpoint_context` adds `mean_j q_ij` and `mean_i q_ij` to the interface update. It ignores the
intra-molecular relation between the two neighbouring contacts (`z_lk`, `z_pk`), which is what carries
"cooperative contact" geometry (SPEC 9.2). Fine for Milestone 0, but do not report it as the triangle
result.

### 6. Missing node signal that costs nothing

The five chemistry flags (HBA/HBD/cation/anion/hydrophobic) exist in `atom_property_masks` but only enter
interface edges, never node features; residue identity is absent; period/group is absent (SPEC 5.1). Once
item 1 is fixed, concatenating the flags and a residue embedding into `x` is a one-line change.

### 7. Smaller items

- LayerScale `s` is unconstrained; after training it can exceed 1 and turn the `q` interpolation into an
  overshoot. Bounding it is part of fix 2.
- Gates are full `Linear(3d, d)` layers: ~0.6 M of the 5.8 M core parameters do gating. Per-channel scalar
  gates would cut parameters without touching the mechanism.
- Intra-edge RBF spans 0-8 Å while spatial edges stop at 5 Å / 4.5 Å; a third of the RBF centres are never
  hit. Tie the RBF cutoff to the spatial cutoff (SPEC 5.3).
- `max_spatial_neighbors` is one value (32) for both molecules; SPEC asks 24 for ligands.
- Pocket residues within 8 Å produce a disconnected covalent graph; RWPE is computed per fragment, so
  protein `pos_enc` mostly encodes "backbone vs side chain" rather than pocket topology. Acceptable, but
  it is not a strong anchor for the protein side.

## Suggested order

1. Residue-template protein chemistry + residue embedding (data, retrain).
2. LN before readout/interface weight, bounded node updates, norm logging (model, retrain).
3. Per-block readout normalisation; dropout out of the core (ablations).
4. Then the recycle-scaling and triangle/equivariance experiments on top of a stable baseline.

Items 1-2 change preprocessing and the checkpoint format; run them as a new config (`bapred2_v0.2.yaml`)
so `runs/base` remains the Milestone-0 reference.
