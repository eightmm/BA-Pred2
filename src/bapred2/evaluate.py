from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from bapred2.config import config_from_dict, load_config
from bapred2.data.dataset import ProcessedComplexDataset
from bapred2.model import model_from_checkpoint
from bapred2.train import count_parameters, metrics, run_eval

DEFAULT_EPS_PRED = (0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01)
DEFAULT_EPS_STATE = (1.00, 0.50, 0.30, 0.20, 0.10, 0.05, 0.02)


def _stop_indices(signal: np.ndarray, eps: float, first_usable: int) -> np.ndarray:
    """Index of the first cycle whose signal falls below ``eps`` (last cycle if it never does).

    ``signal`` is [T, N] aligned so that column t is the quantity observed *at* cycle t+1;
    ``first_usable`` is the earliest index the rule may stop at (1 for prediction deltas,
    which need two readouts to compare).
    """
    n_cycles, n = signal.shape
    stop = np.full(n, n_cycles - 1, dtype=int)
    below = signal < eps
    resolved = np.zeros(n, dtype=bool)
    for t in range(first_usable, n_cycles):
        take = below[t] & ~resolved
        stop[take] = t
        resolved |= take
    return stop


def _summarise(P: np.ndarray, y: np.ndarray, stop: np.ndarray, n_cycles: int) -> dict:
    pred = P[stop, np.arange(P.shape[1])]
    cycles = stop + 1
    out = metrics(y.tolist(), pred.tolist())
    out["mean_cycles"] = float(cycles.mean())
    out["median_cycles"] = float(np.median(cycles))
    out["max_cycles_hit_frac"] = float((cycles == n_cycles).mean())
    out["cycle_histogram"] = {str(c): int((cycles == c).sum()) for c in sorted(set(cycles.tolist()))}
    return out


def adaptive_eval(model, loader, device, max_recycles: int, eps_pred=DEFAULT_EPS_PRED, eps_state=DEFAULT_EPS_STATE) -> dict:
    """Per-complex early exit: run ``max_recycles`` cycles once, then stop each complex at its own
    convergence point. Because the model is deterministic in eval mode and the cycles are sequential,
    reading out at cycle t is identical to having run with ``recycles=t``, so one pass yields every
    fixed-T result plus any stopping policy over the same trajectory."""
    model.eval()
    ids, ys, preds, deltas = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, aux = model(batch, recycles=max_recycles, return_trace=True)
            ids.extend(list(batch.sample_id))
            ys.extend(batch.y.reshape(-1).cpu().tolist())
            preds.append(aux["cycle_pred"].float().cpu())
            deltas.append(((aux["delta_p_graph"] + aux["delta_l_graph"] + aux["delta_q_graph"]) / 3.0).float().cpu())
    P = torch.cat(preds, dim=1).numpy()
    D = torch.cat(deltas, dim=1).numpy()
    y = np.asarray(ys, dtype=float)
    n_cycles = P.shape[0]

    fixed = {str(t + 1): metrics(y.tolist(), P[t].tolist()) for t in range(n_cycles)}
    best_fixed = min(fixed, key=lambda k: fixed[k]["rmse"])

    pred_delta = np.zeros_like(P)
    pred_delta[0] = np.inf  # cycle 1 has nothing to compare against
    pred_delta[1:] = np.abs(np.diff(P, axis=0))

    adaptive = {"pred_delta": {}, "state_delta": {}}
    for eps in eps_pred:
        adaptive["pred_delta"][str(eps)] = _summarise(P, y, _stop_indices(pred_delta, eps, 1), n_cycles)
    for eps in eps_state:
        adaptive["state_delta"][str(eps)] = _summarise(P, y, _stop_indices(D, eps, 0), n_cycles)

    err = np.abs(P - y)
    oracle_idx = err.argmin(axis=0)
    oracle = _summarise(P, y, oracle_idx, n_cycles)
    oracle["note"] = "upper bound only: picks each complex's best cycle using the label, not achievable at inference"

    return {
        "n": len(ids),
        "max_recycles": n_cycles,
        "fixed": fixed,
        "best_fixed_T": best_fixed,
        "adaptive": adaptive,
        "oracle_per_complex_T": oracle,
        "prediction_movement": {
            "mean_abs_first_to_last": float(np.abs(P[-1] - P[0]).mean()),
            "median_abs_first_to_last": float(np.median(np.abs(P[-1] - P[0]))),
            "corr_movement_vs_error_at_T1": float(np.corrcoef(np.abs(P[-1] - P[0]), err[0])[0, 1]),
        },
        "per_complex": [
            {"id": i, "y": float(yy), "preds": [float(v) for v in P[:, k]], "state_delta": [float(v) for v in D[:, k]]}
            for k, (i, yy) in enumerate(zip(ids, y))
        ],
    }


def _print_adaptive(res: dict) -> None:
    bf = res["best_fixed_T"]
    print(f"n={res['n']} max_recycles={res['max_recycles']} | best fixed T={bf} rmse {res['fixed'][bf]['rmse']:.3f} r {res['fixed'][bf]['pearson']:.3f}")
    print("fixed-T rmse: " + " ".join(f"T{t}={res['fixed'][t]['rmse']:.3f}" for t in sorted(res["fixed"], key=int)))
    for rule, entries in res["adaptive"].items():
        print(f"adaptive stop on {rule}:")
        for eps, r in entries.items():
            print(f"  eps={eps:>5}  rmse {r['rmse']:.3f}  mae {r['mae']:.3f}  r {r['pearson']:.3f}  rho {r['spearman']:.3f}"
                  f"  mean cycles {r['mean_cycles']:5.2f}  median {r['median_cycles']:4.1f}  hit-max {r['max_cycles_hit_frac']:.0%}")
    o = res["oracle_per_complex_T"]
    print(f"oracle per-complex T (label-cheating upper bound): rmse {o['rmse']:.3f}  mean cycles {o['mean_cycles']:.2f}")
    m = res["prediction_movement"]
    print(f"prediction movement first->last cycle: mean {m['mean_abs_first_to_last']:.3f}  median {m['median_abs_first_to_last']:.3f}"
          f"  corr(movement, |err@T=1|) {m['corr_movement_vs_error_at_T1']:+.3f}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate BA-Pred2: fixed-T sweep, or per-complex adaptive early exit")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="override the config stored in the checkpoint (model dims still come from the checkpoint)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--recycles", default="1,2,3,4,6,8,12,16")
    ap.add_argument("--adaptive", action="store_true", help="per-complex early exit: run --max-recycles once and stop each complex at its own convergence point")
    ap.add_argument("--max-recycles", type=int, default=16, help="cycle budget for --adaptive")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write the result (and per-sample predictions) as JSON")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = load_config(args.config) if args.config else config_from_dict(ckpt["config"])
    model = model_from_checkpoint(ckpt, cfg.model).to(device)
    ds = ProcessedComplexDataset(args.manifest, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size or cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    header = {"checkpoint": str(Path(args.checkpoint).resolve()), "epoch": ckpt.get("epoch"), "split": args.split, "n": len(ds), "n_params": count_parameters(model)}

    if args.adaptive:
        results = {**header, **adaptive_eval(model, loader, device, args.max_recycles)}
        _print_adaptive(results)
    else:
        results = {**header, "sweep": {}}
        for r in [int(x) for x in args.recycles.split(",")]:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if start is not None:
                start.record()
            res = run_eval(model, loader, device, r, return_predictions=args.out is not None)
            if end is not None:
                end.record()
                torch.cuda.synchronize(device)
                res["wall_ms"] = round(start.elapsed_time(end), 1)
            results["sweep"][str(r)] = res
            print(json.dumps({"recycles": r, **{k: v for k, v in res.items() if k != "predictions"}}))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
