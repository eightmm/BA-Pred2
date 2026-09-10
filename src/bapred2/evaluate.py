from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from bapred2.config import config_from_dict, load_config
from bapred2.data.dataset import ProcessedComplexDataset
from bapred2.model import model_from_checkpoint
from bapred2.train import count_parameters, run_eval


def main():
    ap = argparse.ArgumentParser(description="Evaluate BA-Pred2 and sweep test-time recycles")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="override the config stored in the checkpoint (model dims still come from the checkpoint)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--recycles", default="1,2,3,4,6,8,12,16")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write the sweep (and per-sample predictions) as JSON")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = load_config(args.config) if args.config else config_from_dict(ckpt["config"])
    model = model_from_checkpoint(ckpt, cfg.model).to(device)
    ds = ProcessedComplexDataset(args.manifest, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size or cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    results = {"checkpoint": str(Path(args.checkpoint).resolve()), "epoch": ckpt.get("epoch"), "split": args.split, "n": len(ds), "n_params": count_parameters(model), "sweep": {}}
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
