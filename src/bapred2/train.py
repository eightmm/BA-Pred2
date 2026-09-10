from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from bapred2.config import Config, load_config
from bapred2.data.dataset import ProcessedComplexDataset
from bapred2.model import feature_dims_from_sample, model_from_dims

AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_recycles(values: list[int], probs: list[float]) -> int:
    if len(values) != len(probs):
        raise ValueError("train_recycles and train_recycle_probs must have the same length")
    return random.choices(values, weights=probs, k=1)[0]


def loss_fn(pred, target, name: str, delta: float):
    if name == "mse":
        return F.mse_loss(pred, target)
    if name == "mae":
        return F.l1_loss(pred, target)
    if name == "huber":
        return F.huber_loss(pred, target, delta=delta)
    raise ValueError(f"Unknown loss: {name}")


def metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    mae = float(np.mean(np.abs(a - b)))
    p = float(pearsonr(a, b).statistic) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else float("nan")
    s = float(spearmanr(a, b).statistic) if len(a) > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "pearson": p, "spearman": s}


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_eval(model, loader, device, recycles: int, return_predictions: bool = False):
    model.eval()
    ids, ys, ps, cycle_deltas, state_stats = [], [], [], [], {}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, aux = model(batch, recycles=recycles, return_aux=True)
            target = batch.y.reshape(-1)
            ys.extend(target.cpu().tolist())
            ps.extend(pred.float().cpu().tolist())
            if return_predictions:
                ids.extend(list(batch.sample_id))
            if aux["cycle_delta"].numel():
                cycle_deltas.append(aux["cycle_delta"].cpu())
                for k, v in aux.get("state_stats", {}).items():
                    state_stats.setdefault(k, []).append(v.cpu())
    out = metrics(ys, ps)
    if cycle_deltas:
        out["cycle_delta"] = torch.stack(cycle_deltas).mean(0).tolist()
        out["state_stats"] = {k: torch.stack(v).mean(0).tolist() for k, v in state_stats.items()}
    if return_predictions:
        out["predictions"] = [{"id": i, "y": y, "pred": p} for i, y, p in zip(ids, ys, ps)]
    return out


def make_checkpoint(model, cfg: Config, feature_dims: dict[str, int], epoch: int, val: dict) -> dict:
    return {"model": model.state_dict(), "config": cfg.to_dict(), "feature_dims": feature_dims, "epoch": epoch, "val": val}


def main():
    ap = argparse.ArgumentParser(description="Train BA-Pred2")
    ap.add_argument("--manifest", required=True, help="processed_manifest.csv from bapred2-preprocess")
    ap.add_argument("--config", default="configs/bapred2_base.yaml")
    ap.add_argument("--out", default="runs/bapred2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    ap.add_argument("--limit-train", type=int, default=None, help="use only the first N train graphs (smoke tests)")
    ap.add_argument("--limit-val", type=int, default=None, help="use only the first N val graphs (smoke tests)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    set_seed(cfg.train.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.yaml").write_text(yaml.safe_dump({"config": cfg.to_dict(), "manifest": str(Path(args.manifest).resolve()), "device": str(device), "args": vars(args)}, sort_keys=False))

    train_ds = ProcessedComplexDataset(args.manifest, "train", limit=args.limit_train)
    val_ds = ProcessedComplexDataset(args.manifest, "val", limit=args.limit_val)
    try:
        test_ds = ProcessedComplexDataset(args.manifest, "test")
    except ValueError:
        test_ds = None

    loader_kwargs = {"batch_size": cfg.train.batch_size, "num_workers": cfg.train.num_workers, "persistent_workers": cfg.train.num_workers > 0}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs) if test_ds else None

    feature_dims = feature_dims_from_sample(train_ds[0])
    model = model_from_dims(feature_dims, cfg.model).to(device)
    n_params = count_parameters(model)
    print(f"model parameters: {n_params:,} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds) if test_ds else 0} | device={device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    use_amp = cfg.train.amp and device.type == "cuda"
    if cfg.train.amp_dtype not in AMP_DTYPES:
        raise ValueError(f"train.amp_dtype must be one of {sorted(AMP_DTYPES)}")
    amp_dtype = AMP_DTYPES[cfg.train.amp_dtype]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    best_rmse, bad_epochs, history = math.inf, 0, []
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        losses, recycle_counts = [], []
        t_epoch = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        bar = tqdm(train_loader, desc=f"epoch {epoch:03d}")
        for batch in bar:
            batch = batch.to(device)
            recycles = sample_recycles(cfg.model.train_recycles, cfg.model.train_recycle_probs)
            optimizer.zero_grad(set_to_none=True)
            target = batch.y.reshape(-1)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if cfg.train.cycle_loss_weight > 0:
                    preds = model(batch, recycles=recycles, return_all_cycles=True).float()  # [T, B]
                    loss = loss_fn(preds[-1], target, cfg.train.loss, cfg.train.huber_delta)
                    if preds.size(0) > 1:
                        weights = torch.arange(1, preds.size(0), device=device, dtype=torch.float32) / preds.size(0)
                        early = torch.stack([loss_fn(preds[t], target, cfg.train.loss, cfg.train.huber_delta) for t in range(preds.size(0) - 1)])
                        loss = loss + cfg.train.cycle_loss_weight * (weights * early).sum() / weights.sum()
                else:
                    pred = model(batch, recycles=recycles)
                    loss = loss_fn(pred.float(), target, cfg.train.loss, cfg.train.huber_delta)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            recycle_counts.append(recycles)
            bar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}", r=recycles, gn=f"{float(grad_norm):.2f}")

        scheduler.step()
        val = run_eval(model, val_loader, device, cfg.model.eval_recycles)
        record = {
            "epoch": epoch, "train_loss": float(np.mean(losses)), "lr": scheduler.get_last_lr()[0],
            "mean_train_recycles": float(np.mean(recycle_counts)), "epoch_sec": round(time.time() - t_epoch, 1),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if device.type == "cuda" else None,
            "val": val,
        }
        if test_loader is not None:
            # Monitoring only: checkpoint selection and early stopping use the validation split.
            record["test"] = run_eval(model, test_loader, device, cfg.model.eval_recycles)
        history.append(record)
        print(json.dumps(record))
        (out_dir / "history.json").write_text(json.dumps({"n_params": n_params, "epochs": history}, indent=2))
        torch.save(make_checkpoint(model, cfg, feature_dims, epoch, val), out_dir / "last.pt")
        if val["rmse"] < best_rmse:
            best_rmse, bad_epochs = val["rmse"], 0
            torch.save(make_checkpoint(model, cfg, feature_dims, epoch, val), out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if test_loader is not None:
        ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        result = run_eval(model, test_loader, device, cfg.model.eval_recycles, return_predictions=True)
        summary = {k: v for k, v in result.items() if k != "predictions"}
        print("test", json.dumps(summary))
        (out_dir / "test.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
