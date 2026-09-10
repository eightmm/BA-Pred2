from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GraphConfig:
    pocket_cutoff: float = 8.0
    interface_cutoff: float = 5.0
    protein_spatial_cutoff: float = 5.0
    ligand_spatial_cutoff: float = 4.5
    max_spatial_neighbors: int = 32
    distance_rbf_dim: int = 16
    rwpe_dim: int = 20


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    prelude_layers: int = 1
    dropout: float = 0.1
    layerscale_init: float = 0.1
    use_endpoint_context: bool = True
    train_recycles: list[int] = field(default_factory=lambda: [2, 3, 4, 6, 8])
    train_recycle_probs: list[float] = field(default_factory=lambda: [0.25, 0.25, 0.20, 0.20, 0.10])
    eval_recycles: int = 6


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    loss: str = "huber"
    huber_delta: float = 1.0
    amp: bool = True
    # "bfloat16" needs no loss scaling and is the safe default on Ampere+/Blackwell; "float16" enables GradScaler.
    amp_dtype: str = "bfloat16"
    early_stop_patience: int = 20


@dataclass
class Config:
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key: {type(obj).__name__}.{key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def config_from_dict(values: dict[str, Any]) -> Config:
    return _update_dataclass(Config(), values or {})


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    with open(path, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    return config_from_dict(values)
