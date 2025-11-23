# config.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import yaml

# ---------------------------
# Sub-configs (clean + minimal)
# ---------------------------

@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    dtype: str                 # e.g. "float32", "bfloat16", "float16"
    weight_tying: bool

@dataclass
class DataConfig:
    train_text_path: str
    val_text_path: str
    tokenizer_dir: str
    num_workers: int
    pretoken_chunk_size: int
    special_tokens: List[str]
    tokenizer_chunk_size: int
    batch_size: int

@dataclass
class OptimConfig:  # optimizer + scheduler
    use_muon: bool
    lr: float
    betas: List[float]
    weight_decay: float
    opt_eps: float
    max_norm: float
    grad_clip_eps: float
    total_step_count: int
    warmup_steps: int
    alpha_min: float
    alpha_max: float


@dataclass
class OutputConfig:  # output + eval + checkpoint + wandb control
    debug: bool
    profile: bool
    seed: int
    home_dir: str
    checkpoint_interval: int
    log_interval: int
    val_batches: int
    use_wandb: bool  # whether to log to W&B at all
    wandb_entity: str

# ---------------------------
# Top-level TrainConfig
# ---------------------------

@dataclass
class TrainConfig:
    out: OutputConfig
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig

# ---------------------------
# YAML helpers
# ---------------------------

def to_yaml(cfg: TrainConfig, yamlfile: str) -> None:
    """Dump config (nested dataclasses) to YAML."""
    with open(yamlfile, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)

def from_yaml(yamlfile: str) -> TrainConfig:
    """Load YAML and rebuild nested dataclasses (no backward-compat shims)."""
    with open(yamlfile) as f:
        d: Dict[str, Any] = yaml.safe_load(f) or {}

    return TrainConfig(
        out=OutputConfig(**d["out"]),
        model=ModelConfig(**d["model"]),
        data=DataConfig(**d["data"]),
        optim=OptimConfig(**d["optim"]),
    )

# ---------------------------
# W&B helper: flatten nested dicts
# ---------------------------

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Recursively flatten nested dicts into a single-level dict with dotted keys.
      {"model": {"num_layers": 12}} -> {"model.num_layers": 12}
    """
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        nk = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_dict(v, nk, sep))
        else:
            flat[nk] = v
    return flat
