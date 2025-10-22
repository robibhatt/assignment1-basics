from dataclasses import asdict, dataclass
import yaml


@dataclass
class TrainConfig:
    debug: bool
    profile: bool
    seed: int
    home_dir: str
    train_text_path: str
    val_text_path: str
    tokenizer_dir: str
    num_workers: int
    pretoken_chunk_size: int
    vocab_size: int
    special_tokens: list[str]
    tokenizer_chunk_size: int
    batch_size: int
    context_length: int
    total_step_count: int
    checkpoint_interval: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    # Store dtype as a simple string like "float32"
    dtype: str
    lr: float
    # Store betas as a YAML/JSON-native list
    betas: list[float]
    weight_decay: float
    opt_eps: float
    val_batches: int
    max_norm: float
    grad_clip_eps: float
    alpha_min: float
    alpha_max: float
    warmup_steps: int
    weight_tying: bool


def to_yaml(cfg: TrainConfig, yamlfile: str) -> None:
    """Dump config directly; all fields are YAML/JSON-native."""
    with open(yamlfile, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def from_yaml(yamlfile: str) -> TrainConfig:
    """Load config directly; YAML handles numeric/boolean parsing."""
    with open(yamlfile) as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)
