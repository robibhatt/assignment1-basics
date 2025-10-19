from dataclasses import asdict, dataclass
import torch
import yaml


DTYPE_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "long": torch.long,
    "int32": torch.int32,
    "int": torch.int32,
    "int16": torch.int16,
    "short": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


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
    dtype: torch.dtype
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    opt_eps: float
    val_batches: int
    max_norm: float
    grad_clip_eps: float
    alpha_min: float
    alpha_max: float
    warmup_steps: int


def to_yaml(cfg: TrainConfig,
            yamlfile: str):
    with open(yamlfile, "w") as f:
        safedict = asdict(cfg)
        safedict['dtype'] = str(safedict['dtype']).replace("torch.", "")
        yaml.safe_dump(safedict, f, sort_keys=False)


def from_yaml(yamlfile: str)->TrainConfig:
    with open(yamlfile) as f:
        data = yaml.safe_load(f)

        # some type coersion 
        data["lr"] = float(data["lr"])
        data["opt_eps"] = float(data["opt_eps"])
        data["grad_clip_eps"] = float(data["grad_clip_eps"])
        data["weight_decay"] = float(data["weight_decay"])
        data["betas"] = tuple(float(b) for b in data["betas"])
        data["dtype"] = DTYPE_MAP[str(data["dtype"]).lower()]
        data['alpha_max'] = float(data['alpha_max'])
        data['alpha_min'] = float(data['alpha_min'])

        return TrainConfig(**data)
