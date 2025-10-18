import numpy as np
import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int
import random


def get_batch(x: np.ndarray,
              batch_size: int,
              context_length: int,
              device: str,
    ) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    """
    Grab a sample
    """
    end_indices = [random.randint(context_length, len(x) - 1) for i in range(batch_size)]
    in_batch = [x[end-context_length:end] for end in end_indices]
    out_batch = [x[end-context_length+1:end+1] for end in end_indices]
    return (torch.from_numpy(np.stack(in_batch)).to(device=device, dtype=torch.long),
            torch.from_numpy(np.stack(out_batch)).to(device=device, dtype=torch.long))

