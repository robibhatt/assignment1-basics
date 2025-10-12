from collections.abc import Callable, Iterable
from typing import Optional, Any
import torch
import torch.nn as nn
import math

def lr_cosine_schedule(t: int, 
                       alpha_max: float,
                       alpha_min: float,
                       T_w: int,
                       T_c: int) -> float:
    
    if t < T_w:
        return t/T_w * alpha_max
    
    elif T_w <= t <= T_c:
        return alpha_min + 1/2 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (alpha_max - alpha_min)
    
    else:
        return alpha_min


@torch.no_grad()
def clip_gradients(parameters: list[nn.Parameter],
                   max_norm: float,
                   eps: float = 1e-6):
    
    norm = math.sqrt(sum([(parameter.grad * parameter.grad).sum().item() for parameter in parameters if parameter.grad is not None]))
    if norm > max_norm:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad *= max_norm / (norm + eps)
