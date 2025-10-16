import torch
import torch.nn as nn
import os
import typing


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
)->None:

    obj = {}
    obj['model'] = model.state_dict()
    obj['optimizer'] = optimizer.state_dict()
    obj['iteration'] = iteration

    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
)->int:
    
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']