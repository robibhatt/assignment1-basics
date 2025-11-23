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
    return


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
)->int:
    
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']

def save_checkpoint_multi(
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
)->None:

    obj = {}
    obj['model'] = model.state_dict()
    obj['optimizers'] = [optimizer.state_dict() for optimizer in optimizers]
    obj['iteration'] = iteration

    torch.save(obj, out)
    return


def load_checkpoint_multi(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
)->int:
    
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    for i, optimizer in enumerate(optimizers):
        # load optimizers in order
        optimizer.load_state_dict(obj['optimizers'][i])

    return obj['iteration']