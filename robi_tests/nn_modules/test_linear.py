import torch
from cs336_basics.nn_modules.linear import Linear


def test_linear():
    my_module = Linear(in_features=3,
                       out_features=4,
                       device='cpu',
                       dtype=torch.float16)
    x = torch.randn(5, 3,
                    dtype=torch.float16,
                    device='cpu')
    my_module(x)
