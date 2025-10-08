import torch
import torch.nn as nn
from torch import Tensor
import einops
from jaxtyping import Bool, Float, Int

class Linear(nn.Module):

    def __init__(self, 
        in_features: int,
        out_features:int,
        device:torch.device | None = None,
        dtype:torch.dtype | None = None):

        # initiailize the super module, always gotta do bruv
        super().__init__()
        
        # get initialization variance
        sigma = (2. / (in_features + out_features)) ** (0.5)

        # create an initiailize the weights
        self.W = torch.empty(out_features, in_features, dtype=dtype, device=device)
        nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=sigma,
            a=-3*sigma,
            b=3*sigma
        )

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einops.einsum(self.W, x, "... d_out d_in, ... d_in -> ... d_out")



