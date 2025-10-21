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
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma,
            a=-3*sigma,
            b=3*sigma
        )

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einops.einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
    
    def set_weights(self, weights: Float[Tensor, "d_out d_in"])->None:
        """
        Copies weights
        """
        assert(weights.dtype == self.weight.dtype)
        with torch.no_grad():
            self.weight.copy_(weights)



