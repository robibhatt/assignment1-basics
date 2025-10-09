import torch
import torch.nn as nn
from torch import Tensor
import einops
from jaxtyping import Bool, Float, Int

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, 
        d_model: int,
        num_heads:int,
        device:torch.device | None = None,
        dtype:torch.dtype | None = None):

        # initiailize the super module, always gotta do bruv
        super().__init__()
        
        d_v = d_model // num_heads

        # get initialization variance
        sigma = (2. / (d_model + d_v)) ** (0.5)

        self.W_Q = nn.Parameter(torch.empty(num_heads, d_v, d_model, dtype=dtype, device=device))
        nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=sigma,
            a=-3*sigma,
            b=3*sigma
        )

        self.W_Q = nn.Parameter(torch.empty(num_heads, d_v, d_model, dtype=dtype, device=device))
        nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=sigma,
            a=-3*sigma,
            b=3*sigma
        )

        self.W_Q = nn.Parameter(torch.empty(num_heads, d_v, d_model, dtype=dtype, device=device))
        nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=sigma,
            a=-3*sigma,
            b=3*sigma
        )

        self.W_O = nn.Parameter(torch.empty(num_heads, d_v, d_model, dtype=dtype, device=device))
        nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=sigma,
            a=-3*sigma,
            b=3*sigma
        )


    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einops.einsum(self.W, x, "... d_out d_in, ... d_in -> ... d_out")
    
    def set_weights(self, weights: Float[Tensor, "d_out d_in"])->None:
        """
        Copies weights
        """
        assert(weights.dtype == self.W.dtype)
        with torch.no_grad():
            self.W.copy_(weights)



