import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

class RMSNorm(nn.Module):

    def __init__(self, 
        d_model: int,
        eps:float=1e-5,
        device:torch.device | None = None,
        dtype:torch.dtype | None = None):

        # initiailize the super module, always gotta do bruv
        super().__init__()
        
        self.d_model=d_model
        self.eps=eps

        # create an initiailize the gains
        self.gains = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x_dtype = x.dtype
        x = x.to(torch.float32)
        x_norm = torch.sqrt(torch.mean(x*x, dim=-1, keepdim=True) + self.eps)
        result = self.gains * x / x_norm
        return result.to(dtype=x_dtype)
    
    def set_weights(self, weights: Float[Tensor, "d_model"])->None:
        """
        Copies weights
        """
        assert(weights.dtype == self.gains.dtype)
        with torch.no_grad():
            self.gains.copy_(weights)
        
    



