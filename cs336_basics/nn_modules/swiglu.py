import torch
import torch.nn as nn
from torch import Tensor
import einops
from jaxtyping import Bool, Float, Int
from cs336_basics.nn_modules.linear import Linear

class SwiGLU(nn.Module):

    def __init__(self, 
        d_model: int,
        d_ff:int | None = None,
        device:torch.device | None = None,
        dtype:torch.dtype | None = None):

        # initiailize the super module, always gotta do bruv
        super().__init__()
        
        # create the hidden layer dimension
        if d_ff is None:
            d_ff = int(8/ 3 * d_model)

        # create the 3 matrices that we use
        self.W_1 = Linear(in_features=d_model,
                          out_features=d_ff,
                          device=device,
                          dtype=dtype)
        # notice the dimension flip here!
        self.W_2 = Linear(in_features=d_ff,
                          out_features=d_model,
                          device=device,
                          dtype=dtype)
        # back to W_1 style
        self.W_3 = Linear(in_features=d_model,
                          out_features=d_ff,
                          device=device,
                          dtype=dtype)
        

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        z = self.W_1(x)
        return self.W_2((z * torch.sigmoid(z)) * self.W_3(x))
    
    def set_weights(self, w_1:Float[Tensor, "d_ff d_model"],
                          w_2:Float[Tensor, "d_model d_ff"],
                          w_3:Float[Tensor, "d_ff d_model"])->None:
        """
        Copies weights
        """
        self.W_1.set_weights(weights=w_1)
        self.W_2.set_weights(weights=w_2)
        self.W_3.set_weights(weights=w_3)


