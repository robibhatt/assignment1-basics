import torch
import torch.nn as nn
from torch import Tensor
import einops
from jaxtyping import Bool, Float, Int
from cs336_basics.nn_modules.linear import Linear

class ReLU2(nn.Module):

    def __init__(self, 
        d_model: int,
        d_ff:int | None = None,
        device:torch.device | None = None,
        dtype:torch.dtype | None = None):

        # initiailize the super module, always gotta do bruv
        super().__init__()
        
        # create the hidden layer dimension
        if d_ff is None:
            d_ff = int(4 * d_model)

        # create the 3 matrices that we use
        self.w1 = Linear(in_features=d_model,
                          out_features=d_ff,
                          device=device,
                          dtype=dtype)
        # notice the dimension flip here!
        self.w2 = Linear(in_features=d_ff,
                          out_features=d_model,
                          device=device,
                          dtype=dtype)
        # back to W_1 style
        #self.w3 = Linear(in_features=d_model,
                          #out_features=d_ff,
                          #device=device,
                          #dtype=dtype)
        return
        

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        z = self.w1(x)
        z = torch.max(torch.tensor(0.0,dtype=dtype),z)
        z = z*z
        return self.w2(z) 

    
    def set_weights(self, w_1:Float[Tensor, "d_ff d_model"],
                          w_2:Float[Tensor, "d_model d_ff"],
                          #w_3:Float[Tensor, "d_ff d_model"]
                          )->None:
        """
        Copies weights
        """
        self.w1.set_weights(weights=w_1)
        self.w2.set_weights(weights=w_2)
        #self.w3.set_weights(weights=w_3)
        return

