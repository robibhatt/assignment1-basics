import torch
import torch.nn as nn
from torch import Tensor
import einops
from jaxtyping import Bool, Float, Int

class RoPE(nn.Module):

    def __init__(self, 
        theta: float,
        d_k:int,
        max_seq_len:int,
        device:torch.device | None = None,):

        # initiailize the super module, always gotta do bruv
        super().__init__()
        
        # we can remove this check later
        assert(d_k % 2 == 0)

        numerators = torch.arange(start=0,
                                  end=max_seq_len,
                                  step=1,
                                  device=device,
                                  dtype=torch.float32)
        numerators = numerators.unsqueeze(dim=1)
        denominators = torch.arange(start=0,
                                    end=d_k,
                                    step=2,
                                    device=device,
                                    dtype=torch.float32) / d_k
        denominators = torch.pow(theta, denominators).unsqueeze(dim = 0)

        angles = numerators / denominators
        cos = torch.cos(angles).unsqueeze(-1).unsqueeze(-1)
        sin = torch.sin(angles).unsqueeze(-1).unsqueeze(-1)

        top_rows = torch.cat(tensors=[cos, -sin], 
                             dim=-1)
        bot_rows = torch.cat(tensors=[sin, cos],
                             dim=-1)
        R = torch.cat(tensors=[top_rows, bot_rows],
                      dim=-2)
        
        assert(R.shape == (max_seq_len, d_k//2, 2, 2))

        self.register_buffer(name='R',
                             tensor=R,
                             persistent=False)
        
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool))
        self.register_buffer(name='mask',
                             tensor=mask,
                             persistent=False)
    

    def forward(self, x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"] | None = None) -> Float[Tensor, "... seq_len d_k"]:

        x_dtype = x.dtype
        x = x.to(torch.float32)
        *lead, d_k = x.shape
        if token_positions is not None:
            answer = einops.einsum(self.R[token_positions], x.view(*lead, d_k//2, 2), "... sq dk2 r c, ... sq dk2 c -> ... sq dk2 r")
        else:
            seq_len = x.shape[-2]
            answer = einops.einsum(self.R[:seq_len], x.view(*lead, d_k//2, 2), "sq dk2 r c, ... sq dk2 c -> ... sq dk2 r")
        return einops.rearrange(answer, "... d r -> ... (d r)").to(x_dtype)
        



