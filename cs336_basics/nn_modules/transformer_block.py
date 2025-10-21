import torch
import torch.nn as nn
from torch import Tensor
import einops
from jaxtyping import Bool, Float, Int
from cs336_basics.nn_modules.rms_norm import RMSNorm
from cs336_basics.nn_modules.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.nn_modules.swiglu import SwiGLU
from cs336_basics.nn_modules.rope import RoPE

class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope:nn.Module,
        device:torch.device | None = None,
        dtype:torch.dtype | None = None):

        # initiailize the super module, always gotta do bruv
        super().__init__()

        self.ln1 = RMSNorm(d_model=d_model,
                           device=device,
                           dtype=dtype)
                
        self.attn = MultiHeadSelfAttention(d_model=d_model,
                                           num_heads=num_heads,
                                           rope=rope,
                                           device=device,
                                           dtype=dtype)
        
        self.ln2 = RMSNorm(d_model=d_model,
                    device=device,
                    dtype=dtype)
        
        self.ffn = SwiGLU(d_model=d_model,
                          d_ff=d_ff,
                          device=device,
                          dtype=dtype)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:

        # add the attention in
        x = x + self.attn(self.ln1(x))

        # add the feed forward in
        return x + self.ffn(self.ln2(x))
    



