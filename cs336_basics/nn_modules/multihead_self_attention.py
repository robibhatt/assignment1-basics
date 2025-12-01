import torch
import torch.nn as nn
from torch import Tensor
import einops
from jaxtyping import Bool, Float, Int
from cs336_basics.torch_utils import scaled_dot_product_attention
from cs336_basics.nn_modules.rope import RoPE
from cs336_basics.nn_modules.linear import Linear
from cs336_basics.nn_modules.rms_norm import RMSNorm

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, 
        d_model: int,
        num_heads:int,
        rope:nn.Module,
        device:torch.device | None = None,
        dtype:torch.dtype | None = None,
        use_rope:bool=True):

        # initiailize the super module, always gotta do bruv
        super().__init__()

        # silly rope param in case we don't wanna use rope for some reason
        self.use_rope=use_rope


        # track the number of heads
        self.num_heads = num_heads

        # set rope
        self.rope = rope
            
        # create q k v o matrices
        self.output_proj = Linear(in_features=d_model,
                             out_features=d_model,
                             device=device,
                             dtype=dtype)
        
        # get initialization variance
        sigma = (1. / (d_model)) ** (0.5)

        # create and initiailize the qkv weights
        self.qkv = nn.Parameter(torch.empty(3*d_model, d_model, dtype=dtype, device=device))
        nn.init.trunc_normal_(
            self.qkv,
            mean=0.0,
            std=sigma,
            a=-3*sigma,
            b=3*sigma
        )

        self.lnq = RMSNorm(
                d_model=d_model,
                device=device,
                dtype=dtype
                )

        self.lnk = RMSNorm(
                d_model=d_model,
                device=device,
                dtype=dtype
                )
            
        

    def forward(self, 
        x: Float[Tensor, "... seq_len d_in"], 
        token_positions: Int[Tensor, "... seq_len"] | None = None) -> Float[Tensor, "... seq_len d_out"]:

        qkvx = einops.einsum(self.qkv, x, "three_d_model d_model, ... seq_len d_model -> ... seq_len three_d_model")
        queries, keys, values = torch.chunk(input=qkvx,
                                            chunks=3,
                                            dim=-1)
        queries = einops.rearrange(queries, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        keys = einops.rearrange(keys, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        values = einops.rearrange(values, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)

        # handle the masking
        seq_len = values.shape[-2]
        mask = self.rope.mask[:seq_len, :seq_len]

        if self.use_rope:
            queries = self.rope(queries, token_positions=token_positions)
            keys = self.rope(keys, token_positions=token_positions)

        queries = self.lnq(queries)

        keys = self.lnk(keys)

        new_values = scaled_dot_product_attention(Q=queries, K=keys, V=values, mask=mask)
        """ new values have shape ... h seq_len d_v"""
        new_values = einops.rearrange(new_values, "... h seq_len d_v -> ... seq_len (h d_v)")
        return self.output_proj(new_values)

    
    def set_weights(self, q_proj_weight: Float[Tensor, " hd_k d_in"],
                          k_proj_weight: Float[Tensor, " hd_k d_in"],
                          v_proj_weight: Float[Tensor, " hd_v d_in"],
                          o_proj_weight: Float[Tensor, " d_model hd_v"])->None:
        """
        Copies weights
        """
        with torch.no_grad():
            self.qkv.copy_(torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0))
        self.output_proj.set_weights(weights=o_proj_weight)





