import torch
import torch.nn as nn
from torch import Tensor
import einops
from jaxtyping import Bool, Float, Int
from cs336_basics.torch_utils import scaled_dot_product_attention
from cs336_basics.nn_modules.rope import RoPE
from cs336_basics.nn_modules.linear import Linear

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, 
        d_model: int,
        num_heads:int,
        max_seq_len:int | None = None,
        theta:float | None = None,
        device:torch.device | None = None,
        dtype:torch.dtype | None = None):

        # initiailize the super module, always gotta do bruv
        super().__init__()

        # track the number of heads
        self.num_heads = num_heads

        # set the heads dimension
        d_v = d_model // num_heads

        # get mask
        self.mask = None
        if max_seq_len is not None:
            self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len), diagonal=0) > 0.9

        # get rope
        self.rope = None
        if max_seq_len is not None and theta is not None:
            self.rope = RoPE(theta=theta,
                             d_k=d_v,
                             max_seq_len=max_seq_len,
                             device=device)
            
        # create q k v o matrices
        self.q_proj = Linear(in_features=d_model,
                             out_features=d_model,
                             device=device,
                             dtype=dtype)
        self.k_proj = Linear(in_features=d_model,
                             out_features=d_model,
                             device=device,
                             dtype=dtype)
        self.v_proj = Linear(in_features=d_model,
                             out_features=d_model,
                             device=device,
                             dtype=dtype)
        self.output_proj = Linear(in_features=d_model,
                             out_features=d_model,
                             device=device,
                             dtype=dtype)
        

    def forward(self, 
        x: Float[Tensor, "... seq_len d_in"], 
        token_positions: Int[Tensor, "... seq_len"] | None = None) -> Float[Tensor, "... seq_len d_out"]:

        queries = self.q_proj(x)
        queries = einops.rearrange(queries, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        keys = self.k_proj(x)
        keys = einops.rearrange(keys, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        values = self.v_proj(x)
        values = einops.rearrange(values, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)

        mask = None
        seq_len = values.shape[-2]
        if self.mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0) > 0.9
        else:
            mask = self.mask[:seq_len, :seq_len]

        if self.rope is not None:
            if token_positions is None:
                # get the token positions
                seq_len = x.shape[-2]
                token_positions = torch.arange(0, seq_len, device=x.device)
                token_positions = token_positions.expand(x.shape[:-1])
            queries = self.rope(queries, token_positions=token_positions)
            keys = self.rope(keys, token_positions=token_positions)

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
        self.q_proj.set_weights(weights=q_proj_weight)
        self.k_proj.set_weights(weights=k_proj_weight)
        self.v_proj.set_weights(weights=v_proj_weight)
        self.output_proj.set_weights(weights=o_proj_weight)





