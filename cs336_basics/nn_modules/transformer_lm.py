import torch
import torch.nn as nn
import einops
from torch import Tensor
from jaxtyping import Bool, Float, Int
from cs336_basics.nn_modules.rms_norm import RMSNorm
from cs336_basics.nn_modules.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.nn_modules.swiglu import SwiGLU
from cs336_basics.nn_modules.embedding import Embedding
from cs336_basics.nn_modules.transformer_block import TransformerBlock
from cs336_basics.nn_modules.linear import Linear
from cs336_basics.nn_modules.rope import RoPE

class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device:torch.device | None = None,
        dtype:torch.dtype | None = None,
        weight_tying:bool=False):

        # initiailize the super module, always gotta do bruv
        super().__init__()

        assert(d_model % num_heads == 0)

        self.token_embeddings = Embedding(num_embeddings=vocab_size,
                                          embedding_dim=d_model,
                                          device=device,
                                          dtype=dtype)
        

        # create a rope
        self.rope = RoPE(theta=rope_theta,
                         d_k=d_model // num_heads,
                         max_seq_len=context_length,
                         device=device)
        

        transformer_blocks = []
        for _ in range(num_layers):
            transformer_blocks.append(TransformerBlock(d_model=d_model,
                                                       num_heads=num_heads,
                                                       d_ff=d_ff,
                                                       rope=self.rope,
                                                       device=device,
                                                       dtype=dtype))
        self.layers = nn.Sequential(*transformer_blocks)

        self.ln_final = RMSNorm(d_model=d_model,
                                device=device,
                                dtype=dtype)
        
        self.lm_head = Linear(in_features=d_model,
                              out_features=vocab_size,
                              device=device,
                              dtype=dtype)
        
        if weight_tying:
            self.lm_head is None
        

    def forward(self, 
        in_indices: Int[Tensor, " b seq_len"]) -> Float[Tensor, " b seq_len vocab_size"]:
        
        resid_stream = self.token_embeddings(in_indices)
        resid_stream = self.layers(resid_stream)
        resid_stream = self.ln_final(resid_stream)

        # cases based on weight tying
        if self.lm_head is not None:
            result = self.lm_head(resid_stream)
        else:
            result = einops.einsum(self.token_embeddings.weight, resid_stream, "n_e e_d, ... e_d -> ... n_e")
        return result
        
    



