import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
from cs336_basics.nn_modules.rms_norm import RMSNorm
from cs336_basics.nn_modules.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.nn_modules.swiglu import SwiGLU
from cs336_basics.nn_modules.embedding import Embedding
from cs336_basics.nn_modules.transformer_block import TransformerBlock
from cs336_basics.nn_modules.linear import Linear

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
        dtype:torch.dtype | None = None):

        # initiailize the super module, always gotta do bruv
        super().__init__()

        self.token_embeddings = Embedding(num_embeddings=vocab_size,
                                          embedding_dim=d_model,
                                          device=device,
                                          dtype=dtype)
        
        transformer_blocks = []
        for _ in range(num_layers):
            transformer_blocks.append(TransformerBlock(d_model=d_model,
                                                       num_heads=num_heads,
                                                       d_ff=d_ff,
                                                       max_seq_len=context_length,
                                                       theta=rope_theta,
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
        

    def forward(self, 
        in_indices: Int[Tensor, " b seq_len"]) -> Float[Tensor, " b seq_len vocab_size"]:
        
        resid_stream = self.token_embeddings(in_indices)
        resid_stream = self.layers(resid_stream)
        resid_stream = self.ln_final(resid_stream)
        return self.lm_head(resid_stream)
        
    



