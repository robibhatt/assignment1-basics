import torch
import torch.nn as nn
from torch import Tensor, LongTensor
import einops
from jaxtyping import Bool, Float, Int

class Embedding(nn.Module):

    def __init__(self, 
        num_embeddings: int,
        embedding_dim:int,
        device:torch.device | None = None,
        dtype:torch.dtype | None = None):

        # initiailize the super module, always gotta do bruv
        super().__init__()
        

        # create an initiailize the weights
        self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=1.,
            a=-3.,
            b=3.
        )

    def forward(self, token_ids: Int[Tensor, "batch_size sequence_length"]) -> Float[Tensor, "batch_size sequence_length embedding_dim"]:
        return self.W[token_ids, :]
    
    def set_weights(self, weights: Float[Tensor, "num_embeddings embedding_dim"])->None:
        """
        Copies weights
        """
        assert(weights.dtype == self.W.dtype)
        with torch.no_grad():
            self.W.copy_(weights)



