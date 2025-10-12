import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int
import einops


def softmax(tensor: Float[Tensor, "..."], dim: int)-> Float[Tensor, "..."]:
    max_normalized = tensor - tensor.max(dim=dim, keepdim=True)[0]
    exp = torch.exp(max_normalized)
    return exp/exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    
    d_k = Q.shape[-1]
    qtk = einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k) ** 0.5
    
    if mask is not None:
        qtk = torch.where(mask, qtk, -torch.inf)

    return einops.einsum(softmax(qtk, dim=-1), V, "... queries keys,  ... keys d_v -> ... queries d_v")


def cross_entropy(
    o: Float[Tensor, "... vocab_size"],
    x: Int[Tensor, "..."]
)-> Float[Tensor, ""]:
    
    o -= o.max(dim=-1, keepdim=True)[0]
    output = torch.log(torch.exp(o).sum(dim=-1, keepdim=False)) - o.gather(dim=-1, index=x.unsqueeze(-1))
    return output.mean()
