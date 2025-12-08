from collections.abc import Callable, Iterable
from typing import Optional, Any
import torch
import math
torch.set_default_device("cuda")


class Luon(torch.optim.Optimizer):

    def __init__(self, 
        params: Iterable[torch.nn.Parameter], 
        lr: float = 1e-3,
        mu: float = 0.95,
        weight_decay: float = 0,
        eps: float = 1e-8) -> None:

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults: dict[str, float] = {"lr": lr, "mu": mu, "lambda_": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss: Optional[torch.Tensor] = None if closure is None else closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            mu = group["mu"]
            lambda_: float = group["lambda_"]
            eps: float = group["eps"]

            comp_orthog = suppress_orthog
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # --- init state (keep moments in fp32) ---
                t: int = state.get("t", 0)
                B: torch.Tensor = state.get("B", torch.zeros_like(p, dtype=torch.float32, device=p.device))
                O: torch.Tensor = state.get("O", torch.zeros_like(p, dtype=torch.bfloat16, device=p.device))

                # --- grads (detach, cast to fp32 for stable moment updates) ---
                g32 = p.grad.detach().float()

                # --- update moments (in-place, fp32) ---
                B.mul_(mu).add_(g32, alpha= 1.0 - mu)

                # --- Orthogonalize the momentum
                O = comp_orthog(B.bfloat16(), eps=eps)

                # --- compute pre-factor sqrt(fan-out/fan-in)
                fan_out = p.grad.size(-2)
                fan_in = p.grad.size(-1)
                pre_factor = max(1, fan_out/fan_in)**0.5


                # --- update (do math in fp32, cast to param dtype) ---
                upd32 = (lr * pre_factor * O)
                upd = upd32.to(dtype=p.dtype)

                # --- decoupled weight decay + param update (no autograd) ---
                with torch.no_grad():
                    if lambda_ != 0.0:
                        p.mul_(1.0 - lr * lambda_)
                    p.add_(-upd)

                # --- persist state ---
                state["t"] = t
                state["B"] = B
                state["O"] = O

        return loss

def orthog(G, eps = 1e-7):
    # Using constants found by You Jiacheng (via J. Bernstein's Modula)
    # TODO: should probably compile this, if possible
    # torch.set_default_device("cuda")
    with torch.device(G.get_device()):
        abc_list = [
            (3955./1024, -8306./1024, 5008./1024),
            (3735./1024, -6681./1024, 3463./1024),
            (3799./1024, -6499./1024, 3211./1024),
            (4019./1024, -6385./1024, 2906./1024),
            (2677./1024, -3029./1024, 1162./1024),
            (2172./1024, -1833./1024,  682./1024),       
        ]

        abc_list_basic = [(3.4445, -4.7750, 2.0315)]*5


        
        # flip things around so that the matrix has the right rank
        flip = G.shape[1] > G.shape[0]
        if flip:
            G = G.T
        # normalize the matrix so stuff doesn't blowup
        G = G/(torch.linalg.matrix_norm(G, keepdim=True) + eps)
        I = torch.eye(G.shape[1]).bfloat16()
        for a,b,c in abc_list:
            # faster quintic computation
            A = G.T @ G
            G = G @ ( a * I + b * A + c * A @ A)
        if flip:
            # flip stuff back
            G = G.T
        # print("GRADNORM", torch.linalg.matrix_norm(G))
        # breakpoint()
        return G

def suppress_orthog(G, eps = 1e-8,supp_eps=5e-2,times=10):
    # Using constants found by You Jiacheng (via J. Bernstein's Modula)
    # TODO: should probably compile this, if possible
    # torch.set_default_device("cuda")
    with torch.device(G.get_device()):

        abc_list = [( 1-2*supp_eps, 1 + supp_eps , -1)]*times
        # flip things around so that the matrix has the right rank
        flip = G.shape[1] > G.shape[0]
        if flip:
            G = G.T
        # normalize the matrix so stuff doesn't blowup
        G = G/(torch.linalg.matrix_norm(G, keepdim=True) + eps)
        const = 1./(1-supp_eps + eps)
        I = torch.eye(G.shape[1]).bfloat16()
        for a,b,c in abc_list:
            A = G.T @ G
            # faster quintic computation
            G = const * G @ ( a * I + b * A + c * A.T @ A )
        if flip:
            # flip stuff back
            G = G.T
        # print("GRADNORM", torch.linalg.matrix_norm(G))
        # breakpoint()
        return G



if __name__ == "__main__":
    weights: torch.nn.Parameter = torch.nn.Parameter(5 * torch.randn((10, 10))) 
    opt: AdamW = AdamW([weights], lr=1000.0) 

    for t in range(100):
        t: int  
        opt.zero_grad()  

        loss: torch.Tensor = (weights ** 2).mean()  
        print(loss.cpu().item())

        loss.backward()
        _ = opt.step()  
