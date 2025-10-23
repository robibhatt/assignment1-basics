from collections.abc import Callable, Iterable
from typing import Optional, Any
import torch
import math


class AdamW(torch.optim.Optimizer):

    def __init__(self, 
        params: Iterable[torch.nn.Parameter], 
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0,
        eps: float = 1e-8) -> None:

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults: dict[str, float] = {"lr": lr, "betas": betas, "lambda_": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss: Optional[torch.Tensor] = None if closure is None else closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta_1, beta_2 = group["betas"]
            lambda_: float = group["lambda_"]
            eps: float = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # --- init state (keep moments in fp32) ---
                t: int = state.get("t", 0)
                m: torch.Tensor = state.get("m", torch.zeros_like(p, dtype=torch.float32, device=p.device))
                v: torch.Tensor = state.get("v", torch.zeros_like(p, dtype=torch.float32, device=p.device))

                # --- grads (detach, cast to fp32 for stable moment updates) ---
                g32 = p.grad.detach().float()

                # --- update moments (in-place, fp32) ---
                m.mul_(beta_1).add_(g32, alpha=1.0 - beta_1)
                v.mul_(beta_2).addcmul_(g32, g32, value=1.0 - beta_2)

                # --- bias correction ---
                t += 1
                bc1 = 1.0 - (beta_1 ** t)
                bc2 = 1.0 - (beta_2 ** t)
                mhat = m / bc1
                vhat = v / bc2

                # --- update (do math in fp32, cast to param dtype) ---
                denom = vhat.sqrt().add_(eps)
                upd32 = (lr * mhat) / denom
                upd = upd32.to(dtype=p.dtype)

                # --- decoupled weight decay + param update (no autograd) ---
                with torch.no_grad():
                    if lambda_ != 0.0:
                        p.mul_(1.0 - lr * lambda_)
                    p.add_(-upd)

                # --- persist state ---
                state["t"] = t
                state["m"] = m
                state["v"] = v

        return loss

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
