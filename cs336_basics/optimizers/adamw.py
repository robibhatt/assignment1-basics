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
        loss: Optional[torch.Tensor] = None if closure is None else closure()  # type of variable 'loss'

        for group in self.param_groups:
            lr: float = group["lr"]
            (beta_1, beta_2) = group["betas"]
            lambda_: float = group["lambda_"]
            eps: float = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                p_state: dict[str, Any] = self.state[p]


                # get the grad and the data
                grad: torch.Tensor = p.grad.data
                p_data: torch.Tensor = p.data

                # set t, m, v
                t: int = p_state.get("t", 1)
                m: float = p_state.get("m", torch.zeros_like(p_data))
                v: float = p_state.get("v", torch.zeros_like(p_data))

                # update m and v
                p_state["m"] = beta_1*m + (1-beta_1) * grad
                p_state["v"] = beta_2*v + (1 - beta_2)* (grad*grad)

                # set the effective learning rate
                adjusted_lr = lr * (1 - (beta_2) ** t) **0.5 / (1 - (beta_1) ** t)

                update: torch.Tensor = adjusted_lr * p_state["m"] / (torch.sqrt(p_state["v"]) + eps) 
                p_data -= update
                p_data *= (1 - lr * lambda_)

                # update t
                p_state["t"] = t + 1

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
