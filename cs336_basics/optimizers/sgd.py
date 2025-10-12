from collections.abc import Callable, Iterable
from typing import Optional, Any
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults: dict[str, float] = {"lr": lr}  # type of variable 'defaults'
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss: Optional[torch.Tensor] = None if closure is None else closure()  # type of variable 'loss'

        for group in self.param_groups:  # type of variable 'group': dict[str, Any]
            lr: float = group["lr"]  # type of variable 'lr'

            for p in group["params"]:  # type of variable 'p': torch.nn.Parameter
                if p.grad is None:
                    continue

                p_state: dict[str, Any] = self.state[p]  # type of variable 'state'
                t: int = p_state.get("t", 0)  # type of variable 't'
                grad: torch.Tensor = p.grad.data  # type of variable 'grad'
                p_data: torch.Tensor = p.data  # type of variable 'p_data'

                update: torch.Tensor = (lr / math.sqrt(t + 1)) * grad  # type of variable 'update'
                p_data -= update  # in-place update
                p_state["t"] = t + 1

        return loss  # type of return value: Optional[torch.Tensor]


if __name__ == "__main__":
    weights: torch.nn.Parameter = torch.nn.Parameter(5 * torch.randn((10, 10)))  # type of variable 'weights'
    opt: SGD = SGD([weights], lr=1000.0)  # type of variable 'opt'

    for t in range(100):
        t: int  # type of variable 't'
        opt.zero_grad()  # type: ignore[no-untyped-call]

        loss: torch.Tensor = (weights ** 2).mean()  # type of variable 'loss'
        print(loss.cpu().item())

        loss.backward()
        _ = opt.step()  # type of variable '_': Optional[torch.Tensor]
