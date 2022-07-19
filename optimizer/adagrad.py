import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional



class Adagrad(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-8,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value
                )
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            foreach=foreach,
            maximize=maximize,
        )
        super(Adagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.tensor(0.0)
                init_value = (
                    complex(initial_accumulator_value, initial_accumulator_value)
                    if torch.is_complex(p)
                    else initial_accumulator_value
                )
                state["sum"] = torch.full_like(
                    p, init_value, memory_format=torch.preserve_format
                )

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)

        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            state_sums = []
            state_steps = []

            has_sparse_grad = False
            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        has_sparse_grad = True
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    state_sums.append(state["sum"])
                    state_steps.append(state["step"])
                
            for (param, grad, state_sum, step_t) in zip(params_with_grad, grads, state_sums, state_steps):

                step_t += 1
                step = step_t.item()
                grad = grad if not group['maximize'] else -grad

                if group['weight_decay'] != 0:
                    if grad.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                    grad = grad.add(param, alpha=group['weight_decay'])
            
                lr = group['lr']
                clr = lr / (1 + (step - 1) * group['lr_decay'])
            
                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
                    std = state_sum.sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(group['eps'])
                    param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)
                else:
                    is_complex = torch.is_complex(param)
                    if is_complex:
                        grad = torch.view_as_real(grad)
                        state_sum = torch.view_as_real(state_sum)
                        param = torch.view_as_real(param)
                    state_sum.addcmul_(grad, grad, value=1)
                    std = state_sum.sqrt().add_(group['eps'])
                    param.addcdiv_(grad, std, value=-clr)
                    if is_complex:
                        param = torch.view_as_complex(param)
                        state_sum = torch.view_as_complex(state_sum)
        return loss

def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    if grad_indices.numel() == 0 or values.numel() == 0:
        return torch.empty_like(grad)
    return torch.sparse_coo_tensor(grad_indices, values, size)