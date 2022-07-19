import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional


class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0,
                 centered=False, foreach: Optional[bool] = None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered,
                        weight_decay=weight_decay, foreach=foreach)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)
            group.setdefault('foreach', None)

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
            square_avgs = []
            grad_avgs = []
            momentum_buffer_list = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avgs.append(state['square_avg'])

                if group['momentum'] > 0:
                    momentum_buffer_list.append(state['momentum_buffer'])
                if group['centered']:
                    grad_avgs.append(state['grad_avg'])

            for i, param in enumerate(group['params']):
                state['step'] += 1
                grad = grads[i]
                square_avg = square_avgs[i]

                if group['weight_decay'] != 0:
                    grad = grad.add(param, alpha=group['weight_decay'])

                square_avg.mul_(group['alpha']).addcmul_(grad, grad, value=1 - group['alpha'])

                if group['centered']:
                    grad_avg = grad_avgs[i]
                    grad_avg.mul_(group['alpha']).add_(grad, alpha=1 - group['alpha'])
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = momentum_buffer_list[i]
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    param.add_(buf, alpha=-group['lr'])
                else:
                    param.addcdiv_(grad, avg, value=-group['lr'])

        return loss
