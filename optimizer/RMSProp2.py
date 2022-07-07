import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional
from optimizer.conjugate.scg_param import get_scg3_param_fn


class RMSprop2(Optimizer):
    def __init__(self, params, lr='D0', alpha='D1', eps=1e-8, weight_decay=0, momentum=0,
                 centered=False, foreach: Optional[bool] = None):
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered,
                        weight_decay=weight_decay, foreach=foreach)
        super(RMSprop2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)
            group.setdefault('foreach', None)
            group.setdefault('lr', 'D0')
            group.setdefault('alpha', 'D1')

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
            lr_param_fn = get_scg3_param_fn(group['lr'])
            alpha_param_fn = get_scg3_param_fn(group['alpha'])
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
                lr = lr_param_fn(state['step'])

                if group['weight_decay'] != 0:
                    grad = grad.add(param, alpha=group['weight_decay'])

                _alpha = alpha_param_fn(state['step'])
                square_avg.mul_(_alpha).addcmul_(grad, grad, value=1 - _alpha)

                if group['centered']:
                    grad_avg = grad_avgs[i]
                    grad_avg.mul_(_alpha).add_(grad, alpha=1 - _alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = momentum_buffer_list[i]
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    param.add_(buf, alpha=-lr)
                else:
                    param.addcdiv_(grad, avg, value=-lr)

        return loss
