import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional
from optimizer.conjugate.scg_param import get_scg3_param_fn



class SGD2(Optimizer):
    def __init__(self, params, lr='D0', momentum='No', dampening=0, weight_decay=0, nesterov=False, *, maximize=False):
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD2, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('lr', 'D0')
            group.setdefault('momentum', 'No')


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
            alpha_param_fn = get_scg3_param_fn(group['lr'])
            momentum_param_fn = get_scg3_param_fn(group['momentum'])
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
                if len(state) == 0:
                    state['step'] = 0

            for i, param in enumerate(group['params']):
                state['step'] += 1
                lr = alpha_param_fn(state['step'])
                momentum = momentum_param_fn(state['step'])
                d_p = d_p_list[i]
                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)

                if momentum != 0:
                    buf = momentum_buffer_list[i]

                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                alpha = lr if maximize else -lr
                param.add_(d_p, alpha=alpha)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
 