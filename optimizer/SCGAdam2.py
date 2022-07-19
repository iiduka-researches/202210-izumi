import math
from typing import List

import torch

from torch.optim.optimizer import Optimizer
from optimizer.scg_param import get_scg3_param_fn,get_scg4_param_fn

class SCGAdam2(Optimizer):
    def __init__(self, params, period: int, lr='D0', beta='D1',betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False,
                 cg1_type='D1', cg2_type='D1', lam=2.0) -> None:
        if not 0 <= period:
            raise ValueError("Invalid period: {}".format(period))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(period=period, lr=lr, beta=beta, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
                        cg1_type=cg1_type, cg2_type=cg2_type, lam=lam)
        super(SCGAdam2, self).__init__(params, defaults)
        # self.scg_expect_errors: List[float] = []

    def __setstate__(self, state):
        super(SCGAdam2, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('lr', 'D0')
            group.setdefault('beta', 'D1')
            group.setdefault('cg1_type', 'D1')
            group.setdefault('cg2_type', 'D1')

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            errors = []
            scg1_param_fn = get_scg3_param_fn(group['cg1_type'])
            scg2_param_fn = get_scg4_param_fn(group['cg2_type'])
            alpha_param_fn = get_scg3_param_fn(group['lr'])
            beta_param_fn = get_scg3_param_fn(group['beta'])

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                    state['stochastic_scg'] = None
                    state['past_grad'] = None

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                else:
                    max_exp_avg_sq = None
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    # Decay the first and second moment running average coefficient
                    grad = grad.add(p, alpha=group['weight_decay'])

                if state['stochastic_scg'] is None:
                    state['past_grad'] = grad.clone()
                    state['stochastic_scg'] = (-grad).clone()
                else:
                    scg = state['stochastic_scg']
                    scg1_param = scg1_param_fn(state['step'])
                    scg2_param = scg2_param_fn(state['step'])
                    state['stochastic_scg'] = -(1 + scg1_param) * grad + scg2_param * scg
                    state['past_grad'] = grad.clone()

                beta = beta_param_fn(state['step'])
                exp_avg.mul_(beta).add_(state['stochastic_scg'], alpha=1 - beta)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                lr = alpha_param_fn(state['step'])
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    # denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                    step_size = lr
                else:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    step_size = lr / bias_correction1

                p.addcdiv_(exp_avg, denom, value=step_size)

            # if errors:
            #     self.scg_expect_errors.append(sum(errors) / len(errors))

        return loss
