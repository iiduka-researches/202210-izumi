import math
import torch
from torch.optim.optimizer import Optimizer
from optimizer.conjugate.scg_param import get_scg3_param_fn


class AdamW2(Optimizer):
    def __init__(self, params, lr='D0', beta='D1',betas=(0.9, 0.999), eps=1e-8, weight=1e-2, weight_decay=0, amsgrad=False) -> None:
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight:
            raise ValueError("Invalid weight value: {}".format(weight))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, beta=beta, betas=betas, eps=eps, weight=weight,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW2, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('lr', 'D0')
            group.setdefault('beta', 'D1')

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
                    # state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    # state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                        # state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                else:
                    max_exp_avg_sq = None
                beta1, beta2 = group['betas']

                state['step'] += 1
                lr = alpha_param_fn(state['step'])
                # Perform stepweight decay
                p.mul_(1 - lr * group['weight'])

                beta = beta_param_fn(state['step'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta).add_(grad, alpha=1 - beta)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
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

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

