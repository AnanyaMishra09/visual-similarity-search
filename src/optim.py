"""Optimizers implemented from raw torch.Tensor operations. No torch.optim
anywhere in this file. Each optimizer owns its own per-parameter state
(momentum / moment buffers) and mutates `param.data` in-place under
`torch.no_grad()`, which is exactly what torch.optim does internally, just
without the convenience of inheriting from `torch.optim.Optimizer`.

All three implement the same tiny interface: `.step()` and `.zero_grad()`.
"""
from typing import Iterable, List

import torch


class _BaseOptimizer:
    def __init__(self, params: Iterable[torch.nn.Parameter]):
        self.params: List[torch.nn.Parameter] = [p for p in params if p.requires_grad]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        raise NotImplementedError


class SGDMomentum(_BaseOptimizer):
    """Classic heavy-ball SGD with momentum.

    v_t = mu * v_{t-1} + grad_t  (+ weight_decay * param, if set)
    param_t = param_{t-1} - lr * v_t
    """

    def __init__(self, params, lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.mu = momentum
        self.weight_decay = weight_decay
        self.velocity = {id(p): torch.zeros_like(p.data) for p in self.params}

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            v = self.velocity[id(p)]
            v.mul_(self.mu).add_(grad)
            p.data.add_(v, alpha=-self.lr)


class Adam(_BaseOptimizer):
    """Adam (Kingma & Ba, 2014), bias-corrected first/second moments."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {id(p): torch.zeros_like(p.data) for p in self.params}
        self.v = {id(p): torch.zeros_like(p.data) for p in self.params}
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            m, v = self.m[id(p)], self.v[id(p)]
            m.mul_(self.b1).add_(grad, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(grad, grad, value=1 - self.b2)
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)


class NAdam(_BaseOptimizer):
    """Nesterov-accelerated Adam (Dozat, 2016). Same moment estimates as
    Adam, but the parameter update looks one momentum step ahead by mixing
    in the *current* gradient's contribution to the momentum schedule
    before it's actually accumulated, which is the "Nesterov" part.
    """

    def __init__(self, params, lr: float = 2e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {id(p): torch.zeros_like(p.data) for p in self.params}
        self.v = {id(p): torch.zeros_like(p.data) for p in self.params}
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            m, v = self.m[id(p)], self.v[id(p)]
            m.mul_(self.b1).add_(grad, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(grad, grad, value=1 - self.b2)

            b1_t = self.b1 ** self.t
            b1_t1 = self.b1 ** (self.t + 1)
            m_hat = (self.b1 * m) / (1 - b1_t1) + ((1 - self.b1) * grad) / (1 - b1_t)
            v_hat = v / (1 - self.b2 ** self.t)
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)


def build_optimizer(name: str, params, lr: float, **kwargs):
    name = name.lower()
    if name == "sgd":
        return SGDMomentum(params, lr=lr, momentum=kwargs.get("momentum", 0.9),
                            weight_decay=kwargs.get("weight_decay", 0.0))
    if name == "adam":
        return Adam(params, lr=lr, betas=kwargs.get("betas", (0.9, 0.999)),
                     eps=kwargs.get("eps", 1e-8), weight_decay=kwargs.get("weight_decay", 0.0))
    if name == "nadam":
        return NAdam(params, lr=lr, betas=kwargs.get("betas", (0.9, 0.999)),
                      eps=kwargs.get("eps", 1e-8), weight_decay=kwargs.get("weight_decay", 0.0))
    raise ValueError(f"Unknown optimizer: {name}")
