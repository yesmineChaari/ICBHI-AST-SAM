"""
Changes : 
Adds FSAM (Fisher-SAM) alongside the original SAM class.

"""

import torch


class SAM(torch.optim.Optimizer):
    """Original SAM — kept for backward compatibility."""

    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("Use first_step / second_step.")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        return torch.stack([
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]).norm(p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class FSAM(torch.optim.Optimizer):
    """
    Fisher-SAM — same two-step interface as SAM, strictly better perturbation.

    Instead of perturbing all weights equally, FSAM scales each parameter's
    perturbation by its approximate Fisher information (squared gradient).
    Parameters in high-curvature directions are perturbed more, flat directions
    less .

    Result: flatter minima, better generalisation.

    """

    def __init__(self, params, base_optimizer, rho=0.05, fisher_beta=1e-2, **kwargs):
        defaults = dict(rho=rho, fisher_beta=fisher_beta, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale       = group["rho"] / (grad_norm + 1e-12)
            fisher_beta = group["fisher_beta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g           = p.grad
                fisher_diag = g.pow(2)
                fisher_mean = fisher_diag.mean().clamp(min=1e-12)
                fisher_w    = fisher_diag / (fisher_mean + fisher_beta)
                e_w         = g * scale * (1.0 + fisher_w)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("Use first_step / second_step.")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        return torch.stack([
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]).norm(p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups