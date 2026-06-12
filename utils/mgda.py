"""MGDA: minimum-norm gradient in the convex hull of task gradients.

Used by V52d / V53e to balance pose vs zero-drift (and optional inject-recovery)
without manual lambda tuning. See docs/V52_DESIGN.md §10.
"""
from __future__ import annotations

import torch


def _project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Project v onto the probability simplex {x >= 0, sum x = 1}."""
    if v.numel() == 1:
        return torch.ones_like(v)
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    if not cond.any():
        return torch.full_like(v, 1.0 / v.numel())
    rho = cond.nonzero(as_tuple=False)[-1, 0]
    theta = cssv[rho] / (rho.to(v.dtype) + 1.0)
    return torch.clamp(v - theta, min=0.0)


def _dot_grad_lists(a: list[torch.Tensor], b: list[torch.Tensor]) -> torch.Tensor:
    return sum((ga * gb).sum() for ga, gb in zip(a, b) if ga is not None and gb is not None)


def _add_grad_lists(
    acc: list[torch.Tensor | None],
    grads: list[torch.Tensor],
    weight: torch.Tensor,
) -> list[torch.Tensor]:
    out = []
    for a, g in zip(acc, grads):
        if g is None:
            out.append(a)
            continue
        contrib = g * weight
        out.append(contrib if a is None else a + contrib)
    return out


def mgda_frank_wolfe_alphas(
    grad_lists: list[list[torch.Tensor]],
    n_iter: int = 32,
) -> torch.Tensor:
    """Return simplex weights alpha (K,) for MGDA combination."""
    if not grad_lists:
        raise ValueError("mgda_frank_wolfe_alphas: empty gradient list")
    if len(grad_lists) == 1:
        return torch.ones(1, device=grad_lists[0][0].device)

    device = next(g for gl in grad_lists for g in gl if g is not None).device
    dtype = next(g for gl in grad_lists for g in gl if g is not None).dtype
    k = len(grad_lists)
    alpha = torch.full((k,), 1.0 / k, device=device, dtype=dtype)
    n_params = len(grad_lists[0])

    for _ in range(n_iter):
        g_mix = [None] * n_params
        for coeff, task_grads in zip(alpha, grad_lists):
            g_mix = _add_grad_lists(g_mix, task_grads, coeff)
        grad_alpha = torch.stack([
            2.0 * _dot_grad_lists(task_grads, g_mix) for task_grads in grad_lists
        ])
        alpha = _project_simplex(alpha - 0.05 * grad_alpha)
    return alpha


def mgda_weighted_loss(
    task_losses: dict,
    weight_params: list,
    pose_key: str = 'pose',
) -> tuple[torch.Tensor, dict]:
    """Build a single scalar loss = sum_k alpha_k * L_k via MGDA weights.

    MGDA weights are computed from gradients w.r.t. a *small* bottleneck
    (e.g. corr_head), then one full backward() traverses the graph once.
    This avoids K retain_graph passes through DINOv2/PointGPT.
    """
    items = [(k, v) for k, v in task_losses.items() if v is not None]
    if not items:
        raise ValueError("mgda_weighted_loss: no task losses")
    if len(items) == 1:
        return items[0][1], {'mgda_alpha': {items[0][0]: 1.0}}

    grad_lists = []
    for i, (_, loss) in enumerate(items):
        grads = torch.autograd.grad(
            loss, weight_params,
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        grad_lists.append([
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, weight_params)
        ])

    alpha = mgda_frank_wolfe_alphas(grad_lists)
    weighted = sum(alpha[i] * items[i][1] for i in range(len(items)))
    alpha_dict = {items[i][0]: float(alpha[i].item()) for i in range(len(items))}
    return weighted, {'mgda_alpha': alpha_dict}


def mgda_minimum_norm(flat_grads: list[torch.Tensor], n_iter: int = 32) -> torch.Tensor:
    """Combine K flat gradient vectors (legacy API)."""
    if not flat_grads:
        raise ValueError("mgda_minimum_norm: empty gradient list")
    if len(flat_grads) == 1:
        return flat_grads[0].clone()
    grad_lists = [[g] for g in flat_grads]
    alpha = mgda_frank_wolfe_alphas(grad_lists, n_iter=n_iter)
    return sum(alpha[i] * flat_grads[i] for i in range(len(flat_grads)))


def mgda_combine_param_grads(
    param_grads_per_task: list[list[torch.Tensor]],
) -> list[torch.Tensor]:
    if not param_grads_per_task:
        raise ValueError("empty param_grads_per_task")
    alpha = mgda_frank_wolfe_alphas(param_grads_per_task)
    n_params = len(param_grads_per_task[0])
    combined = [None] * n_params
    for coeff, task_grads in zip(alpha, param_grads_per_task):
        combined = _add_grad_lists(combined, task_grads, coeff)
    return combined


def mgda_multitask_backward(
    model,
    task_losses: dict,
    params: list,
    retain_graph: bool = False,
) -> list[torch.Tensor]:
    """Legacy: per-param MGDA grads (slow on full model — prefer mgda_weighted_loss)."""
    del model, retain_graph
    items = [(k, v) for k, v in task_losses.items() if v is not None]
    if not items:
        raise ValueError("mgda_multitask_backward: no task losses")
    grad_lists = []
    for i, (_, loss) in enumerate(items):
        grads = torch.autograd.grad(
            loss, params, retain_graph=(i < len(items) - 1), allow_unused=True)
        grad_lists.append([g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)])
    return mgda_combine_param_grads(grad_lists)
