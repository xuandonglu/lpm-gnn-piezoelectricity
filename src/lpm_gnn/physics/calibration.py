"""Small-coordinate specimen calibration utilities."""

import torch


def residual(prediction, target):
    phase = prediction[:, 1] - target[:, 1]
    return torch.stack(
        (prediction[:, 0] - target[:, 0], torch.atan2(torch.sin(phase), torch.cos(phase))), dim=1
    )


def step_vector_cg(jac, res, diagonal, damping, free=None):
    """J products are explicit feature contractions, never J.T@J storage."""
    count = res.numel()
    scale = torch.rsqrt(diagonal)
    gradient = (jac * res[:, :, None]).sum(dim=(0, 1)) / count
    if free is None:
        free = torch.ones_like(gradient)
    rhs = -scale * gradient * free

    def action(y):
        direction = scale * y * free
        observation_direction = (jac * direction).sum(dim=2)
        return (
            free
            * scale
            * (
                (jac * observation_direction[:, :, None]).sum(dim=(0, 1)) / count
                + damping * diagonal * direction
            )
            + (1 - free) * y
        )

    y = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rr = (r * r).sum()
    initial = float(rr)
    for _ in range(15):
        ap = action(p)
        den = (p * ap).sum()
        if float(rr) < max(1e-28, initial * 1e-22) or float(den) <= 0:
            break
        alpha = rr / den
        y = y + alpha * p
        r = r - alpha * ap
        next_rr = (r * r).sum()
        p = r + (next_rr / rr) * p
        rr = next_rr
    return scale * y * free, gradient
