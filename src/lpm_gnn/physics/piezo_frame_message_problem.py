"""Intact double-window planar piezo frame: geometry/ports, not a matrix model.

Reuses the explicitly scalar direct-star PiezoMessageGraph.  No old assembled
model/FEM is imported.  The uniform square background cells encode actual area
and same-side connectivity only.  Material is homogeneous y-poled PZT-4 plane
strain unless explicitly parameterized.  This is a synthetic device geometry,
not a manufactured specimen.  No crack or duplicated face DOF is permitted.
"""

import numpy as np
import torch

from .piezo_message_geometry import MessageGeometry
from .piezo_message_graph import MessageConstraints


def make_frame_geometry(n, width=0.02):
    """20mm square; windows x=5..9/11..15mm, y=5..15mm, no cracks.

    n must be divisible by 20 so the same physical hole/pad boundaries are
    represented exactly at every permitted level; no staircase geometry drift.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 20 or n % 20:
        raise ValueError("Frame subdivisions must be a positive multiple of20")
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Positive finite square width required")
    base = MessageGeometry.rectangle(n, width=width, crack_cells=0)
    centers = base.coordinates[base.cells].mean(axis=1)
    holes = (
        (centers[:, 1] > 0.25 * width)
        & (centers[:, 1] < 0.75 * width)
        & (
            ((centers[:, 0] > 0.25 * width) & (centers[:, 0] < 0.45 * width))
            | ((centers[:, 0] > 0.55 * width) & (centers[:, 0] < 0.75 * width))
        )
    )
    cells = base.cells[~holes]
    used = np.unique(cells)
    remap = np.full(base.n, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    mesh = MessageGeometry(
        base.coordinates[used], remap[cells], width / n, np.empty((0, 2), dtype=np.int64)
    )
    expected_area = width**2 * (1 - 2 * 0.2 * 0.5)
    if (
        not np.isclose(mesh.volumes.sum(), expected_area, rtol=1e-12, atol=1e-18)
        or len(np.unique(mesh.coordinates, axis=0)) != mesh.n
        or mesh.face_pairs.size
    ):
        raise RuntimeError("Intact frame topology/area validation failed")
    return mesh


def frame_boundary_ids(mesh):
    xy = np.asarray(mesh.coordinates)
    width = float(xy[:, 0].max())
    tolerance = mesh.spacing * 1e-8
    bottom = np.flatnonzero(abs(xy[:, 1]) < tolerance)
    top = np.flatnonzero(abs(xy[:, 1] - width) < tolerance)
    if not len(bottom) or not len(top):
        raise ValueError("Missing full upper/lower external edge")
    return bottom, top


def frame_constraints(mesh, V=0.0, device="cpu"):
    """Bottom ux,uy,phi=0; upper phi=V; all other mechanics free.

    No top ux clamp, no rigid uy tie, no free-electrode/circuit unknown.  This
    matches the retained independent no-crack reference's original load pad.
    """
    bottom, top = frame_boundary_ids(mesh)
    prescribed = {3 * int(i) + ch: 0.0 for i in bottom for ch in (0, 1, 2)}
    prescribed.update({3 * int(i) + 2: float(V) for i in top})
    return MessageConstraints(mesh.n, prescribed=prescribed, face_pairs=(), device=device)


def frame_pad_weights(mesh):
    """Work-conjugate shape of a uniform 4mm upper central load pad.

    It is not an observation-space average.  Full particle fields remain
    available; weights specify a physical applied traction and its port work.
    """
    _, top = frame_boundary_ids(mesh)
    xy = mesh.coordinates
    width = float(xy[:, 0].max())
    weights = np.zeros(mesh.n)
    ordered = top[np.argsort(xy[top, 0])]
    for a, b in zip(ordered[:-1], ordered[1:]):
        x1, x2 = xy[a, 0], xy[b, 0]
        if x1 >= 0.4 * width - 1e-12 and x2 <= 0.6 * width + 1e-12:
            weights[[a, b]] += (x2 - x1) / 2
    if not np.isclose(weights.sum(), 0.2 * width, rtol=1e-12, atol=1e-15):
        raise ValueError("Pad boundaries must align with physical geometry")
    return weights / weights.sum()


def frame_load(mesh, force, device="cpu"):
    value = torch.zeros((mesh.n, 3), dtype=torch.float64, device=device)
    value[:, 1] = float(force) * torch.as_tensor(
        frame_pad_weights(mesh), dtype=torch.float64, device=device
    )
    return value


def relative_norm(actual, expected):
    return float(
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected).clamp_min(1e-30)
    )


def frame_diagnostics(problem, theta, state, full_rhs, force, voltage):
    """Fresh true residual, independent energy/charge/force bookkeeping."""
    graph, constraints = problem.graph, problem.constraints
    xy = graph.coordinates
    bottom = (xy[:, 1].abs() < graph.spacing * 1e-8).nonzero().flatten()
    top = ((xy[:, 1] - xy[:, 1].max()).abs() < graph.spacing * 1e-8).nonzero().flatten()
    action = graph.action(state, theta)
    residual = constraints.contract(action - full_rhs)
    rhs = constraints.rhs(graph, theta, full_rhs)
    scale = problem.scaling(theta)
    denominator = torch.linalg.vector_norm(scale * rhs).clamp_min(1e-30)
    q = -action[top, 2].sum()
    q0 = -action[bottom, 2].sum()
    work = (full_rhs * state).sum()
    h = graph.energy(state, theta)
    zero_piezo = theta * theta.new_tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    mechanical_state = state * state.new_tensor([1.0, 1.0, 0.0])
    electric_state = state * state.new_tensor([0.0, 0.0, 1.0])
    wm = graph.energy(mechanical_state, zero_piezo)
    wd = -graph.energy(electric_state, zero_piezo)
    stored = wm + wd
    source_stored = 0.5 * (work + float(voltage) * q)
    enthalpy_source = 0.5 * (work - float(voltage) * q)
    energy_scale = max(float(abs(stored)), float(abs(source_stored)), 1e-30)
    charge_scale = max(float(abs(q)), float(abs(q0)), 1e-30)
    reaction_y = action[bottom, 1].sum()
    force_scale = max(abs(float(force)), float(action[:, 1].abs().sum()), 1e-30)
    return dict(
        fresh_scaled_relative_true_residual=float(
            torch.linalg.vector_norm(scale * residual) / denominator
        ),
        electrode_charge_C=float(q),
        grounded_charge_C=float(q0),
        charge_balance_relative_error=float(abs(q + q0)) / charge_scale,
        bottom_reaction_N=float(reaction_y),
        force_balance_absolute_N=float(abs(reaction_y + float(force))),
        force_balance_relative_to_total_internal_scale=float(abs(reaction_y + float(force)))
        / force_scale,
        enthalpy_J=float(h),
        mechanical_energy_J=float(wm),
        dielectric_energy_J=float(wd),
        positive_stored_energy_J=float(stored),
        source_stored_energy_J=float(source_stored),
        stored_source_relative_error=float(abs(stored - source_stored)) / energy_scale,
        enthalpy_source_relative_error=float(abs(h - enthalpy_source)) / energy_scale,
        independent_energy_action_relative_error=float(abs(2 * h - (state * action).sum()))
        / energy_scale,
        all_physical_outputs_SI=True,
        residual_not_a_field_error_bound=True,
        no_crack=True,
        original_published_LPM_reproduction=False,
    )
