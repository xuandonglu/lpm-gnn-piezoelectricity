"""Pure-message positive-block V-cycle for the uncracked two-window frame.

Scope is deliberately narrow: a 20 mm square with rectangular windows
x=[5,9] and [11,15] mm, y=[5,15] mm, uniform square background cells and
n=20,40,80.  Bottom ux,uy,phi and top phi are fixed; top ux,uy are natural.
All other DOFs are free and there are no ties or duplicate-coordinate nodes.
This is a separate adapter, not a relaxation of the old rectangular MG guard.

One finest physical graph action with e=0 and input phi sign-reversed gives
the positive mechanical/dielectric block.  Piezoelectric coupling is omitted
ONLY in the preconditioner; the complete outer action is not modified.
The coarser callable actions are indexed restriction/action/prolongation
compositions.  Transfers are bilinear inside an ACTUAL material cell, never
a cell inside a window.  No global K/M/T/P/R, local Hessian, sparse backend,
direct linear solve or inverse is formed.

The diagonal is a positive proxy, not a claimed exact constrained Galerkin
diagonal or bound.  Residual-dependent energy damping and finite-budget
coarse vector CG make the V-cycle variable, requiring a flexible outer solve.
No speed superiority or new numerical algorithm is claimed.
"""

import math
import time

import numpy as np
import torch

from .piezo_graph_multilevel import IndexedTransfer
from .piezo_message_graph import MessageConstraints, PiezoMessageGraph


def _scalar(value):
    return float(value.detach().item())


def _dot(left, right, name):
    value = torch.vdot(left, right).real
    if not bool(torch.isfinite(value).item()):
        raise FloatingPointError("Nonfinite scalar in " + name)
    return value


def _norm(vector):
    if not vector.numel():
        return 0.0
    scale = torch.abs(vector).max()
    if _scalar(scale) == 0.0:
        return 0.0
    return _scalar(
        scale * torch.sqrt(torch.vdot(vector / scale, vector / scale).real.clamp_min(0.0))
    )


def _helpers():
    # Problem helpers do not import this module.  Lazy loading also lets static
    # audits import this file without creating any physical model or solve.
    from .piezo_frame_message_problem import frame_constraints, make_frame_geometry

    return make_frame_geometry, frame_constraints


def _geometry_and_constraints(problem):
    """Check graph geometry and homogeneous constraints; never solve a PDE."""
    graph, constraints = problem.graph, problem.constraints
    if not isinstance(graph, PiezoMessageGraph) or not isinstance(constraints, MessageConstraints):
        raise TypeError("Frame V-cycle requires PiezoMessageGraph and MessageConstraints")
    xy = graph.coordinates
    if (
        xy.dtype != torch.float64
        or xy.ndim != 2
        or xy.shape[1] != 2
        or not bool(torch.isfinite(xy).all().item())
    ):
        raise ValueError("Frame V-cycle requires finite real-double planar coordinates")
    points = xy.detach().cpu().numpy()
    if len(set(map(tuple, points.tolist()))) != len(points):
        raise ValueError("Duplicate-coordinate/crack nodes are outside the frame hierarchy")
    h = float(graph.spacing)
    n = int(round(0.02 / h)) if math.isfinite(h) and h > 0 else -1
    if n not in (20, 40, 80) or not math.isclose(h, 0.02 / n, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("Frame hierarchy supports only n=20,40,80 in a 20 mm square")
    make_geometry, make_constraints = _helpers()
    mesh = make_geometry(n, width=0.02)
    expected = np.asarray(mesh.coordinates, dtype=float)
    if points.shape != expected.shape or not np.allclose(
        points, expected, rtol=0.0, atol=1e-10 * h
    ):
        raise ValueError("Coordinates/order must match the exact uncracked two-window frame")
    if not np.allclose(
        graph.volumes.detach().cpu().numpy(),
        np.asarray(mesh.volumes),
        rtol=1e-12,
        atol=1e-12 * h * h,
    ):
        raise ValueError("Particle areas must match actual material quarter cells")
    metadata = graph.metadata
    if metadata.get("tip_sector_stars", 0) or metadata.get("ambiguous_real_center_nodes", []):
        raise ValueError("Tip-sector/cracked graphs are outside the frame hierarchy")
    expected_constraints = make_constraints(mesh, V=0.0, device=xy.device)
    for name in ("full_to_free", "free_representatives", "column_counts"):
        if not torch.equal(getattr(constraints, name), getattr(expected_constraints, name)):
            raise ValueError("Frame hierarchy requires bottom ux,uy,phi and top phi fixed, no ties")
    if not bool(torch.all(constraints.column_counts == 1).item()):
        raise ValueError("Free ties/electrode-port unknowns are outside this frame hierarchy")
    # A nonzero but uniform top voltage is permitted; its lift is irrelevant to
    # homogeneous correction.  All mechanical and bottom potential lifts are 0.
    lift = constraints.lift.detach().cpu().numpy()
    top = np.isclose(points[:, 1], 0.02, rtol=0.0, atol=1e-10 * h)
    zero_lift = lift.copy()
    zero_lift[top, 2] = 0.0
    if not np.all(zero_lift == 0.0) or not np.all(lift[top, 2] == lift[top, 2][0]):
        raise ValueError("Only a uniform top potential may have a nonzero fixed value")
    return n, mesh


def _node_grid(mesh, n):
    xy = np.asarray(mesh.coordinates, dtype=float)
    scaled = xy / (0.02 / n)
    integer = np.rint(scaled).astype(int)
    if not np.allclose(scaled, integer, rtol=0.0, atol=1e-9):
        raise ValueError("Frame geometry must remain background-node aligned")
    return [tuple(map(int, p)) for p in integer]


def _transfer(fine, coarse_mesh, coarse_constraints, nc, device):
    """Build only indexed incidences, choosing one containing material cell."""
    fine_grid = fine["node_grid"]
    coarse_grid = _node_grid(coarse_mesh, nc)
    coarse_node = {key: i for i, key in enumerate(coarse_grid)}
    fine_node = {key: i for i, key in enumerate(fine_grid)}
    if len(coarse_node) != len(coarse_grid) or len(fine_node) != len(fine_grid):
        raise ValueError("Distinct frame nodes are required")
    coarse_map = coarse_constraints.full_to_free.detach().cpu().tolist()
    fine_map = fine["constraints"].full_to_free.detach().cpu().tolist()
    coarse_rep = coarse_constraints.free_representatives.detach().cpu().tolist()
    # Injection supplies a positive coordinate scale, not a separate PDE model.
    injection = []
    for dof in coarse_rep:
        ix, iy = coarse_grid[dof // 3]
        node = fine_node.get((2 * ix, 2 * iy))
        index = -1 if node is None else fine_map[3 * node + dof % 3]
        if index < 0:
            raise ValueError("Every coarse free channel needs a same-coordinate fine free node")
        injection.append(index)
    injection = torch.tensor(injection, dtype=torch.long, device=device)
    coarse_scale = fine["scale"][injection]
    material_cells = set()
    for cell in np.asarray(coarse_mesh.cells):
        corners = [coarse_grid[int(node)] for node in cell]
        lower = (min(p[0] for p in corners), min(p[1] for p in corners))
        if set(corners) != {
            (lower[0], lower[1]),
            (lower[0] + 1, lower[1]),
            (lower[0] + 1, lower[1] + 1),
            (lower[0], lower[1] + 1),
        }:
            raise ValueError("Actual coarse material cells must be aligned unit squares")
        material_cells.add(lower)
    entries = {}
    containing_cell_indices = []
    fine_rep = fine["constraints"].free_representatives.detach().cpu().tolist()
    for row, dof in enumerate(fine_rep):
        ix, iy = fine_grid[dof // 3]
        channel = dof % 3
        cx, cy = ix // 2, iy // 2
        candidates_x = [cx - 1, cx] if ix % 2 == 0 else [cx]
        candidates_y = [cy - 1, cy] if iy % 2 == 0 else [cy]
        candidates = sorted(
            (x, y) for y in candidates_y for x in candidates_x if (x, y) in material_cells
        )
        if not candidates:
            raise ValueError("Fine free node lacks an actual containing coarse material cell")
        x, y = candidates[0]
        containing_cell_indices.append((x, y))
        tx, ty = ix / 2 - x, iy / 2 - y
        for dx, dy, weight in (
            (0, 0, (1 - tx) * (1 - ty)),
            (1, 0, tx * (1 - ty)),
            (1, 1, tx * ty),
            (0, 1, (1 - tx) * ty),
        ):
            if weight == 0.0:
                continue
            node = coarse_node.get((x + dx, y + dy))
            if node is None:
                raise ValueError("Containing material cell references a missing node")
            column = coarse_map[3 * node + channel]
            if column >= 0:
                key = (row, column)
                entries[key] = entries.get(key, 0.0) + weight
            # Fixed coarse values are homogeneous zero for error correction.
    if not entries:
        raise ValueError("Empty frame interpolation support")
    pairs = sorted(entries)
    source = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    destination = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    weights = torch.tensor([entries[p] for p in pairs], dtype=torch.float64, device=device)
    weights = weights * coarse_scale[destination] / fine["scale"][source]
    transfer = IndexedTransfer(
        source, destination, weights, n_fine=len(fine_rep), n_coarse=len(coarse_rep)
    )
    # Expose geometry-only support for tests; these are cell IDs, not coefficients.
    support = dict(
        interpolation_entries=len(entries),
        material_cells=len(material_cells),
        containing_material_cells=containing_cell_indices,
        cross_window_cells_used=0,
    )
    return transfer, coarse_scale, coarse_grid, support


class FrameBlockVCyclePreconditioner:
    """Callable finite-budget frame V-cycle for a flexible outer iteration."""

    @torch.no_grad()
    def __init__(
        self,
        problem,
        theta,
        scale,
        action,
        *,
        pre_smooth=2,
        post_smooth=2,
        omega=0.7,
        coarse_iterations=32,
        coarse_rtol=0.05,
    ):
        tick = time.perf_counter()
        for name, value in (
            ("pre_smooth", pre_smooth),
            ("post_smooth", post_smooth),
            ("coarse_iterations", coarse_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(name + " must be an integer")
        if min(pre_smooth, post_smooth) < 0 or coarse_iterations < 1:
            raise ValueError("Nonnegative smooth counts and positive coarse budget required")
        if not math.isfinite(omega) or not 0 < omega <= 1:
            raise ValueError("Energy-damped omega must lie in (0,1]")
        if not math.isfinite(coarse_rtol) or not 0 <= coarse_rtol < 1:
            raise ValueError("coarse_rtol must lie in [0,1)")
        if not callable(action):
            raise TypeError("The unchanged scaled outer action must be callable")
        n, mesh = _geometry_and_constraints(problem)
        graph, constraints = problem.graph, problem.constraints
        if (
            not isinstance(scale, torch.Tensor)
            or scale.dtype != torch.float64
            or scale.shape != (constraints.reduced_dofs,)
            or scale.device != graph.coordinates.device
            or not bool(torch.all(torch.isfinite(scale) & (scale > 0)).item())
        ):
            raise ValueError("Scale must be a finite positive float64 free-DOF vector")
        graph.validate_parameters(theta)
        self.pre_smooth, self.post_smooth = pre_smooth, post_smooth
        self.omega, self.coarse_iterations, self.coarse_rtol = omega, coarse_iterations, coarse_rtol
        self.device, self.dtype, self.size = scale.device, scale.dtype, scale.numel()
        self.stats = dict(
            calls=0,
            positive_finest_action_calls=0,
            finest_block_graph_action_calls=0,
            transfer_prolong_calls=0,
            transfer_restrict_calls=0,
            smoothing_steps=0,
            coarse_cg_steps=0,
            coarse_cg_budget_exhaustions=0,
            energy_limited_smoothing_steps=0,
            elapsed_apply_seconds=0.0,
        )
        no_piezo = theta.detach().clone()
        no_piezo[4:7] = 0.0
        sign = scale.new_tensor([1.0, 1.0, -1.0])

        def positive_action(value):
            self.stats["positive_finest_action_calls"] += 1
            self.stats["finest_block_graph_action_calls"] += 1
            state = constraints.prolong(scale * value) * sign
            return scale * constraints.contract(graph.action(state, no_piezo))

        mechanical_diag = graph.mechanical_diagonal(theta)
        dielectric_diag = graph.dielectric_diagonal(theta)
        local_diag = torch.stack(
            (mechanical_diag[:, 0], mechanical_diag[:, 1], dielectric_diag), dim=1
        )
        diagonal = (scale.square() * constraints.contract(local_diag)).detach()
        if not bool(torch.all(torch.isfinite(diagonal) & (diagonal > 0)).item()):
            raise ValueError("Finite positive finest block diagonal required")
        self.levels = [
            dict(
                n=n,
                constraints=constraints,
                node_grid=_node_grid(mesh, n),
                diagonal=diagonal,
                scale=scale.detach().clone(),
                action=positive_action,
                transfer=None,
            )
        ]
        make_geometry, make_constraints = _helpers()
        support_audits = []
        while n > 20:
            fine = self.levels[-1]
            n //= 2
            coarse_mesh = make_geometry(n, width=0.02)
            coarse_constraints = make_constraints(coarse_mesh, V=0.0, device=self.device)
            transfer, coarse_scale, coarse_grid, support = _transfer(
                fine, coarse_mesh, coarse_constraints, n, self.device
            )
            fine["transfer"] = transfer
            support_audits.append(support)
            diagonal = torch.zeros(
                coarse_constraints.reduced_dofs, dtype=self.dtype, device=self.device
            ).index_add(0, transfer.dst, transfer.weights.square() * fine["diagonal"][transfer.src])
            if not bool(torch.all(torch.isfinite(diagonal) & (diagonal > 0)).item()):
                raise ValueError("Nonpositive coarse diagonal proxy; inspect frame support")

            def coarse_action(value, parent=fine["action"], mapping=transfer):
                self.stats["transfer_prolong_calls"] += 1
                lifted = mapping.prolong(value)
                response = parent(lifted)
                self.stats["transfer_restrict_calls"] += 1
                return mapping.restrict(response)

            self.levels.append(
                dict(
                    n=n,
                    constraints=coarse_constraints,
                    node_grid=coarse_grid,
                    diagonal=diagonal,
                    scale=coarse_scale,
                    action=coarse_action,
                    transfer=None,
                )
            )
        self.metadata = dict(
            method="pure-message positive-block geometric frame V-cycle",
            scope="uncracked 20 mm two-window square frame; n20/40/80; bottom fixed, top phi fixed, no ties",
            hierarchy=[
                dict(
                    n=level["n"],
                    particles=len(level["node_grid"]),
                    free_dofs=level["constraints"].reduced_dofs,
                )
                for level in self.levels
            ],
            pre_smooth=pre_smooth,
            post_smooth=post_smooth,
            omega=omega,
            coarse_iterations=coarse_iterations,
            coarse_rtol=coarse_rtol,
            smoother="residual-dependent energy-damped diagonal direction",
            coarse_diagonal="positive sum(weight^2*parent_diag) proxy; not exact Galerkin or proven bound",
            transfer="indexed bilinear within actual material cells; scale_coarse/scale_fine",
            restriction="transpose sum, not averaging",
            transfer_support=support_audits,
            positive_block="one finest graph action, e=0, input phi sign reversed",
            outer_iteration_requires_flexible_preconditioning=True,
            piezo_cross_blocks_in_preconditioner=False,
            piezo_cross_blocks_in_outer_operator_unchanged=True,
            coarse_action_uses_finest_graph=True,
            global_K_M_T_P_R_constructed=False,
            local_Hessian_constructed=False,
            dense_or_sparse_solver_calls=0,
            trained_network=False,
            speed_advantage_claimed=False,
            setup_seconds=time.perf_counter() - tick,
        )

    def _smooth(self, level, rhs, x, residual, count):
        for _ in range(count):
            z = residual / level["diagonal"]
            numerator = _dot(residual, z, "smoother numerator")
            if _scalar(numerator) == 0.0:
                break
            if _scalar(numerator) < 0.0:
                raise RuntimeError("Nonpositive diagonal direction in frame V-cycle")
            az = level["action"](z)
            curvature = _dot(z, az, "smoother curvature")
            if _scalar(curvature) <= 0.0:
                raise RuntimeError("Nonpositive block curvature; frame SPD assumptions failed")
            optimum = numerator / curvature
            alpha = torch.clamp(optimum, max=self.omega)
            self.stats["energy_limited_smoothing_steps"] += int(_scalar(optimum) < self.omega)
            x, residual = x + alpha * z, residual - alpha * az
            self.stats["smoothing_steps"] += 1
        return x, residual

    def _coarse_cg(self, level, rhs):
        x, residual = torch.zeros_like(rhs), rhs.clone()
        initial = _norm(residual)
        if initial == 0.0:
            return x
        target = self.coarse_rtol * initial
        z = residual / level["diagonal"]
        direction = z.clone()
        rz = _dot(residual, z, "coarse CG initial")
        for _ in range(self.coarse_iterations):
            ad = level["action"](direction)
            curvature = _dot(direction, ad, "coarse CG curvature")
            if _scalar(curvature) <= 0.0 or _scalar(rz) <= 0.0:
                raise RuntimeError("Nonpositive coarse CG curvature; frame SPD assumptions failed")
            alpha = rz / curvature
            x, residual = x + alpha * direction, residual - alpha * ad
            self.stats["coarse_cg_steps"] += 1
            if _norm(residual) <= target:
                return x
            z = residual / level["diagonal"]
            next_rz = _dot(residual, z, "coarse CG updated")
            if _scalar(next_rz) == 0.0:
                return x
            direction, rz = z + (next_rz / rz) * direction, next_rz
        self.stats["coarse_cg_budget_exhaustions"] += 1
        return x  # Coarse budget exhaustion is not outer convergence.

    def _cycle(self, index, rhs):
        level = self.levels[index]
        if level["transfer"] is None:
            return self._coarse_cg(level, rhs)
        x, residual = torch.zeros_like(rhs), rhs.clone()
        x, residual = self._smooth(level, rhs, x, residual, self.pre_smooth)
        self.stats["transfer_restrict_calls"] += 1
        coarse_rhs = level["transfer"].restrict(residual)
        coarse_correction = self._cycle(index + 1, coarse_rhs)
        self.stats["transfer_prolong_calls"] += 1
        x = x + level["transfer"].prolong(coarse_correction)
        residual = rhs - level["action"](x)
        x, residual = self._smooth(level, rhs, x, residual, self.post_smooth)
        return x

    @torch.no_grad()
    def __call__(self, value):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (self.size,)
            or value.dtype != self.dtype
            or value.device != self.device
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ValueError("Frame V-cycle requires a finite real-double free vector")
        tick = time.perf_counter()
        self.stats["calls"] += 1
        result = self._cycle(0, value)
        self.stats["elapsed_apply_seconds"] += time.perf_counter() - tick
        if not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("Nonfinite frame V-cycle correction")
        return result

    def diagnostics(self):
        return dict(metadata=dict(self.metadata), stats=dict(self.stats))


def make_frame_vcycle_factory(
    *, pre_smooth=2, post_smooth=2, omega=0.7, coarse_iterations=32, coarse_rtol=0.05
):
    """Return factory(problem,theta,scale,action) for MessageEquilibrium.

    Setup only reads geometry, constraints and analytic diagonal messages.
    Actual physical actions begin when the returned preconditioner is applied.
    """

    def factory(problem, theta, scale, action):
        instance = FrameBlockVCyclePreconditioner(
            problem,
            theta,
            scale,
            action,
            pre_smooth=pre_smooth,
            post_smooth=post_smooth,
            omega=omega,
            coarse_iterations=coarse_iterations,
            coarse_rtol=coarse_rtol,
        )
        problem.last_preconditioner = instance
        return instance

    return factory


frame_vcycle_factory = make_frame_vcycle_factory
