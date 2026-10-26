"""Restricted, matrix-free geometric block V-cycle for MessageEquilibrium.

This is a conventional preconditioner, not a trained network/new solver.  It
accepts ONLY a complete regular rectangular lattice with no duplicate points,
no free ties, all three channels fixed on its upper/lower edges, and no other
Dirichlet DOFs.  It deliberately refuses cracked/perforated/tied geometries.

The scaled saddle action A is used to form its positive diagonal BLOCKS by
two actions: M_u A M_u - M_phi A M_phi.  Mechanical cross-components remain;
piezoelectric cross-blocks are omitted ONLY in this preconditioner, never in
the outer equilibrium action.  Coarse actions are P.T A_block P, implemented
as callable compositions.  There is no global K/P/T tensor or factorization.

Transfers use bilinear physical-coordinate interpolation, with the necessary
scale_coarse/scale_fine conversion for the outer solver's scaled unknowns.
The coarse diagonal is sum(P_ij**2 * diag_i), a POSITIVE PROXY, NOT the exact
Galerkin diagonal or a proven bound.  A residual-dependent energy-damped
diagonal smoother prevents an underestimated proxy from selecting an excessive
step on an SPD block: alpha=min(omega, (r.T z)/(z.T A z)), z=D_proxy^-1 r.
Coarsest correction is vector-only CG with a finite iteration budget.

This variable/nonlinear preconditioner requires a flexible outer iteration.
Coarse matvecs STILL call the finest physical graph; a speed benefit is not
assumed.  Internal finest-action calls are counted separately from FGMRES's
top-level calls.  No performance or broad-grid robustness claim is made.
"""

import math
import time

import torch

from .piezo_graph_multilevel import IndexedTransfer


def _scalar(value):
    return float(value.detach().item())


def _positive_dot(left, right, name):
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


def _regular_rectangle(problem):
    """Read geometry/constraints only. No mesh/model/solver is instantiated."""
    graph, constraints = problem.graph, problem.constraints
    coordinates = graph.coordinates
    if (
        coordinates.dtype != torch.float64
        or coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or not bool(torch.isfinite(coordinates).all())
    ):
        raise ValueError("V-cycle requires finite float64 planar coordinates")
    xy = coordinates.detach().cpu().tolist()
    if len(set(map(tuple, xy))) != len(xy):
        raise ValueError("Crack/duplicate-coordinate nodes must not be merged by this V-cycle")
    xaxis, yaxis = sorted({p[0] for p in xy}), sorted({p[1] for p in xy})
    nx, ny = len(xaxis) - 1, len(yaxis) - 1
    if nx < 1 or ny < 2 or len(xy) != (nx + 1) * (ny + 1):
        raise ValueError("V-cycle supports a complete regular rectangle only")
    hx, hy = (xaxis[-1] - xaxis[0]) / nx, (yaxis[-1] - yaxis[0]) / ny
    coordinate_tolerance = 1e-9 * max(hx, hy)
    if (
        min(hx, hy) <= 0
        or abs(hx - hy) > coordinate_tolerance
        or any(abs(xaxis[i] - xaxis[0] - i * hx) > coordinate_tolerance for i in range(nx + 1))
        or any(abs(yaxis[i] - yaxis[0] - i * hy) > coordinate_tolerance for i in range(ny + 1))
    ):
        raise ValueError("Uniform square-cell rectangular background is required")
    xlookup, ylookup = {v: i for i, v in enumerate(xaxis)}, {v: i for i, v in enumerate(yaxis)}
    node_grid = [(xlookup[x], ylookup[y]) for x, y in xy]
    volumes = graph.volumes.detach().cpu().tolist()
    for (ix, iy), volume in zip(node_grid, volumes):
        target = hx * hy * (0.5 if ix in (0, nx) else 1.0) * (0.5 if iy in (0, ny) else 1.0)
        if not math.isclose(volume, target, rel_tol=1e-10, abs_tol=1e-14 * hx * hy):
            raise ValueError("Particle areas do not describe an intact complete rectangle")
    metadata = getattr(graph, "metadata", {})
    if metadata.get("tip_sector_stars", 0) or metadata.get("ambiguous_real_center_nodes", []):
        raise ValueError("Cracked/tip-sector graph is outside this V-cycle scope")
    if not bool(torch.all(constraints.column_counts == 1).item()):
        raise ValueError("Free ties/electrode-port unknowns are not supported")
    full_to_free = constraints.full_to_free.detach().cpu().tolist()
    for node, (_, iy) in enumerate(node_grid):
        for channel in range(3):
            free = full_to_free[3 * node + channel] >= 0
            if free != (iy not in (0, ny)):
                raise ValueError("Exactly the upper/lower edges must fix ALL three channels")
    representatives = constraints.free_representatives.detach().cpu().tolist()
    fine_dofs = []
    for dof in representatives:
        ix, iy = node_grid[dof // 3]
        fine_dofs.append((ix, iy, dof % 3))
    if len(set(fine_dofs)) != len(fine_dofs):
        raise ValueError("Distinct free geometric channels required")
    return nx, ny, fine_dofs


class BlockVCyclePreconditioner:
    """Callable positive-block V-cycle; constructed by factory below.

    Attributes ``metadata`` and ``stats`` expose scope and cumulative work.
    ``diagnostics()`` returns serializable copies without running an action.
    """

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
        coarse_iterations=24,
        coarse_rtol=0.05,
        max_levels=10,
        min_coarse_cells=4,
    ):
        tick = time.perf_counter()
        for name, value in (
            ("pre_smooth", pre_smooth),
            ("post_smooth", post_smooth),
            ("coarse_iterations", coarse_iterations),
            ("max_levels", max_levels),
            ("min_coarse_cells", min_coarse_cells),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(name + " must be an integer")
        if (
            min(pre_smooth, post_smooth) < 0
            or coarse_iterations < 1
            or max_levels < 1
            or min_coarse_cells < 1
        ):
            raise ValueError("Nonnegative smooth counts and positive hierarchy/budget required")
        if not math.isfinite(omega) or not 0 < omega <= 1:
            raise ValueError("Energy-damped omega must lie in (0,1]")
        if not math.isfinite(coarse_rtol) or not 0 <= coarse_rtol < 1:
            raise ValueError("coarse_rtol must lie in [0,1)")
        if not callable(action):
            raise TypeError("A scaled saddle action is required")
        nx, ny, dofs = _regular_rectangle(problem)
        constraints, graph = problem.constraints, problem.graph
        if (
            not isinstance(scale, torch.Tensor)
            or scale.dtype != torch.float64
            or scale.shape != (len(dofs),)
            or scale.device != graph.coordinates.device
            or not bool(torch.all(torch.isfinite(scale) & (scale > 0)).item())
        ):
            raise ValueError("scale must be a finite positive float64 free-DOF vector")
        self.pre_smooth, self.post_smooth = pre_smooth, post_smooth
        self.omega, self.coarse_iterations, self.coarse_rtol = omega, coarse_iterations, coarse_rtol
        self.device, self.dtype = scale.device, scale.dtype
        self.size = len(dofs)
        self.stats = dict(
            calls=0,
            positive_finest_action_calls=0,
            finest_saddle_action_calls=0,
            transfer_prolong_calls=0,
            transfer_restrict_calls=0,
            smoothing_steps=0,
            coarse_cg_steps=0,
            coarse_cg_budget_exhaustions=0,
            energy_limited_smoothing_steps=0,
            elapsed_apply_seconds=0.0,
        )
        mechanical = constraints.free_mechanical_mask.detach().clone()
        electrical = constraints.free_electrical_mask.detach().clone()

        def positive_action(value):
            self.stats["positive_finest_action_calls"] += 1
            # Both disjoint channel actions count, including any exact-zero block.
            self.stats["finest_saddle_action_calls"] += 2
            first = action(value * mechanical)
            second = action(value * electrical)
            return first * mechanical - second * electrical

        mechanical_diag = graph.mechanical_diagonal(theta)
        dielectric_diag = graph.dielectric_diagonal(theta)
        unscaled_diag = constraints.contract(
            torch.stack((mechanical_diag[:, 0], mechanical_diag[:, 1], dielectric_diag), dim=1)
        )
        diagonal = (scale.square() * unscaled_diag).detach()
        if not bool(torch.all(torch.isfinite(diagonal) & (diagonal > 0)).item()):
            raise ValueError("Positive finite untied block diagonal required")
        self.levels = [
            dict(
                action=positive_action,
                diagonal=diagonal,
                dofs=dofs,
                nx=nx,
                ny=ny,
                scale=scale.detach().clone(),
                transfer=None,
            )
        ]
        while (
            len(self.levels) < max_levels
            and ny >= 4
            and min(nx, ny) > min_coarse_cells
            and nx % 2 == 0
            and ny % 2 == 0
        ):
            fine = self.levels[-1]
            nx, ny = nx // 2, ny // 2
            coarse_dofs = [
                (ix, iy, ch) for iy in range(1, ny) for ix in range(nx + 1) for ch in range(3)
            ]
            fine_lookup = {key: i for i, key in enumerate(fine["dofs"])}
            coarse_lookup = {key: i for i, key in enumerate(coarse_dofs)}
            injection = torch.tensor(
                [fine_lookup[(2 * ix, 2 * iy, ch)] for ix, iy, ch in coarse_dofs],
                dtype=torch.long,
                device=self.device,
            )
            coarse_scale = fine["scale"][injection]
            source, destination, geometric_weights = [], [], []
            for i, (ix, iy, ch) in enumerate(fine["dofs"]):
                xs = [(ix // 2, 1.0)] if ix % 2 == 0 else [(ix // 2, 0.5), (ix // 2 + 1, 0.5)]
                ys = [(iy // 2, 1.0)] if iy % 2 == 0 else [(iy // 2, 0.5), (iy // 2 + 1, 0.5)]
                for cx, wx in xs:
                    for cy, wy in ys:
                        j = coarse_lookup.get((cx, cy, ch))
                        # Boundary coarse homogeneous DOFs are exactly zero.
                        if j is not None:
                            source.append(i)
                            destination.append(j)
                            geometric_weights.append(wx * wy)
            source = torch.tensor(source, dtype=torch.long, device=self.device)
            destination = torch.tensor(destination, dtype=torch.long, device=self.device)
            weights = torch.tensor(geometric_weights, dtype=self.dtype, device=self.device)
            weights = weights * coarse_scale[destination] / fine["scale"][source]
            transfer = IndexedTransfer(
                source, destination, weights, n_fine=len(fine["dofs"]), n_coarse=len(coarse_dofs)
            )
            fine["transfer"] = transfer
            diagonal = torch.zeros(
                len(coarse_dofs), dtype=self.dtype, device=self.device
            ).index_add(0, destination, weights.square() * fine["diagonal"][source])
            if not bool(torch.all(torch.isfinite(diagonal) & (diagonal > 0)).item()):
                raise ValueError("Nonpositive coarse diagonal proxy; inspect transfer support")

            def coarse_action(value, parent_action=fine["action"], mapping=transfer):
                self.stats["transfer_prolong_calls"] += 1
                lifted = mapping.prolong(value)
                response = parent_action(lifted)
                self.stats["transfer_restrict_calls"] += 1
                return mapping.restrict(response)

            self.levels.append(
                dict(
                    action=coarse_action,
                    diagonal=diagonal,
                    dofs=coarse_dofs,
                    nx=nx,
                    ny=ny,
                    scale=coarse_scale,
                    transfer=None,
                )
            )
        self.metadata = dict(
            method="matrix-free geometric positive-block V-cycle",
            scope="intact regular rectangle; only top/bottom all-channel Dirichlet; no free ties",
            hierarchy=[
                dict(nx=level["nx"], ny=level["ny"], free_dofs=len(level["dofs"]))
                for level in self.levels
            ],
            pre_smooth=pre_smooth,
            post_smooth=post_smooth,
            omega=omega,
            coarse_iterations=coarse_iterations,
            coarse_rtol=coarse_rtol,
            smoother="residual-dependent energy-damped diagonal direction",
            coarse_diagonal="positive sum(P_ij^2 diag_i) proxy; not exact Galerkin or proven bound",
            transfer="indexed channel-separated physical bilinear with scale_coarse/scale_fine",
            restriction="transpose sum, not averaging",
            outer_iteration_requires_flexible_preconditioning=True,
            piezo_cross_blocks_in_preconditioner=False,
            piezo_cross_blocks_in_outer_operator_unchanged=True,
            coarse_action_uses_finest_graph=True,
            global_K_P_T_constructed=False,
            dense_or_sparse_solver_calls=0,
            trained_network=False,
            speed_advantage_claimed=False,
            setup_seconds=time.perf_counter() - tick,
        )

    def _smooth(self, level, rhs, x, residual, count):
        for _ in range(count):
            z = residual / level["diagonal"]
            numerator = _positive_dot(residual, z, "smoother numerator")
            if _scalar(numerator) == 0.0:
                break
            if _scalar(numerator) < 0.0:
                raise RuntimeError("Nonpositive diagonal direction in V-cycle")
            az = level["action"](z)
            curvature = _positive_dot(z, az, "smoother curvature")
            if _scalar(curvature) <= 0.0:
                raise RuntimeError("Nonpositive block curvature; SPD V-cycle assumptions failed")
            optimum = numerator / curvature
            alpha = torch.clamp(optimum, max=self.omega)
            self.stats["energy_limited_smoothing_steps"] += int(_scalar(optimum) < self.omega)
            x = x + alpha * z
            residual = residual - alpha * az
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
        rz = _positive_dot(residual, z, "coarse CG initial")
        for _ in range(self.coarse_iterations):
            ad = level["action"](direction)
            curvature = _positive_dot(direction, ad, "coarse CG curvature")
            if _scalar(curvature) <= 0.0 or _scalar(rz) <= 0.0:
                raise RuntimeError("Nonpositive coarse CG curvature; SPD assumptions failed")
            alpha = rz / curvature
            x = x + alpha * direction
            residual = residual - alpha * ad
            self.stats["coarse_cg_steps"] += 1
            if _norm(residual) <= target:
                return x
            z = residual / level["diagonal"]
            next_rz = _positive_dot(residual, z, "coarse CG updated")
            if _scalar(next_rz) == 0.0:
                return x
            direction = z + (next_rz / rz) * direction
            rz = next_rz
        self.stats["coarse_cg_budget_exhaustions"] += 1
        # Budget exhaustion is a weak correction, not outer-solve convergence.
        return x

    def _cycle(self, index, rhs):
        level = self.levels[index]
        if level["transfer"] is None:
            return self._coarse_cg(level, rhs)
        x = torch.zeros_like(rhs)
        residual = rhs.clone()
        x, residual = self._smooth(level, rhs, x, residual, self.pre_smooth)
        self.stats["transfer_restrict_calls"] += 1
        correction_rhs = level["transfer"].restrict(residual)
        coarse_correction = self._cycle(index + 1, correction_rhs)
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
            raise ValueError("V-cycle requires a finite real-double vector of its free size/device")
        tick = time.perf_counter()
        self.stats["calls"] += 1
        result = self._cycle(0, value)
        self.stats["elapsed_apply_seconds"] += time.perf_counter() - tick
        if not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("Nonfinite V-cycle correction")
        return result

    def diagnostics(self):
        return dict(metadata=dict(self.metadata), stats=dict(self.stats))


def make_block_vcycle_factory(
    *,
    pre_smooth=2,
    post_smooth=2,
    omega=0.7,
    coarse_iterations=24,
    coarse_rtol=0.05,
    max_levels=10,
    min_coarse_cells=4,
):
    """Return factory(problem, theta, scale, action) for MessageEquilibrium.

    Example::

        factory = make_block_vcycle_factory(coarse_iterations=24)
        problem = MessageEquilibrium(graph, constraints,
                                     preconditioner_factory=factory)

    A new hierarchy/counter object is constructed for each reduced solve.  No
    caching across different materials/constraints is silently performed.
    """

    def factory(problem, theta, scale, action):
        return BlockVCyclePreconditioner(
            problem,
            theta,
            scale,
            action,
            pre_smooth=pre_smooth,
            post_smooth=post_smooth,
            omega=omega,
            coarse_iterations=coarse_iterations,
            coarse_rtol=coarse_rtol,
            max_levels=max_levels,
            min_coarse_cells=min_coarse_cells,
        )

    return factory


block_vcycle_factory = make_block_vcycle_factory
