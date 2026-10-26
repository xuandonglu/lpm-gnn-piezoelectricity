"""Constrained pure-message forward/adjoint. No global matrix construction.

The positive scaling is an incidence-summed UNCONSTRAINED diagonal magnitude.
For tied DOFs it is not claimed to equal the constrained operator diagonal.
All diagonal operations are elementwise vector operations. No factorization,
sparse matvec, local/global matrix inverse, or unrolled autograd solve is used.
"""

import time

import torch

from .piezo_graph_iterations import solve


class MessageEquilibrium:
    def __init__(
        self, graph, constraints, rtol=1e-10, maxiter=1500, restart=80, preconditioner_factory=None
    ):
        self.graph = graph
        self.constraints = constraints
        self.rtol, self.maxiter, self.restart = rtol, maxiter, restart
        self.preconditioner_factory = preconditioner_factory
        self.forward_infos, self.adjoint_infos = [], []

    def scaling(self, theta):
        mechanical = self.graph.mechanical_diagonal(theta)
        dielectric = self.graph.dielectric_diagonal(theta)
        magnitude = torch.stack((mechanical[:, 0], mechanical[:, 1], dielectric), dim=1)
        diagonal_bound = self.constraints.contract(magnitude)
        if not bool(torch.all(diagonal_bound > 0)):
            raise ValueError("Nonpositive message scaling; inspect support/geometry/material")
        return torch.rsqrt(diagonal_bound)

    def reduced_solve(self, theta, rhs, initial=None):
        tick = time.perf_counter()
        scale = self.scaling(theta).detach()

        def action(y):
            return scale * self.constraints.action(scale * y, self.graph, theta)

        preconditioner = (
            None
            if self.preconditioner_factory is None
            else self.preconditioner_factory(self, theta, scale, action)
        )
        x0 = None if initial is None else initial / scale
        y, info = solve(
            action,
            scale * rhs,
            x0=x0,
            preconditioner=preconditioner,
            rtol=self.rtol,
            atol=0.0,
            maxiter=self.maxiter,
            restart=self.restart,
        )
        z = scale * y
        residual = rhs - self.constraints.action(z, self.graph, theta)
        block_residual = {}
        for name, mask in (
            ("mechanical", self.constraints.free_mechanical_mask),
            ("electrical", self.constraints.free_electrical_mask),
        ):
            denominator = torch.linalg.vector_norm(rhs[mask])
            numerator = torch.linalg.vector_norm(residual[mask])
            block_residual[name] = float(numerator / torch.clamp(denominator, min=1e-30))
        info = dict(info)
        info.update(
            block_true_residual=block_residual,
            elapsed_wrapper_seconds=time.perf_counter() - tick,
            scaling="positive incidence-summed local diagonal; not exact tied diagonal",
            global_matrix_constructed=False,
        )
        return z, info

    def solve(self, theta, full_rhs=None):
        if full_rhs is None:
            full_rhs = theta.new_zeros((self.graph.n, 3))
        else:
            # Normalize before the custom Function: reshape/to remain ordinary
            # autograd operations, so a flat input receives a flat gradient.
            full_rhs = torch.as_tensor(full_rhs, dtype=theta.dtype, device=theta.device).reshape(
                self.graph.n, 3
            )
        return _ImplicitMessageEquilibrium.apply(theta, full_rhs, self)

    def solve_detached(self, theta, full_rhs=None, initial=None):
        if full_rhs is None:
            full_rhs = theta.new_zeros((self.graph.n, 3))
        with torch.no_grad():
            rhs = self.constraints.rhs(self.graph, theta, full_rhs)
            z, info = self.reduced_solve(theta, rhs, initial)
            state = self.constraints.expand(z)
        self.forward_infos.append(info)
        return state, info


class _ImplicitMessageEquilibrium(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, full_rhs, problem):
        state, info = problem.solve_detached(theta, full_rhs)
        if not info.get("converged", False):
            raise RuntimeError("Pure-message forward failed convergence: " + str(info))
        ctx.problem = problem
        ctx.save_for_backward(theta.detach(), state.detach())
        return state

    @staticmethod
    def backward(ctx, output_gradient):
        theta, state = ctx.saved_tensors
        problem = ctx.problem
        with torch.no_grad():
            reduced_rhs = problem.constraints.contract(output_gradient)
            reduced_adjoint, info = problem.reduced_solve(theta, reduced_rhs)
            adjoint = problem.constraints.prolong(reduced_adjoint)
        problem.adjoint_infos.append(info)
        if not info.get("converged", False):
            raise RuntimeError("Pure-message adjoint failed convergence: " + str(info))
        with torch.enable_grad():
            parameter = theta.detach().requires_grad_(True)
            contraction = -(adjoint * problem.graph.action(state, parameter)).sum()
            (gradient,) = torch.autograd.grad(contraction, parameter)
        return gradient, adjoint, None
