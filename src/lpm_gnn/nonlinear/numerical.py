"""Newton-FGMRES and replay of a recorded Krylov/Newton budget."""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from pyamg.krylov import fgmres


class Newton:
    def __init__(self, cycle):
        self.c = cycle
        self.physical = cycle.physical
        self.physical.build_multigrid()
        s = self.physical.scaling

        # The neural fixed-budget smoother uses omega=.65. Its squared
        # high-frequency relaxation polynomial can be singular as a Krylov
        # preconditioner. A smaller, validation-tested numerical damping is
        # used here; it changes neither the neural predictor nor its training.
        def stable_apply(rhs):
            x = 0.3 * rhs / cycle.diag
            x += cycle.p @ cycle.lu.solve(cycle.pt @ (rhs - cycle.k @ x))
            x += 0.3 * (rhs - cycle.k @ x) / cycle.diag
            return x

        self.coupled = sla.LinearOperator(
            cycle.k.shape, matvec=lambda r: stable_apply(r / s) / s, dtype=np.float64
        )

    def solve(
        self, b, lift, params, preconditioner="coupled", capture=False, tol=1e-10, max_outer=20
    ):
        p = self.physical
        c = self.c
        tic = time.perf_counter()
        y = b[c.top, 1] @ c.basis + params[1] * c.voltage
        b = b.ravel()
        lift = lift.ravel()
        traces = []
        fields = []
        den = max(np.linalg.norm(p.scaling * (b - p.kh @ lift)[p.free]), 1e-30)
        inner_total = 0
        converged = False

        def record(kind, outer, y, inner=None):
            traces.append(
                dict(kind=kind, outer=outer, inner=inner, seconds=time.perf_counter() - tic)
            )
            fields.append(y.copy())

        if capture:
            record("linear_initial", 0, y)
        for outer in range(max_outer):
            r = (b - p.action(y))[p.free]
            rel = np.linalg.norm(p.scaling * r) / den
            if rel < tol:
                converged = True
                break
            j = p.jacobian(y)
            s = sp.diags(p.scaling)
            scaled = (s @ j @ s).tocsr()
            count = [0]

            def callback(delta):
                count[0] += 1
                if capture:
                    value = y.copy()
                    value[p.free] += p.scaling * delta
                    record("Krylov_candidate", outer, value, count[0])

            result, code = fgmres(
                scaled,
                p.scaling * r,
                M=self.coupled if preconditioner == "coupled" else p.preconditioner,
                tol=min(0.02, max(1e-8, 0.1 * rel)),
                restart=100,
                maxiter=6,
                callback=callback,
            )
            if code:
                raise RuntimeError("FGMRES nonconvergence " + str(code))
            inner_total += count[0]
            delta = p.scaling * result
            alpha = 1.0
            for _ in range(16):
                trial = y.copy()
                trial[p.free] += alpha * delta
                nr = np.linalg.norm(p.scaling * (b - p.action(trial))[p.free]) / den
                if nr < (1 - 1e-4 * alpha) * rel:
                    y = trial
                    break
                alpha *= 0.5
            else:
                raise RuntimeError("Newton line search nonconvergence")
            if capture:
                record("accepted_Newton", outer + 1, y)
        final = np.linalg.norm(p.scaling * (b - p.action(y))[p.free]) / den
        converged = bool(final < tol)
        if max_outer == 20 and not converged:
            raise RuntimeError("Newton nonconvergence")
        return (
            y.reshape(-1, 3),
            dict(
                seconds=time.perf_counter() - tic,
                relative_residual=float(final),
                inner_iterations=inner_total,
                converged=converged,
                preconditioner=preconditioner,
            ),
            traces,
            fields,
        )


class _BudgetReached(Exception):
    pass


class PrefixNewton(Newton):
    def replay(self, b, lift, params, preconditioner, event):
        p = self.physical
        c = self.c
        y = b[c.top, 1] @ c.basis + params[1] * c.voltage
        if event["kind"] == "linear_initial":
            return y.reshape(-1, 3)
        b = b.ravel()
        lift = lift.ravel()
        den = max(np.linalg.norm(p.scaling * (b - p.kh @ lift)[p.free]), 1e-30)
        for outer in range(20):
            r = (b - p.action(y))[p.free]
            rel = np.linalg.norm(p.scaling * r) / den
            if rel < 1e-10:
                raise RuntimeError("Requested event was not reached before convergence")
            j = p.jacobian(y)
            s = sp.diags(p.scaling)
            scaled = (s @ j @ s).tocsr()
            count = [0]
            candidate = [None]

            def callback(delta):
                count[0] += 1
                if (
                    event["kind"] == "Krylov_candidate"
                    and outer == event["outer"]
                    and count[0] == event["inner"]
                ):
                    value = y.copy()
                    value[p.free] += p.scaling * delta
                    candidate[0] = value
                    raise _BudgetReached()

            try:
                result, code = fgmres(
                    scaled,
                    p.scaling * r,
                    M=self.coupled if preconditioner == "coupled" else p.preconditioner,
                    tol=min(0.02, max(1e-8, 0.1 * rel)),
                    restart=100,
                    maxiter=6,
                    callback=callback,
                )
            except _BudgetReached:
                return candidate[0].reshape(-1, 3)
            if code:
                raise RuntimeError("FGMRES nonconvergence " + str(code))
            delta = p.scaling * result
            alpha = 1.0
            for _ in range(16):
                trial = y.copy()
                trial[p.free] += alpha * delta
                nr = np.linalg.norm(p.scaling * (b - p.action(trial))[p.free]) / den
                if nr < (1 - 1e-4 * alpha) * rel:
                    y = trial
                    break
                alpha *= 0.5
            else:
                raise RuntimeError("Newton line search nonconvergence")
            if event["kind"] == "accepted_Newton" and outer + 1 == event["outer"]:
                return y.reshape(-1, 3)
        raise RuntimeError("Requested event was not reached")
