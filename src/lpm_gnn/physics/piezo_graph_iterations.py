"""Matrix-free, vector-only restarted flexible GMRES in PyTorch.

The solver calls a homogeneous linear ``operator(v)`` and an optional right
``preconditioner(v)``.  The latter may be stateful/nonstationary: distinct
preconditioned basis vectors are saved (FGMRES, not fixed-preconditioner GMRES).
Neither a global matrix nor a dense/sparse linear solver is constructed/called.
Only vectors and lists of scalar Hessenberg/Givens entries are retained.
The least-squares update is accumulated via vector directions Z R^{-1}; no
triangular solve is repeated on every inner step. This is algebraically the
same right-FGMRES least-squares candidate, not a different physical solver.

Every inner iteration forms its candidate and explicitly evaluates b-Ax.  Only
that TRUE residual can stop the iteration; the Givens estimate is diagnostic.
This costs an extra top-level operator call per iteration and is not optimized.
Counts include initial and true-residual actions, but not actions hidden inside
a user-supplied preconditioner.  No learning or speed advantage is claimed.

Supported states are one-dimensional dense float64 or complex128 tensors.
Iterations are forward-only under no_grad, not an unrolled differentiable
solver.  Inverse problems should separately use a justified implicit derivative.
"""

import math
import time

import torch


def _norm(vector):
    # Dot reductions only; no torch.linalg solver/norm or matrix representation.
    # Scale first: finite large/small vectors must not overflow/underflow in vdot.
    if vector.numel() == 0:
        return torch.zeros((), dtype=vector.real.dtype, device=vector.device)
    scale = torch.abs(vector).max()
    if _number(scale) == 0.0:
        return scale
    scaled = vector / scale
    return scale * torch.sqrt(torch.clamp(torch.vdot(scaled, scaled).real, min=0.0))


def _number(scalar):
    return float(scalar.detach().item())


def _check_vector(vector, template, name):
    if not isinstance(vector, torch.Tensor):
        raise TypeError("%s must return a torch.Tensor" % name)
    if vector.layout != torch.strided or vector.ndim != 1 or vector.shape != template.shape:
        raise ValueError("%s must have the same dense vector shape as rhs" % name)
    if vector.dtype != template.dtype or vector.device != template.device:
        raise TypeError("%s must preserve rhs dtype and device" % name)
    if not bool(torch.isfinite(vector).all().item()):
        raise FloatingPointError("Nonfinite values in %s" % name)
    return vector


def _givens(first, second):
    """Unitary scalar rotation [[c,s],[-conj(s),c]] annihilating second."""
    abs_first, abs_second = torch.abs(first), torch.abs(second)
    if _number(abs_second) == 0.0:
        return torch.ones_like(abs_first), torch.zeros_like(first), first
    if _number(abs_first) == 0.0:
        # c=0; s*b=|b|, also for a complex b.
        return (
            torch.zeros_like(abs_first),
            torch.conj(second) / abs_second,
            abs_second.to(first.dtype),
        )
    magnitude = torch.hypot(abs_first, abs_second)
    phase = first / abs_first
    cosine = abs_first / magnitude
    sine = phase * torch.conj(second) / magnitude
    return cosine, sine, phase * magnitude


def _back_substitute(columns, transformed_rhs, count, relative_pivot_tolerance):
    """Hand back substitution using scalar lists, never a dense solve.

    columns[j][i] is R_ij.  Returns (coefficients, reason).  A numerically zero
    pivot is reported rather than silently replaced, inverted, or regularized.
    """
    coefficients = [None] * count
    tiny = torch.finfo(torch.float64).tiny
    for row in range(count - 1, -1, -1):
        diagonal = columns[row][row]
        column_scale = max(
            (_number(torch.abs(value)) for value in columns[row][: row + 1]), default=0.0
        )
        if _number(torch.abs(diagonal)) <= relative_pivot_tolerance * max(column_scale, tiny):
            return None, "singular_scalar_triangular_factor"
        value = transformed_rhs[row].clone()
        for column in range(row + 1, count):
            value = value - columns[column][row] * coefficients[column]
        coefficients[row] = value / diagonal
        if not bool(torch.isfinite(coefficients[row]).item()):
            return None, "nonfinite_scalar_back_substitution"
    return coefficients, None


@torch.no_grad()
def solve(
    operator,
    rhs,
    x0=None,
    preconditioner=None,
    rtol=1e-9,
    atol=0.0,
    maxiter=1000,
    restart=40,
    *,
    breakdown_tol=None,
    reorthogonalize=True,
    callback=None,
):
    """Solve A x = rhs with restarted right-preconditioned FGMRES.

    Parameters
    ----------
    operator : callable vector -> vector
        Fixed homogeneous linear graph action.  It must not mutate its input.
    preconditioner : optional callable vector -> vector
        Right inverse approximation, allowed to vary by call.  It must not
        mutate its input or change vector shape/dtype/device.  None is identity.
        A vector diagonal can be captured in a closure, e.g. lambda v: v/diag.
    maxiter : int
        Maximum total Arnoldi steps across ALL restart cycles, not cycle count.
    restart : int
        Maximum number of basis vectors/Arnoldi steps in a cycle.
    rtol, atol : float
        Converged iff norm(rhs-operator(x)) <= max(atol, rtol*norm(rhs)).
    breakdown_tol : optional float
        Relative numerical-dependence/pivot threshold; default 64*float64 eps.
    callback : optional callable(record)
        Receives a copy of each trace record (initial and true-residual steps).

    Returns
    -------
    x, info
        info includes true residuals, time trace, top-level matvec counts,
        preconditioner calls, iterations/cycles, convergence and breakdown reason.
        Arnoldi/Givens residual estimates never independently certify convergence.
        Numerical rank breakdown returns the latest truly evaluated x, not NaNs.
        Malformed/nonfinite operator or preconditioner outputs raise explicitly.

    Zero RHS admits x=0 for a linear action: if a nonzero x0 is not already a
    solution, zero is checked explicitly, avoiding an unattainable relative-only
    target caused by roundoff.  This is not a general nonlinear equation solver.
    """
    if not callable(operator) or (preconditioner is not None and not callable(preconditioner)):
        raise TypeError("operator and optional preconditioner must be callable")
    if not isinstance(rhs, torch.Tensor) or rhs.layout != torch.strided or rhs.ndim != 1:
        raise TypeError("rhs must be a dense one-dimensional torch.Tensor")
    if rhs.dtype not in (torch.float64, torch.complex128):
        raise TypeError("Only float64 and complex128 are supported")
    if not bool(torch.isfinite(rhs).all().item()):
        raise ValueError("rhs must be finite")
    for name, value in (("maxiter", maxiter), ("restart", restart)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("%s must be an integer" % name)
    if maxiter < 0 or restart < 1:
        raise ValueError("Require maxiter>=0 and restart>=1")
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("rtol/atol must be finite and nonnegative")
    if breakdown_tol is None:
        breakdown_tol = 64 * torch.finfo(torch.float64).eps
    if not math.isfinite(breakdown_tol) or breakdown_tol < 0:
        raise ValueError("breakdown_tol must be finite and nonnegative")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    if x0 is None:
        x = torch.zeros_like(rhs)
    else:
        x = _check_vector(x0, rhs, "x0").detach().clone()
    rhs_fixed = rhs.detach().clone()
    tick = time.perf_counter()
    matvecs, preconditioner_calls, iterations, cycles = 0, 0, 0, 0
    peak_vectors, peak_hessenberg_scalars = 0, 0
    trace = []
    rhs_norm = _number(_norm(rhs_fixed))
    if not math.isfinite(rhs_norm):
        raise FloatingPointError("rhs norm is not representable in float64")
    tolerance = max(float(atol), float(rtol) * rhs_norm)
    reference_norm = rhs_norm if rhs_norm > 0 else 1.0
    last_estimate = None

    def action(vector):
        nonlocal matvecs
        matvecs += 1
        return _check_vector(operator(vector), rhs_fixed, "operator output")

    def record(residual_norm, kind, estimate=None):
        entry = {
            "iteration": iterations,
            "cycle": cycles,
            "kind": kind,
            "elapsed_seconds": time.perf_counter() - tick,
            "true_residual_norm": float(residual_norm),
            "relative_true_residual": float(residual_norm / reference_norm),
            "arnoldi_residual_estimate": None if estimate is None else float(estimate),
            "matvecs": matvecs,
            "preconditioner_calls": preconditioner_calls,
        }
        trace.append(entry)
        if callback is not None:
            callback(dict(entry))

    def finish(residual_norm, converged, reason):
        return x, {
            "algorithm": "restarted_right_flexible_GMRES_scalar_Givens",
            "converged": bool(converged),
            "reason": reason,
            "breakdown": reason
            not in ("converged", "initial_true_residual", "zero_rhs", "maxiter"),
            "iterations": iterations,
            "cycles": cycles,
            "restarts": max(0, cycles - 1),
            "restart": restart,
            "maxiter": maxiter,
            "matvecs": matvecs,
            "preconditioner_calls": preconditioner_calls,
            "rhs_norm": rhs_norm,
            "tolerance": tolerance,
            "true_residual_norm": float(residual_norm),
            "relative_true_residual": float(residual_norm / reference_norm),
            "relative_residual_reference": "norm(rhs) if positive, otherwise 1",
            "last_arnoldi_residual_estimate": last_estimate,
            "elapsed_seconds": time.perf_counter() - tick,
            "peak_basis_vectors_V_plus_update_directions": peak_vectors,
            "peak_scalar_hessenberg_entries": peak_hessenberg_scalars,
            "global_matrix_assembled": False,
            "dense_or_sparse_solve_calls": 0,
            "autograd_iteration_tape": False,
            "matvec_count_scope": "top-level operator calls; preconditioner internals are not included",
            "true_residual_check_every_inner_step": True,
            "trace": trace,
        }

    residual = rhs_fixed - action(x)
    residual_norm = _number(_norm(residual))
    if not math.isfinite(residual_norm):
        raise FloatingPointError("Initial true residual norm is nonfinite")
    record(residual_norm, "initial")
    if residual_norm <= tolerance:
        return finish(residual_norm, True, "initial_true_residual")
    if rhs_norm == 0:
        x = torch.zeros_like(rhs_fixed)
        residual = rhs_fixed - action(x)
        residual_norm = _number(_norm(residual))
        record(residual_norm, "zero_rhs_check")
        if residual_norm <= tolerance:
            return finish(residual_norm, True, "zero_rhs")
        # A(0)!=0 indicates the required homogeneous linear-action contract failed.
        return finish(residual_norm, False, "nonhomogeneous_operator_zero_rhs")
    if maxiter == 0:
        return finish(residual_norm, False, "maxiter")

    while iterations < maxiter:
        cycles += 1
        base_x = x.clone()
        beta = _norm(residual)
        vectors = [residual / beta]
        update_directions, columns, cosines, sines = [], [], [], []
        transformed_rhs = [beta.to(rhs.dtype)]
        cycle_steps = min(restart, maxiter - iterations)
        for column in range(cycle_steps):
            if preconditioner is None:
                z = vectors[column].clone()
            else:
                preconditioner_calls += 1
                z = _check_vector(
                    preconditioner(vectors[column]), rhs_fixed, "preconditioner output"
                ).clone()
            # Clone: identity actions may return their input, which Arnoldi must not mutate.
            work = action(z).clone()
            action_norm = _number(_norm(work))
            if not math.isfinite(action_norm):
                record(residual_norm, "nonfinite_arnoldi_action_norm", last_estimate)
                return finish(residual_norm, False, "nonfinite_arnoldi_action_norm")
            entries = []
            for vector in vectors[: column + 1]:
                coefficient = torch.vdot(vector, work)
                entries.append(coefficient)
                work = work - coefficient * vector
            if reorthogonalize:
                for row, vector in enumerate(vectors[: column + 1]):
                    correction = torch.vdot(vector, work)
                    entries[row] = entries[row] + correction
                    work = work - correction * vector
            next_norm = _norm(work)
            next_number = _number(next_norm)
            dependent = next_number <= breakdown_tol * max(
                action_norm, torch.finfo(torch.float64).tiny
            )
            entries.append(next_norm.to(rhs.dtype))
            if not dependent:
                vectors.append(work / next_norm)
            for row in range(column):
                upper, lower = entries[row], entries[row + 1]
                entries[row] = cosines[row] * upper + sines[row] * lower
                entries[row + 1] = -torch.conj(sines[row]) * upper + cosines[row] * lower
            cosine, sine, diagonal = _givens(entries[column], entries[column + 1])
            cosines.append(cosine)
            sines.append(sine)
            entries[column] = diagonal
            entries[column + 1] = torch.zeros_like(diagonal)
            columns.append(entries)
            transformed_rhs.append(torch.zeros_like(diagonal))
            upper, lower = transformed_rhs[column], transformed_rhs[column + 1]
            transformed_rhs[column] = cosine * upper + sine * lower
            transformed_rhs[column + 1] = -torch.conj(sine) * upper + cosine * lower
            last_estimate = _number(torch.abs(transformed_rhs[column + 1]))
            iterations += 1
            peak_vectors = max(peak_vectors, len(vectors) + len(update_directions) + 1)
            peak_hessenberg_scalars = max(
                peak_hessenberg_scalars, sum(len(values) for values in columns)
            )
            column_scale = max(
                (_number(torch.abs(value)) for value in entries[: column + 1]), default=0.0
            )
            if _number(torch.abs(diagonal)) <= breakdown_tol * max(
                column_scale, torch.finfo(torch.float64).tiny
            ):
                record(residual_norm, "singular_scalar_triangular_factor", last_estimate)
                return finish(residual_norm, False, "singular_scalar_triangular_factor")
            direction = z.clone()
            for coefficient, previous in zip(entries[:column], update_directions):
                direction = direction - coefficient * previous
            direction = direction / diagonal
            update_directions.append(direction)
            candidate = x + transformed_rhs[column] * direction
            if not bool(torch.isfinite(candidate).all().item()):
                record(residual_norm, "nonfinite_candidate", last_estimate)
                return finish(residual_norm, False, "nonfinite_candidate")
            candidate_residual = rhs_fixed - action(candidate)
            candidate_norm = _number(_norm(candidate_residual))
            if not math.isfinite(candidate_norm):
                record(residual_norm, "nonfinite_true_residual", last_estimate)
                return finish(residual_norm, False, "nonfinite_true_residual")
            x, residual, residual_norm = candidate, candidate_residual, candidate_norm
            record(residual_norm, "inner", last_estimate)
            if residual_norm <= tolerance:
                return finish(residual_norm, True, "converged")
            if dependent:
                return finish(residual_norm, False, "arnoldi_breakdown_without_true_convergence")
            # A misleading small recurrence estimate triggers a reliable restart,
            # never an estimated-residual-only success.
            if last_estimate <= tolerance:
                break
    return finish(residual_norm, False, "maxiter")


fgmres = solve
