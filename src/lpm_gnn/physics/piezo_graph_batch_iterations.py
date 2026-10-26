"""Independent batched vector-only FGMRES; no matrices/factorizations.

Each row is its OWN system, Krylov basis, unitary Givens rotations, scalar
Hessenberg coefficients, update directions and true-residual stopping test.
The batch dimension is only a scheduling dimension, never a coupled model.
The operator/preconditioner must map [B,N] to [B,N] independently by row.
Converged/broken rows are masked from further updates.  An implementation of
the batched operator can still evaluate zero rows: work accounting says so.

No global K/P/T, matrix multiplication, sparse action or linear solve is used.
The scalar-vector update is d_j=(z_j-sum_i R_ij*d_i)/R_jj and x+=g_j*d_j.
Every live row receives an explicit b-Ax test at every inner iteration.
Forward-only under no_grad; this is not an unrolled learned solver.
"""

import math
import time

import torch


def _norm_rows(value):
    scale = value.abs().amax(dim=1)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    reduced = (value / safe[:, None]).abs().square().sum(dim=1)
    return scale * torch.sqrt(reduced.clamp_min(0.0))


def _givens_rows(first, second):
    a, b = first.abs(), second.abs()
    magnitude = torch.hypot(a, b)
    safe_a = torch.where(a > 0, a, torch.ones_like(a))
    safe_m = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
    phase = torch.where(a > 0, first / safe_a, torch.ones_like(first))
    cosine = a / safe_m
    sine = phase * second.conj() / safe_m
    diagonal = phase * magnitude
    # b=0 includes a=b=0: choose the identity rotation, not a zero rotation.
    cosine = torch.where(b == 0, torch.ones_like(cosine), cosine)
    sine = torch.where(b == 0, torch.zeros_like(sine), sine)
    diagonal = torch.where(b == 0, first, diagonal)
    return cosine, sine, diagonal


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
    """Return (x[B,N], batch_info), with batch_info['row_infos'] per system.

    ``maxiter`` limits each row's TOTAL Arnoldi steps across restart cycles.
    A row can converge earlier without changing another row's Krylov update.
    Callback, if given, receives (row_index, copy_of_trace_record).
    Only float64/complex128 are supported.  This is a fixed linear operator
    solve, not a general nonlinear root finder.  Preconditioners may vary.
    """
    if not callable(operator) or (preconditioner is not None and not callable(preconditioner)):
        raise TypeError("Need a callable operator and optional preconditioner")
    if (
        not isinstance(rhs, torch.Tensor)
        or rhs.layout != torch.strided
        or rhs.ndim != 2
        or min(rhs.shape) < 1
        or rhs.dtype not in (torch.float64, torch.complex128)
    ):
        raise TypeError("rhs must be nonempty dense float64/complex128 [B,N]")
    if not bool(torch.isfinite(rhs).all().item()):
        raise ValueError("rhs must be finite")
    for name, value in (("maxiter", maxiter), ("restart", restart)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(name + " must be an integer")
    if maxiter < 0 or restart < 1:
        raise ValueError("Require maxiter>=0 and restart>=1")
    if not all(math.isfinite(v) and v >= 0 for v in (rtol, atol)):
        raise ValueError("rtol/atol must be finite nonnegative")
    if breakdown_tol is None:
        breakdown_tol = 64 * torch.finfo(torch.float64).eps
    if not math.isfinite(breakdown_tol) or breakdown_tol < 0:
        raise ValueError("breakdown_tol must be finite nonnegative")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    b = rhs.detach().clone()
    batch, n = b.shape
    device = b.device
    tick = time.perf_counter()
    batched_actions, batched_preconditioner_calls = 0, 0

    def checked(value, name):
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.shape != b.shape
            or value.dtype != b.dtype
            or value.device != device
        ):
            raise TypeError(name + " must preserve dense batch shape/dtype/device")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError("Nonfinite " + name)
        return value

    def action(value):
        nonlocal batched_actions
        batched_actions += 1
        return checked(operator(value), "operator output")

    x = torch.zeros_like(b) if x0 is None else checked(x0, "x0").detach().clone()
    rhs_norm = _norm_rows(b)
    if not bool(torch.isfinite(rhs_norm).all().item()):
        raise FloatingPointError("RHS norm is not representable")
    tolerance = torch.maximum(torch.full_like(rhs_norm, float(atol)), float(rtol) * rhs_norm)
    reference = torch.where(rhs_norm > 0, rhs_norm, torch.ones_like(rhs_norm))
    iterations = torch.zeros(batch, dtype=torch.long, device=device)
    cycles = torch.zeros_like(iterations)
    logical_actions = torch.ones_like(iterations)
    logical_preconditioner = torch.zeros_like(iterations)
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    failed = torch.zeros_like(done)
    reasons = ["maxiter"] * batch
    traces = [[] for _ in range(batch)]
    completion_times = [None] * batch
    residual = b - action(x)
    residual_norm = _norm_rows(residual)
    if not bool(torch.isfinite(residual_norm).all().item()):
        raise FloatingPointError("Initial true residual is nonfinite")
    last_estimate = torch.full_like(rhs_norm, float("nan"))
    peak_vectors, peak_hessenberg_entries = 0, 0

    def record(mask, kind, estimates=None):
        elapsed = time.perf_counter() - tick
        rows = mask.nonzero(as_tuple=False).flatten().cpu().tolist()
        its, cycs = iterations.cpu().tolist(), cycles.cpu().tolist()
        rn, rr = residual_norm.cpu().tolist(), (residual_norm / reference).cpu().tolist()
        ma, pc = logical_actions.cpu().tolist(), logical_preconditioner.cpu().tolist()
        es = None if estimates is None else estimates.cpu().tolist()
        for row in rows:
            estimate = None if es is None or not math.isfinite(es[row]) else es[row]
            entry = dict(
                iteration=its[row],
                cycle=cycs[row],
                kind=kind,
                elapsed_seconds=elapsed,
                true_residual_norm=rn[row],
                relative_true_residual=rr[row],
                arnoldi_residual_estimate=estimate,
                matvecs=ma[row],
                preconditioner_calls=pc[row],
                batched_action_calls_so_far=batched_actions,
            )
            traces[row].append(entry)
            if callback is not None:
                callback(row, dict(entry))

    def mark(mask, reason, is_failure=False):
        rows = mask.nonzero(as_tuple=False).flatten().cpu().tolist()
        elapsed = time.perf_counter() - tick
        for row in rows:
            reasons[row], completion_times[row] = reason, elapsed
        if is_failure:
            failed.logical_or_(mask)
        else:
            done.logical_or_(mask)

    all_rows = torch.ones_like(done)
    record(all_rows, "initial")
    mark(residual_norm <= tolerance, "initial_true_residual")
    zero_rows = (~done) & (rhs_norm == 0)
    if bool(zero_rows.any().item()):
        # Homogeneous A admits zero; do not let the unrepresentable relative
        # target for rhs=0 consume an iteration budget from another row.
        zero = torch.zeros_like(b)
        at_zero = action(zero)
        logical_actions += zero_rows.to(torch.long)
        x = torch.where(zero_rows[:, None], zero, x)
        residual = torch.where(zero_rows[:, None], b - at_zero, residual)
        residual_norm = _norm_rows(residual)
        record(zero_rows, "zero_rhs_check")
        mark(zero_rows & (residual_norm <= tolerance), "zero_rhs")
        mark(zero_rows & (residual_norm > tolerance), "nonhomogeneous_operator_zero_rhs", True)

    while bool(((~done) & (~failed) & (iterations < maxiter)).any().item()):
        active = (~done) & (~failed) & (iterations < maxiter)
        cycles += active.to(torch.long)
        beta = _norm_rows(residual)
        safe_beta = torch.where(active, beta, torch.ones_like(beta))
        vectors = [
            torch.where(active[:, None], residual / safe_beta[:, None], torch.zeros_like(residual))
        ]
        directions, columns, cosines, sines = [], [], [], []
        transformed_rhs = [torch.where(active, beta, torch.zeros_like(beta)).to(b.dtype)]
        for column in range(restart):
            active &= iterations < maxiter
            if not bool(active.any().item()):
                break
            before = active.clone()
            if preconditioner is None:
                z = vectors[column].clone()
            else:
                batched_preconditioner_calls += 1
                logical_preconditioner += active.to(torch.long)
                z = checked(preconditioner(vectors[column]), "preconditioner output").clone()
            z = torch.where(active[:, None], z, torch.zeros_like(z))
            work = action(z).clone()
            logical_actions += active.to(torch.long)
            action_norm = _norm_rows(work)
            bad_action_norm = active & (~torch.isfinite(action_norm))
            mark(bad_action_norm, "nonfinite_arnoldi_action_norm", True)
            active &= ~bad_action_norm
            entries = []
            for vector in vectors[: column + 1]:
                coefficient = (vector.conj() * work).sum(dim=1)
                entries.append(coefficient)
                work = work - coefficient[:, None] * vector
            if reorthogonalize:
                for row, vector in enumerate(vectors[: column + 1]):
                    correction = (vector.conj() * work).sum(dim=1)
                    entries[row] = entries[row] + correction
                    work = work - correction[:, None] * vector
            next_norm = _norm_rows(work)
            dependent = next_norm <= breakdown_tol * action_norm.clamp_min(
                torch.finfo(torch.float64).tiny
            )
            entries.append(next_norm.to(b.dtype))
            normalizable = active & (~dependent)
            safe_next = torch.where(normalizable, next_norm, torch.ones_like(next_norm))
            vectors.append(
                torch.where(
                    normalizable[:, None], work / safe_next[:, None], torch.zeros_like(work)
                )
            )
            for row in range(column):
                upper, lower = entries[row], entries[row + 1]
                entries[row] = cosines[row] * upper + sines[row] * lower
                entries[row + 1] = -sines[row].conj() * upper + cosines[row] * lower
            cosine, sine, diagonal = _givens_rows(entries[column], entries[column + 1])
            cosines.append(cosine)
            sines.append(sine)
            entries[column], entries[column + 1] = diagonal, torch.zeros_like(diagonal)
            columns.append(entries)
            transformed_rhs.append(torch.zeros_like(diagonal))
            upper, lower = transformed_rhs[column], transformed_rhs[column + 1]
            transformed_rhs[column] = cosine * upper + sine * lower
            transformed_rhs[column + 1] = -sine.conj() * upper + cosine * lower
            estimate = transformed_rhs[column + 1].abs()
            last_estimate = torch.where(active, estimate, last_estimate)
            iterations += before.to(torch.long)
            peak_vectors = max(peak_vectors, len(vectors) + len(directions) + 1)
            peak_hessenberg_entries = max(peak_hessenberg_entries, sum(len(col) for col in columns))
            column_scale = torch.zeros_like(rhs_norm)
            for entry in entries[: column + 1]:
                column_scale = torch.maximum(column_scale, entry.abs())
            singular = active & (
                diagonal.abs()
                <= breakdown_tol * column_scale.clamp_min(torch.finfo(torch.float64).tiny)
            )
            mark(singular, "singular_scalar_triangular_factor", True)
            live = active & (~singular)
            direction = z.clone()
            for coefficient, previous in zip(entries[:column], directions):
                direction = direction - coefficient[:, None] * previous
            safe_diagonal = torch.where(live, diagonal, torch.ones_like(diagonal))
            direction = torch.where(
                live[:, None], direction / safe_diagonal[:, None], torch.zeros_like(direction)
            )
            directions.append(direction)
            candidate = x + transformed_rhs[column][:, None] * direction
            bad_candidate = live & (~torch.isfinite(candidate).all(dim=1))
            mark(bad_candidate, "nonfinite_candidate", True)
            live &= ~bad_candidate
            # Zero the inactive rows before calling the independent batch action.
            candidate_action = action(
                torch.where(live[:, None], candidate, torch.zeros_like(candidate))
            )
            logical_actions += live.to(torch.long)
            candidate_residual = b - candidate_action
            candidate_norm = _norm_rows(candidate_residual)
            bad_residual = live & (~torch.isfinite(candidate_norm))
            mark(bad_residual, "nonfinite_true_residual", True)
            live &= ~bad_residual
            x = torch.where(live[:, None], candidate, x)
            residual = torch.where(live[:, None], candidate_residual, residual)
            residual_norm = torch.where(live, candidate_norm, residual_norm)
            record(before, "inner", last_estimate)
            newly_done = live & (residual_norm <= tolerance)
            mark(newly_done, "converged")
            bad_dependence = live & dependent & (~newly_done)
            mark(bad_dependence, "arnoldi_breakdown_without_true_convergence", True)
            # Small recurrence but not small true residual: wait for next cycle.
            reliable_restart = live & (~newly_done) & (~bad_dependence) & (estimate <= tolerance)
            active = live & (~newly_done) & (~bad_dependence) & (~reliable_restart)

    elapsed = time.perf_counter() - tick
    rn, rel = residual_norm.cpu().tolist(), (residual_norm / reference).cpu().tolist()
    it, cy = iterations.cpu().tolist(), cycles.cpu().tolist()
    ma, pc = logical_actions.cpu().tolist(), logical_preconditioner.cpu().tolist()
    dn, fl = done.cpu().tolist(), failed.cpu().tolist()
    rhsn, tol = rhs_norm.cpu().tolist(), tolerance.cpu().tolist()
    estimates = last_estimate.cpu().tolist()
    row_infos = []
    for row in range(batch):
        row_infos.append(
            dict(
                algorithm="independent_batched_right_FGMRES",
                converged=dn[row],
                reason=reasons[row],
                breakdown=fl[row],
                iterations=it[row],
                cycles=cy[row],
                restarts=max(0, cy[row] - 1),
                restart=restart,
                maxiter=maxiter,
                matvecs=ma[row],
                preconditioner_calls=pc[row],
                rhs_norm=rhsn[row],
                tolerance=tol[row],
                true_residual_norm=rn[row],
                relative_true_residual=rel[row],
                last_arnoldi_residual_estimate=estimates[row]
                if math.isfinite(estimates[row])
                else None,
                elapsed_seconds=completion_times[row]
                if completion_times[row] is not None
                else elapsed,
                batch_elapsed_seconds=elapsed,
                matvec_count_scope="logical actions for this live row; masked zero rows can still be evaluated",
                true_residual_check_every_inner_step=True,
                global_matrix_assembled=False,
                dense_or_sparse_solve_calls=0,
                autograd_iteration_tape=False,
                trace=traces[row],
            )
        )
    return x, dict(
        row_infos=row_infos,
        converged=all(dn),
        batch_size=batch,
        vector_size=n,
        batched_action_calls=batched_actions,
        batched_preconditioner_calls=batched_preconditioner_calls,
        row_actions_evaluated_including_masked_rows=batch * batched_actions,
        peak_vectors_per_row=peak_vectors,
        peak_scalar_hessenberg_entries_per_row=peak_hessenberg_entries,
        elapsed_seconds=elapsed,
        global_matrix_assembled=False,
        dense_or_sparse_solve_calls=0,
        independent_rows=True,
        true_residual_check_every_inner_step=True,
    )


fgmres = solve
