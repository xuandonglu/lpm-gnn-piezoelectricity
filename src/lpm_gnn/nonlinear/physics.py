"""Fixed-law nonlinear data generator and exact sparse reference physics.

Dimensionless state channels are (ux / 2e-8, uy / 2e-8, phi / 100).
The prescribed reversible nonlinear enthalpy is identical in data generation,
PI loss evaluation and the learned physical update.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch
from torch import nn

from lpm_gnn.physics.piezo_frame_message_problem import (
    frame_boundary_ids,
    frame_constraints,
    make_frame_geometry,
)
from lpm_gnn.physics.piezo_message_graph import PiezoMessageGraph

EPS = 1e-6
EF = 5000.0
WM = 115e9 * EPS**2
WD = (8.8541878128e-12 * 635) * EF**2
WC = np.sqrt(WM * WD)
STATE_SCALE = np.array([0.02 * EPS, 0.02 * EPS, 100.0])
ENERGY_SCALE = WM * 0.02**2 * 0.8 * 0.001


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    for attempt in range(30):
        try:
            temp.write_text(data, encoding="utf-8")
            temp.replace(path)
            return
        except PermissionError:
            time.sleep(0.1)
    raise RuntimeError("Could not persist " + str(path))


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def physical_arrays(n):
    mesh = make_frame_geometry(n)
    g = PiezoMessageGraph(mesh)
    numpy = lambda x: x.detach().cpu().numpy()
    cb, cn, cv = [
        numpy(getattr(g, k)) for k in ("coefficient_bond", "coefficient_node", "coefficient_value")
    ]
    direction = numpy(g.bond_direction)
    rows = np.repeat(cb, 2)
    cols = np.column_stack((3 * cn, 3 * cn + 1)).ravel()
    e = sp.coo_matrix(
        ((cv[:, None] * direction[cb]).ravel(), (rows, cols)), shape=(g.nbond, 3 * g.n)
    ).tocsr()
    d = sp.coo_matrix((cv, (cb, 3 * cn + 2)), shape=e.shape).tocsr()
    bs = numpy(g.bond_star)
    ax = numpy(g.axial)
    di = numpy(g.diagonal)
    gather = sp.coo_matrix(
        (np.ones(g.nbond), (bs, np.arange(g.nbond))), shape=(g.nstar, g.nbond)
    ).tocsr()
    sa = gather @ sp.diags(ax) @ e
    sd = gather @ sp.diags(di) @ e
    ps = gather @ sp.diags(ax * direction[:, 1]) @ d
    use = (numpy(g.bond_kind)[cb] >= 2) & (numpy(g.bond_kind)[cb] < 4)
    star = bs[cb[use]]
    node = cn[use]
    signed = cv[use] * direction[cb[use], 1]
    cs = sp.coo_matrix(
        (signed / (2 * g.spacing * EPS), (star, 3 * node + 1)), shape=(g.nstar, 3 * g.n)
    ).tocsr()
    ct = sp.coo_matrix(
        (-signed / (2 * g.spacing * EF), (star, 3 * node + 2)), shape=cs.shape
    ).tocsr()
    spring, tv, eta, alpha, beta = [numpy(x) for x in g._material_coefficients(g.initial_theta)]
    weights = numpy(g.star_scale)
    wb = weights[bs]
    k = (
        e.T @ sp.diags(wb * spring) @ e
        + e.T @ sp.diags(wb * eta * direction[:, 1]) @ d
        + d.T @ sp.diags(wb * eta * direction[:, 1]) @ e
        - d.T @ sp.diags(wb * beta) @ d
        + sa.T @ sp.diags(weights * tv) @ sa
        + sd.T @ sp.diags(weights * tv) @ sd
        + sa.T @ sp.diags(weights * alpha) @ ps
        + ps.T @ sp.diags(weights * alpha) @ sa
    ).tocsr()
    k.eliminate_zeros()
    mask = (frame_constraints(mesh).full_to_free.numpy() >= 0).reshape(g.n, 3)
    bottom, top = frame_boundary_ids(mesh)
    scale = np.tile(STATE_SCALE, g.n)
    kh = (sp.diags(scale) @ k @ sp.diags(scale) / ENERGY_SCALE).tocsr()
    volume = weights * g.spacing**2
    csh, cth = cs @ sp.diags(scale), ct @ sp.diags(scale)
    diag = np.abs(kh.diagonal()).reshape(g.n, 3)
    if np.any(diag <= 0):
        raise RuntimeError("Nonpositive diagonal magnitude")
    maps = dict(e=e, d=d, sa=sa, sd=sd, ps=ps, cs=cs, ct=ct)
    return dict(
        n=n,
        mesh=mesh,
        graph=g,
        maps=maps,
        k=k,
        kh=kh,
        csh=csh.tocsr(),
        cth=cth.tocsr(),
        spring=spring,
        tv=tv,
        eta=eta,
        alpha=alpha,
        beta=beta,
        wb=wb,
        weights=weights,
        volume=volume,
        mask=mask,
        top=top,
        bottom=bottom,
        diag=diag,
        scale=scale,
    )


def nonlinear_parts(s, t, lib=np):
    """Return (dH/ds, dH/dt, d2H/ds2, d2H/dsdt, d2H/dt2).

    Here s and t are axial strain / EPS and axial electric field / EF.
    With g_k(z) = z**2/2 - log(cosh(k*z))/k**2, the correction is
    .45*WM*(.7*g_1(s)+.3*g_1.8(s))
    -.35*WD*(.65*g_1(t)+.35*g_1.5(t))
    +.06*WC*tanh(s)**2*tanh(t)**2.
    """
    ts, ts2 = lib.tanh(s), lib.tanh(1.8 * s)
    tt, tt2 = lib.tanh(t), lib.tanh(1.5 * t)
    fs, ft = ts**2, tt**2
    ds, dt = 2 * ts * (1 - ts**2), 2 * tt * (1 - tt**2)
    ss, st = 2 * (1 - ts**2) * (1 - 3 * ts**2), 2 * (1 - tt**2) * (1 - 3 * tt**2)
    gs = 0.45 * WM * (0.7 * (s - ts) + 0.3 * (s - ts2 / 1.8)) + 0.06 * WC * ds * ft
    gt = -0.35 * WD * (0.65 * (t - tt) + 0.35 * (t - tt2 / 1.5)) + 0.06 * WC * fs * dt
    hss = 0.45 * WM * (0.7 * ts**2 + 0.3 * ts2**2) + 0.06 * WC * ss * ft
    htt = -0.35 * WD * (0.65 * tt**2 + 0.35 * tt2**2) + 0.06 * WC * fs * st
    hst = 0.06 * WC * ds * dt
    return gs, gt, hss, hst, htt


def load_case(a, case):
    mesh = a["mesh"]
    top = a["top"]
    ordered = top[np.argsort(mesh.coordinates[top, 0])]
    x = mesh.coordinates[ordered, 0]
    qw = np.full(len(x), mesh.spacing)
    qw[[0, -1]] *= 0.5
    shape = np.exp(-0.5 * ((x - case["center_mm"] * 0.001) / (case["sigma_mm"] * 0.001)) ** 2) * qw
    shape /= shape.sum()
    applied = np.zeros((mesh.n, 3))
    applied[ordered, 1] = case["force_N"] * shape
    lift = np.zeros_like(applied)
    lift[:, 2] = mesh.coordinates[:, 1] / 0.02 * case["voltage_V"]
    bh = applied * STATE_SCALE[None] / ENERGY_SCALE
    lh = lift / STATE_SCALE[None]
    return applied, bh, lh


def sample_cases(size, seed, split):
    rng = np.random.default_rng(seed)
    numbers = rng.uniform(size=(size, 4))
    return [
        dict(
            id=f"{split}_{i:04d}",
            force_N=float(-0.25 + 0.5 * z[0]),
            voltage_V=float(-100 + 200 * z[1]),
            center_mm=float(4 + 12 * z[2]),
            sigma_mm=float(1 + 2 * z[3]),
        )
        for i, z in enumerate(numbers)
    ]


class SparsePhysics:
    def __init__(self, n=40):
        tic = time.perf_counter()
        self.a = physical_arrays(n)
        a = self.a
        self.free = np.flatnonzero(a["mask"].ravel())
        self.kh = a["kh"]
        self.cs = a["csh"]
        self.ct = a["cth"]
        self.diag = a["diag"].ravel()[self.free]
        self.scaling = 1 / np.sqrt(self.diag)
        self.kfree = self.kh[self.free][:, self.free].tocsr()
        self.csfree = self.cs[:, self.free]
        self.ctfree = self.ct[:, self.free]
        self.linear_lu = sla.splu(self.kfree.tocsc())
        self.setup_seconds = time.perf_counter() - tic

    def action(self, y):
        s = self.cs @ y
        t = self.ct @ y
        gs, gt, *_ = nonlinear_parts(s, t)
        vol = self.a["volume"] / ENERGY_SCALE
        return self.kh @ y + self.cs.T @ (vol * gs) + self.ct.T @ (vol * gt)

    def jacobian(self, y):
        _, _, ss, st, tt = nonlinear_parts(self.cs @ y, self.ct @ y)
        v = self.a["volume"] / ENERGY_SCALE
        cs = self.csfree
        ct = self.ctfree
        return (
            self.kfree
            + cs.T @ sp.diags(v * ss) @ cs
            + cs.T @ sp.diags(v * st) @ ct
            + ct.T @ sp.diags(v * st) @ cs
            + ct.T @ sp.diags(v * tt) @ ct
        ).tocsr()

    def solve(
        self,
        case,
        *,
        method="direct",
        tol=1e-10,
        trace_fields=False,
        initial="lift",
        inner_rtol=0.02,
    ):
        tic = time.perf_counter()
        applied, bh, lh = load_case(self.a, case)
        b = bh.ravel()
        y = lh.ravel().copy()
        linear_rhs = (b - self.kh @ y)[self.free]
        den = max(np.linalg.norm(self.scaling * linear_rhs), 1e-30)
        if initial == "linear":
            y[self.free] += self.linear_lu.solve(linear_rhs)
        histories = []
        snapshots = []
        steps = 0
        inner = 0
        for outer in range(20):
            r = (b - self.action(y))[self.free]
            rel = np.linalg.norm(self.scaling * r) / den
            row = dict(
                newton=outer, relative_residual=float(rel), seconds=time.perf_counter() - tic
            )
            histories.append(row)
            if trace_fields:
                snapshots.append(y.copy())
            if rel < tol:
                break
            j = self.jacobian(y)
            if method == "direct":
                dy = sla.spsolve(j.tocsc(), r)
                used = 1
            elif method == "fgmres":
                from pyamg.krylov import fgmres

                scaled = sp.diags(self.scaling) @ j @ sp.diags(self.scaling)
                count = []
                target = min(inner_rtol, max(1e-8, 0.1 * rel))
                sol, info = fgmres(
                    scaled,
                    self.scaling * r,
                    M=self.preconditioner,
                    tol=target,
                    restart=100,
                    maxiter=6,
                    callback=lambda x: count.append(1),
                )
                dy = self.scaling * sol
                used = len(count)
                row["linear_info"] = int(info)
                if info != 0:
                    raise RuntimeError("FGMRES failed " + str(case["id"]) + " " + str(row))
            else:
                raise ValueError(method)
            inner += used
            alpha = 1.0
            for ls in range(16):
                trial = y.copy()
                trial[self.free] += alpha * dy
                rr = (b - self.action(trial))[self.free]
                rrnorm = np.linalg.norm(self.scaling * rr) / den
                if np.isfinite(rrnorm) and rrnorm < (1 - 1e-4 * alpha) * rel:
                    y = trial
                    break
                alpha *= 0.5
            else:
                raise RuntimeError("Line search failed " + str(case["id"]))
            row.update(step_length=alpha, linear_iterations=used)
            steps += 1
        fresh = np.linalg.norm(self.scaling * (b - self.action(y))[self.free]) / den
        if fresh >= tol:
            raise RuntimeError("Reference nonconvergence " + str(case["id"]) + " " + str(fresh))
        info = dict(
            case=case,
            method=method,
            initial=initial,
            converged=True,
            relative_residual=float(fresh),
            newton_steps=steps,
            linear_iterations=inner,
            seconds=time.perf_counter() - tic,
            trace=histories,
        )
        return y.reshape(-1, 3), info, snapshots

    def build_multigrid(self):
        """Geometry-matched positive-block V-cycle, numerical baseline only."""
        coarse = physical_arrays(20)
        p, inject = transfer_arrays(self.a, coarse)
        pf = sp.kron(p, sp.eye(3), format="csr")[self.free][
            :, np.flatnonzero(coarse["mask"].ravel())
        ]
        k = self.kfree
        channels = self.free % 3
        coo = k.tocoo()
        same = (channels[coo.row] < 2) == (channels[coo.col] < 2)
        values = coo.data[same] * np.where(channels[coo.row[same]] < 2, 1.0, -1.0)
        q = sp.coo_matrix((values, (coo.row[same], coo.col[same])), shape=k.shape).tocsr()
        s = sp.diags(self.scaling)
        q = (s @ q @ s).tocsr()
        sc = 1 / np.sqrt(coarse["diag"].ravel()[np.flatnonzero(coarse["mask"].ravel())])
        ps = (sp.diags(1 / self.scaling) @ pf @ sp.diags(sc)).tocsr()
        qc = (ps.T @ q @ ps).tocsc()
        coarse_lu = sla.splu(qc)
        diag = q.diagonal()

        def apply(r):
            x = np.zeros_like(r)
            for _ in range(2):
                x += 0.6 * (r - q @ x) / diag
            x += ps @ coarse_lu.solve(ps.T @ (r - q @ x))
            for _ in range(2):
                x += 0.6 * (r - q @ x) / diag
            return x

        self.preconditioner = sla.LinearOperator(q.shape, matvec=apply, dtype=np.float64)
        self.mg_metadata = dict(
            levels=[40, 20],
            pre_smoothing=2,
            post_smoothing=2,
            omega=0.6,
            auxiliary="positive elastic/dielectric blocks; full coupled tangent in outer FGMRES",
            coarse_solver="sparse LU factored once during shared setup",
            sparse_assembled=True,
        )


def transfer_arrays(fine, coarse):
    from .graph_layers import make_transfer

    t = make_transfer(fine["mesh"], coarse["mesh"], "cpu")
    fi = t["fine"].numpy()
    ci = t["coarse"].numpy()
    w = t["prolong"].numpy()
    p = sp.coo_matrix((w, (fi, ci)), shape=(fine["mesh"].n, coarse["mesh"].n)).tocsr()
    lookup = {
        tuple(np.rint(x / fine["mesh"].spacing).astype(int)): i
        for i, x in enumerate(fine["mesh"].coordinates)
    }
    inject = np.array(
        [
            lookup[tuple(np.rint(x / fine["mesh"].spacing).astype(int))]
            for x in coarse["mesh"].coordinates
        ]
    )
    return p, inject


def tensor_sparse(a, device, dtype):
    a = a.tocoo()
    return torch.sparse_coo_tensor(
        torch.tensor(np.vstack((a.row, a.col)), device=device),
        torch.tensor(a.data, dtype=dtype, device=device),
        a.shape,
        device=device,
    ).coalesce()


def mm(matrix, x):
    """Constant graph incidence applied to a batch of scalar node fields."""
    return torch.sparse.mm(matrix, x.T).T


class GraphPhysics(nn.Module):
    def __init__(self, a, device="cpu", dtype=torch.float32):
        super().__init__()
        self.n = a["mesh"].n
        self.nstar = a["graph"].nstar
        for name, value in a["maps"].items():
            # State is dimensionless, so scale columns once, not each message.
            value = value @ sp.diags(a["scale"])
            self.register_buffer(name, tensor_sparse(value, device, dtype))
            self.register_buffer(name + "_t", tensor_sparse(value.T, device, dtype))
        for name in ("spring", "tv", "eta", "alpha", "beta", "wb", "weights", "volume", "diag"):
            self.register_buffer(name, torch.as_tensor(a[name], dtype=dtype, device=device))
        self.register_buffer("mask", torch.tensor(a["mask"], dtype=dtype, device=device))
        self.register_buffer("ny", a["graph"].bond_direction[:, 1].to(device=device, dtype=dtype))
        self.register_buffer(
            "xy", torch.tensor(a["mesh"].coordinates / 0.02, dtype=dtype, device=device)
        )
        self.register_buffer("volumes", torch.tensor(a["mesh"].volumes, dtype=dtype, device=device))
        self.register_buffer("sign", torch.tensor([1.0, 1.0, -1.0], dtype=dtype, device=device))

    def action(self, y):
        x = y.flatten(1)
        ell = mm(self.e, x)
        delta = mm(self.d, x)
        sa = mm(self.sa, x)
        sd = mm(self.sd, x)
        ps = mm(self.ps, x)
        fl = self.wb * (self.spring * ell + self.eta * self.ny * delta)
        fp = self.wb * (self.eta * self.ny * ell - self.beta * delta)
        out = mm(self.e_t, fl) + mm(self.d_t, fp)
        out = out + mm(self.sa_t, self.weights * (self.tv * sa + self.alpha * ps))
        out = (
            out
            + mm(self.sd_t, self.weights * self.tv * sd)
            + mm(self.ps_t, self.weights * self.alpha * sa)
        )
        gs, gt, *_ = nonlinear_parts(mm(self.cs, x), mm(self.ct, x), torch)
        out = out + mm(self.cs_t, self.volume * gs) + mm(self.ct_t, self.volume * gt)
        # Each map includes state scaling; its transpose is the conjugate action.
        return (out / ENERGY_SCALE).reshape_as(y)

    def residual(self, y, b):
        return (b - self.action(y)) * self.mask

    def jacobi_residual(self, y, b):
        return self.residual(y, b) / self.diag * self.sign

    def constrain(self, y, lift):
        return y * self.mask + lift * (1 - self.mask)


def field_errors(a, prediction, target):
    # Dimensionless displacement scales are equal in x/y. Potential separately.
    p = np.asarray(prediction)
    t = np.asarray(target)
    v = a["mesh"].volumes
    u = np.sum((p[..., :2] - t[..., :2]) ** 2 * v[:, None], axis=(-2, -1)) / np.maximum(
        np.sum(t[..., :2] ** 2 * v[:, None], axis=(-2, -1)), 1e-30
    )
    mask = a["mask"][:, 2] > 0
    e = np.sum((p[..., mask, 2] - t[..., mask, 2]) ** 2 * v[mask], axis=-1) / np.maximum(
        np.sum(t[..., mask, 2] ** 2 * v[mask], axis=-1), 1e-30
    )
    return dict(
        displacement_relative_L2=np.sqrt(u),
        potential_relative_L2=np.sqrt(e),
        combined_relative_L2=np.sqrt((u + e) / 2),
    )
