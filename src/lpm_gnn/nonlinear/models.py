"""Conditional graph predictors and differentiable physical updates."""

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from .graph_layers import GraphLevel, SharedKernel
from .multigrid import CoupledCycle
from .physics import ENERGY_SCALE, mm, nonlinear_parts, tensor_sparse, transfer_arrays


class CompactEnv(nn.Module):
    def __init__(self, basis_path, normalization, device="cpu", dtype=torch.float64):
        super().__init__()
        self.cycle = CoupledCycle(basis_path, 1, 1, 0.65)
        a = self.cycle.physical.a
        coarse = self.cycle.coarse
        self.a = a
        self.device = device
        self.dtype = dtype
        self.graph = GraphLevel(coarse, device, dtype)
        p, inject = transfer_arrays(a, coarse)
        average = (
            sp.diags(1 / np.asarray(p.T @ a["mesh"].volumes).ravel())
            @ p.T
            @ sp.diags(a["mesh"].volumes)
        )
        arrays = dict(
            k=a["kh"],
            cs=a["csh"],
            ct=a["cth"],
            p=p,
            average=average,
            kfree=self.cycle.k,
            pfree=self.cycle.p,
            ptfree=self.cycle.pt,
        )
        arrays["cst"] = a["csh"].T
        arrays["ctt"] = a["cth"].T
        for name, value in arrays.items():
            self.register_buffer(name, tensor_sparse(value, device, dtype))
        self.register_buffer("free", torch.tensor(self.cycle.free, device=device))
        self.register_buffer("inject", torch.tensor(inject, device=device))
        self.register_buffer("mask", torch.tensor(a["mask"], device=device, dtype=dtype))
        self.register_buffer("diag", torch.tensor(a["diag"], device=device, dtype=dtype))
        self.register_buffer(
            "signed_diag", torch.tensor(self.cycle.diag, device=device, dtype=dtype)
        )
        self.register_buffer(
            "vol", torch.tensor(a["volume"] / ENERGY_SCALE, device=device, dtype=dtype)
        )
        self.register_buffer("volume", torch.tensor(a["mesh"].volumes, device=device, dtype=dtype))
        self.register_buffer(
            "field_scale", torch.tensor(normalization["field_scale"], device=device, dtype=dtype)
        )
        self.register_buffer(
            "output_scale", torch.tensor(normalization["output_scale"], device=device, dtype=dtype)
        )
        self.register_buffer(
            "static",
            torch.tensor(
                np.column_stack((2 * a["mesh"].coordinates / 0.02 - 1, a["mask"])),
                device=device,
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "force_basis",
            torch.tensor(
                self.cycle.basis.reshape(len(self.cycle.top), -1, 3), device=device, dtype=dtype
            ),
        )
        self.register_buffer(
            "voltage", torch.tensor(self.cycle.voltage.reshape(-1, 3), device=device, dtype=dtype)
        )
        self.register_buffer("top", torch.tensor(self.cycle.top, device=device))
        inv = self.cycle.lu.solve(np.eye(self.cycle.kc.shape[0]))
        self.register_buffer("coarse_inverse", torch.tensor(inv, device=device, dtype=dtype))
        self.register_buffer(
            "fixed_reference",
            torch.tensor(normalization["reference_squared"], device=device, dtype=dtype),
        )
        self.cpu_lu = False

    def csr(self):
        for name, value in list(self.named_buffers()):
            if value.layout == torch.sparse_coo:
                setattr(self, name, value.to_sparse_csr())
        return self

    def initial(self, b, params):
        return (
            torch.einsum("bt,tnc->bnc", b[:, self.top, 1], self.force_basis)
            + params[:, 1, None, None] * self.voltage
        )

    def act(self, y):
        flat = y.flatten(1)
        s = mm(self.cs, flat)
        t = mm(self.ct, flat)
        gs, gt, *_ = nonlinear_parts(s, t, torch)
        return (
            mm(self.k, flat) + mm(self.cst, gs * self.vol) + mm(self.ctt, gt * self.vol)
        ).reshape_as(y)

    def residual(self, y, b):
        return (b - self.act(y)) * self.mask

    def transfer(self, m, x):
        b, n, c = x.shape
        return (
            torch.sparse.mm(m, x.permute(1, 0, 2).reshape(n, -1))
            .reshape(m.shape[0], b, c)
            .permute(1, 0, 2)
        )

    def apply(self, r):
        rhs = r.flatten(1)[:, self.free]
        z = 0.65 * rhs / self.signed_diag
        rc = mm(self.ptfree, rhs - mm(self.kfree, z))
        if self.cpu_lu:
            assert not rc.requires_grad and rc.device.type == "cpu"
            ec = torch.from_numpy(self.cycle.lu.solve(rc.numpy().T).T.copy()).to(rc.dtype)
        else:
            ec = rc @ self.coarse_inverse.T
        z = z + mm(self.pfree, ec)
        z = z + 0.65 * (rhs - mm(self.kfree, z)) / self.signed_diag
        full = torch.zeros_like(r).flatten(1)
        full[:, self.free] = z
        return full.reshape_as(r)

    def features(self, y, b, lift, params, r=None, coarse=True):
        if r is None:
            rn = torch.zeros_like(y)
        else:
            scale = torch.sqrt((r.square() * self.mask).sum(1) / self.mask.sum(0) + 1e-24)
            rn = r / scale[:, None]
        f = torch.cat(
            (
                self.static[None].expand(len(y), -1, -1),
                b / self.diag,
                lift,
                params[:, None].expand(-1, len(self.mask), -1),
                y / self.field_scale,
                rn,
            ),
            -1,
        )
        return self.transfer(self.average, f) if coarse else f

    def field_loss(self, y, target):
        vol = self.volume
        u = ((y[..., :2] - target[..., :2]).square().sum(-1) * vol).sum(-1)
        ud = (target[..., :2].square().sum(-1) * vol).sum(-1).clamp_min(1e-12)
        mask = self.mask[:, 2] > 0
        ph = ((y[:, mask, 2] - target[:, mask, 2]).square() * vol[mask]).sum(-1)
        pd = (target[:, mask, 2].square() * vol[mask]).sum(-1).clamp_min(1e-12)
        return 0.5 * (u / ud + ph / pd)

    def physics_loss(self, y, b, lift):
        r = self.residual(y, b) / torch.sqrt(self.diag)
        um = self.mask[:, :2] > 0
        pm = self.mask[:, 2] > 0
        ru = r[..., :2][:, um].square().sum(-1) / self.fixed_reference[0]
        rp = r[:, pm, 2].square().sum(-1) / self.fixed_reference[1]
        return 0.5 * (ru + rp)


class CompactPredictor(nn.Module):
    def __init__(self, kind="LearnedWeight", width=24, cycles=1):
        super().__init__()
        self.kind = kind
        self.cycles = cycles
        self.width = width
        self.kernel = SharedKernel(width)
        self.local = nn.Sequential(nn.Linear(width + 21, width), nn.SiLU(), nn.Linear(width, 3))
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)

    def forward(self, env, b, lift, params):
        y = env.initial(b, params)
        if self.kind in ("CompactData", "CompactPI"):
            fc = env.features(y, b, lift, params)
            h = self.kernel.propagate(self.kernel.encoder(fc), env.graph, params, 1, count=3)
            hf = env.transfer(env.p, h)
            features = env.features(y, b, lift, params, coarse=False)
            return y + self.local(torch.cat((hf, features), -1)) * env.output_scale * env.mask
        for stage in range(self.cycles):
            r = env.residual(y, b)
            h = self.kernel.encoder(env.features(y, b, lift, params, r))
            h = self.kernel.propagate(h, env.graph, params, min(stage, 2), count=2)
            # Positive state-dependent residual weights; bounded for the fixed
            # budget, with no claim of repeated-cycle convergence.
            weight = 0.8 + 0.35 * torch.tanh(env.transfer(env.p, self.kernel.decoder(h)))
            y = y + env.apply(r * weight)
        return y


class SchedulePredictor(CompactPredictor):
    def __init__(self, kind="ScheduleWeight", width=16, cycles=2):
        super().__init__(kind, 16, cycles)
        self.kernel.decoder = nn.Sequential(nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 3 * cycles))
        nn.init.zeros_(self.kernel.decoder[-1].weight)
        nn.init.zeros_(self.kernel.decoder[-1].bias)

    def forward(self, env, b, lift, params):
        y = env.initial(b, params)
        r = env.residual(y, b)
        h = self.kernel.encoder(env.features(y, b, lift, params, r))
        h = self.kernel.propagate(h, env.graph, params, 1, count=1)
        weights = 0.8 + 0.35 * torch.tanh(env.transfer(env.p, self.kernel.decoder(h))).reshape(
            len(y), len(env.mask), self.cycles, 3
        )
        for stage in range(self.cycles):
            if stage:
                r = env.residual(y, b)
            y = y + env.apply(r * weights[:, :, stage])
        return y
