"""Antiparallel-poling PZT-4 bimorph geometry and matrix-free V-cycle."""

import numpy as np
import torch

from .piezo_graph_multilevel import IndexedTransfer
from .piezo_message_geometry import MessageGeometry
from .piezo_message_graph import MessageConstraints, PiezoMessageGraph
from .piezo_message_multigrid import BlockVCyclePreconditioner
from .piezo_message_solve import MessageEquilibrium

L, H, WIDTH = 0.010, 0.001, 0.001
VOLTAGE, FORCE = 10.0, 0.001
SCHEME = "opposite"


class OppositePolingGraph(PiezoMessageGraph):
    """Fixed antiparallel poling on one conforming lattice; identical C and kappa.

    At a center on the layer interface the half-area piezoelectric coefficients
    average to zero. No interface DOFs, grain reconnection or switching model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        y = self.coordinates[self.star_center, 1] - H / 2
        self.register_buffer(
            "poling", torch.where(y.abs() < H * 1e-10, torch.zeros_like(y), torch.sign(y))
        )

    def energy(self, state, theta):
        spring, tv, eta, alpha, beta = self._material_coefficients(theta)
        ell, delta = self._differences(state)
        ny = self.bond_direction[:, 1]
        si = self.bond_star
        sa = self._sum_star(self.axial * ell)
        sd = self._sum_star(self.diagonal * ell)
        p = self._sum_star(self.axial * ny * delta)
        local = self._sum_star(
            0.5 * spring * ell.square()
            + eta * self.poling[si] * ny * ell * delta
            - 0.5 * beta * delta.square()
        )
        return (
            self.star_scale
            * (local + 0.5 * tv * (sa.square() + sd.square()) + alpha * self.poling * sa * p)
        ).sum()

    def action(self, state, theta):
        spring, tv, eta, alpha, beta = self._material_coefficients(theta)
        ell, delta = self._differences(state)
        ny = self.bond_direction[:, 1]
        si = self.bond_star
        sa = self._sum_star(self.axial * ell)
        sd = self._sum_star(self.diagonal * ell)
        p = self._sum_star(self.axial * ny * delta)
        fl = (
            spring * ell
            + tv * (self.axial * sa[si] + self.diagonal * sd[si])
            + self.poling[si] * (eta * ny * delta + alpha * self.axial * p[si])
        )
        fp = -beta * delta + self.poling[si] * (eta * ny * ell + alpha * self.axial * ny * sa[si])
        response = (
            torch.cat((fl[:, None] * self.bond_direction, fp[:, None]), dim=1)
            * self.star_scale[si, None]
        )
        return state.new_zeros((self.n, 3)).index_add(
            0,
            self.coefficient_node,
            self.coefficient_value[:, None] * response[self.coefficient_bond],
        )


def make_graph(mesh):
    return (OppositePolingGraph if SCHEME == "opposite" else PiezoMessageGraph)(
        mesh, thickness=WIDTH
    )


def geometry(ny):
    return MessageGeometry.rectangle(10 * ny, ny, width=L, height=H)


def boundary(mesh, case):
    xy = mesh.coordinates
    left = np.flatnonzero(np.isclose(xy[:, 0], 0, atol=1e-14))
    tip = np.flatnonzero(np.isclose(xy[:, 0], L, atol=1e-14))
    middle = np.flatnonzero(np.isclose(xy[:, 1], H / 2, atol=1e-14))
    outer = np.flatnonzero(
        np.isclose(xy[:, 1], 0, atol=1e-14) | np.isclose(xy[:, 1], H, atol=1e-14)
    )
    if SCHEME == "opposite":
        middle = np.flatnonzero(np.isclose(xy[:, 1], 0, atol=1e-14))
        outer = np.flatnonzero(np.isclose(xy[:, 1], H, atol=1e-14))
    prescribed = {3 * int(i) + c: 0.0 for i in left for c in (0, 1)}
    prescribed.update({3 * int(i) + 2: 0.0 for i in middle})
    if case != "sensor_open":
        prescribed.update(
            {3 * int(i) + 2: (VOLTAGE if case.startswith("actuator") else 0.0) for i in outer}
        )
    if case == "actuator_blocked":
        prescribed.update({3 * int(i) + 1: 0.0 for i in tip})
    weights = np.zeros(mesh.n)
    ordered = tip[np.argsort(xy[tip, 1])]
    for a, b in zip(ordered[:-1], ordered[1:]):
        weights[[a, b]] += (xy[b, 1] - xy[a, 1]) / 2 / H
    rhs = np.zeros((mesh.n, 3))
    if case.startswith("sensor"):
        rhs[:, 1] = FORCE * weights
    con = MessageConstraints(mesh.n, prescribed=prescribed, electrode_groups=(outer,))
    return (
        con,
        rhs,
        dict(left=left, tip=tip, middle=middle, outer=outer, weights=weights),
        prescribed,
    )


class BimorphVCycle(BlockVCyclePreconditioner):
    @torch.no_grad()
    def __init__(self, problem, theta, scale, action, case, ny):
        self.device = scale.device
        self.dtype = scale.dtype
        self.size = len(scale)
        self.pre_smooth = self.post_smooth = 2
        self.omega = 0.7
        self.coarse_iterations = 96
        self.coarse_rtol = 0.025
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
        con = problem.constraints
        mechanical = con.free_mechanical_mask
        electrical = con.free_electrical_mask

        def positive(v):
            self.stats["positive_finest_action_calls"] += 1
            self.stats["finest_saddle_action_calls"] += 2
            return action(v * mechanical) * mechanical - action(v * electrical) * electrical

        md = problem.graph.mechanical_diagonal(theta)
        ed = problem.graph.dielectric_diagonal(theta)
        diag = scale.square() * con.contract(torch.stack((md[:, 0], md[:, 1], ed), dim=1))
        self.levels = [dict(action=positive, diagonal=diag, transfer=None)]
        fine_con = con
        fine_scale = scale
        fine_ny = ny
        transfer_checks = []
        while fine_ny >= 4 and fine_ny % 4 == 0:
            coarse_ny = fine_ny // 2
            coarse_mesh = geometry(coarse_ny)
            coarse_con, _, _, _ = boundary(coarse_mesh, case)
            coarse_graph = make_graph(coarse_mesh)
            cp = MessageEquilibrium(coarse_graph, coarse_con)
            coarse_scale = cp.scaling(theta)
            lookup = coarse_con.full_to_free.numpy()
            entries = {}
            for fi, dof in enumerate(fine_con.free_representatives.tolist()):
                node, ch = divmod(dof, 3)
                iy, ix = divmod(node, 10 * fine_ny + 1)
                xs = [(ix // 2, 1.0)] if ix % 2 == 0 else [(ix // 2, 0.5), (ix // 2 + 1, 0.5)]
                ys = [(iy // 2, 1.0)] if iy % 2 == 0 else [(iy // 2, 0.5), (iy // 2 + 1, 0.5)]
                for cx, wx in xs:
                    for cy, wy in ys:
                        cj = int(lookup[3 * (cy * (10 * coarse_ny + 1) + cx) + ch])
                        if cj >= 0:
                            entries[fi, cj] = entries.get((fi, cj), 0.0) + wx * wy
            pairs = list(entries)
            src = torch.tensor([a for a, b in pairs])
            dst = torch.tensor([b for a, b in pairs])
            w = (
                torch.tensor([entries[k] for k in pairs], dtype=torch.float64)
                * coarse_scale[dst]
                / fine_scale[src]
            )
            P = IndexedTransfer(
                src, dst, w, n_fine=fine_con.reduced_dofs, n_coarse=coarse_con.reduced_dofs
            )
            # Check actual tied-field interpolation, not only a matrix transpose identity.
            gen = torch.Generator().manual_seed(104 + fine_ny)
            cv = torch.randn(coarse_con.reduced_dofs, generator=gen, dtype=torch.float64)
            fv = torch.randn(fine_con.reduced_dofs, generator=gen, dtype=torch.float64)
            transpose = float(
                abs(torch.dot(fv, P.prolong(cv)) - torch.dot(P.restrict(fv), cv))
                / max(abs(float(torch.dot(fv, P.prolong(cv)))), 1e-30)
            )
            full = coarse_con.prolong(coarse_scale * cv).reshape(
                coarse_ny + 1, 10 * coarse_ny + 1, 3
            )
            interp = (
                torch.nn.functional.interpolate(
                    full.permute(2, 0, 1)[None],
                    size=(fine_ny + 1, 10 * fine_ny + 1),
                    mode="bilinear",
                    align_corners=True,
                )[0]
                .permute(1, 2, 0)
                .reshape(-1, 3)
            )
            expected = interp.reshape(-1)[fine_con.free_representatives] / fine_scale
            field = float(torch.norm(P.prolong(cv) - expected) / torch.norm(expected))
            if max(transpose, field) > 2e-12:
                raise RuntimeError("Transfer constraint mismatch")
            transfer_checks.append(
                dict(fine_ny=fine_ny, coarse_ny=coarse_ny, transpose=transpose, field=field)
            )
            parent = self.levels[-1]
            parent["transfer"] = P
            cd = torch.zeros(coarse_con.reduced_dofs, dtype=torch.float64).index_add(
                0, dst, w.square() * parent["diagonal"][src]
            )

            def ca(v, pa=parent["action"], mapping=P):
                return mapping.restrict(pa(mapping.prolong(v)))

            self.levels.append(dict(action=ca, diagonal=cd, transfer=None))
            fine_con, fine_scale, fine_ny = coarse_con, coarse_scale, coarse_ny
        self.metadata = dict(
            kind="generalized indexed auxiliary positive-block V-cycle",
            levels=len(self.levels),
            coarse_iterations=self.coarse_iterations,
            transfer_checks=transfer_checks,
            global_matrix_assembled=False,
        )


def metrics(state, action, bc, rhs, graph=None):
    state = np.asarray(state)
    action = np.asarray(action)
    voltage = float(state[bc["outer"], 2].mean())
    charge = float(-action[bc["outer"], 2].sum())
    qground = float(-action[bc["middle"], 2].sum())
    electrode = np.union1d(bc["outer"], bc["middle"])
    nodal_charge_l1 = float(np.abs(action[electrode, 2]).sum())
    result = dict(
        tip_mean_m=float(bc["weights"] @ state[:, 1]),
        voltage_V=voltage,
        charge_C=charge,
        tip_reaction_y_N=float(action[bc["tip"], 1].sum() - rhs[bc["tip"], 1].sum()),
        clamp_reaction_y_N=float(action[bc["left"], 1].sum()),
        charge_balance_relative=abs(charge + qground) / max(nodal_charge_l1, 1e-30),
        charge_balance_absolute_C=abs(charge + qground),
        charge_balance_reference="sum of absolute nodal charges over both electrodes",
        outer_equipotential_range_V=float(np.ptp(state[bc["outer"], 2])),
        net_force_absolute_N=float(action[:, 1].sum()),
    )
    if graph is not None:
        theta = graph.initial_theta
        ts = torch.tensor(state, dtype=torch.float64)
        noc = theta * theta.new_tensor([1, 1, 1, 1, 0, 0, 0, 1, 1])
        wm = float(graph.energy(ts * ts.new_tensor([1, 1, 0]), noc))
        wd = float(-graph.energy(ts * ts.new_tensor([0, 0, 1]), noc))
        source = 0.5 * (np.sum(state * rhs) + voltage * charge)
        result.update(
            stored_J=wm + wd,
            source_stored_J=source,
            source_energy_relative=abs(wm + wd - source) / max(abs(wm + wd), abs(source), 1e-30),
        )
    return result
