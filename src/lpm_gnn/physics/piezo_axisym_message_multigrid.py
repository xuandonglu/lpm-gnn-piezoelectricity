"""Indexed positive-block/inertia V-cycle for the uniform axisymmetric graph.

Conventional flexible preconditioning, not learning or a new solver. No global
K/M/P/R or direct coarse solve. Coarse actions still use the finest graph.
The preconditioner drops piezo cross blocks, makes the electric block positive,
and ADDS inertia; the outer physical action retains coupling and minus inertia.
Only bare uniform r,z grids with axis ur fixed and top/bottom phi fixed apply.
"""

import math
import time

import torch


def norm_rows(x):
    scale = x.abs().amax(dim=1)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale * torch.sqrt((x / safe[:, None]).abs().square().sum(dim=1))


class BatchedTransfer:
    def __init__(self, src, dst, weights, nf, nc, fine_scale, coarse_scale):
        self.src, self.dst = src, dst
        self.nf, self.nc = nf, nc
        self.weights = weights[None, :] * coarse_scale[:, dst] / fine_scale[:, src]

    def prolong(self, z):
        return z.new_zeros((z.shape[0], self.nf)).index_add(
            1, self.src, self.weights * z[:, self.dst]
        )

    def restrict(self, x):
        return x.new_zeros((x.shape[0], self.nc)).index_add(
            1, self.dst, self.weights * x[:, self.src]
        )


class AxisymmetricBlockVCycle:
    @torch.no_grad()
    def __init__(
        self,
        problem,
        real,
        imag,
        frequencies,
        scale,
        scaled_action,
        pre_smooth=2,
        post_smooth=2,
        coarse_iterations=40,
        coarse_rtol=0.1,
        omega=0.7,
    ):
        start = time.perf_counter()
        self.problem = problem
        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth
        self.coarse_iterations, self.coarse_rtol, self.omega = coarse_iterations, coarse_rtol, omega
        self.stats = dict(
            calls=0,
            finest_positive_actions=0,
            prolong_calls=0,
            restrict_calls=0,
            coarse_steps=0,
            smoothing_steps=0,
            coarse_budget_exhaustions=0,
            apply_seconds=0.0,
        )
        graph = problem.graph
        batch = frequencies.numel()
        self.batch = batch
        no_e = real.detach().clone()
        no_e[..., 5:8] = 0
        zero_imag = torch.zeros(7, dtype=torch.float64, device=real.device)
        zero_frequency = torch.zeros_like(frequencies)
        omega2 = (2 * math.pi * frequencies).square()

        def positive(y):
            self.stats["finest_positive_actions"] += 1
            state = problem.prolong(scale * y)
            signed = torch.stack((state[:, :, 0], state[:, :, 1], -state[:, :, 2]), dim=2)
            output = graph.action(signed, no_e, zero_imag, zero_frequency)
            mass = omega2[:, None] * graph.node_mass_kg[None, :]
            output = torch.stack(
                (
                    output[:, :, 0] + mass * state[:, :, 0],
                    output[:, :, 1] + mass * state[:, :, 1],
                    output[:, :, 2],
                ),
                dim=2,
            )
            return scale * problem.contract(output)

        # Exact fine positive diagonal in the same scaled coordinates.
        diagonal = problem.static_diagonal(real).abs().reshape(-1)[problem.free_indices][None, :]
        mass_full = torch.stack(
            (graph.node_mass_kg, graph.node_mass_kg, torch.zeros_like(graph.node_mass_kg)), dim=1
        )
        mass_free = mass_full.reshape(-1)[problem.free_indices]
        diagonal = scale.square() * (diagonal + omega2[:, None] * mass_free[None, :])
        nr, nz = graph.nr, graph.nz
        free_ids = problem.free_indices.detach().cpu().tolist()
        dofs = [(int((i // 3) // (nz + 1)), int((i // 3) % (nz + 1)), int(i % 3)) for i in free_ids]
        self.levels = [
            dict(
                action=positive,
                diagonal=diagonal,
                dofs=dofs,
                nr=nr,
                nz=nz,
                scale=scale,
                transfer=None,
            )
        ]
        while nr % 2 == 0 and nz % 2 == 0 and nz >= 4:
            fine = self.levels[-1]
            nr //= 2
            nz //= 2
            coarse_dofs = [
                (i, j, c)
                for i in range(nr + 1)
                for j in range(nz + 1)
                for c in range(3)
                if not (c == 0 and i == 0) and not (c == 2 and j in (0, nz))
            ]
            lookup = {d: i for i, d in enumerate(fine["dofs"])}
            coarse_lookup = {d: i for i, d in enumerate(coarse_dofs)}
            injection = torch.tensor(
                [lookup[(2 * i, 2 * j, c)] for i, j, c in coarse_dofs],
                dtype=torch.long,
                device=real.device,
            )
            coarse_scale = fine["scale"][:, injection]
            src, dst, weight = [], [], []
            for k, (i, j, c) in enumerate(fine["dofs"]):
                xs = [(i // 2, 1.0)] if i % 2 == 0 else [(i // 2, 0.5), (i // 2 + 1, 0.5)]
                zs = [(j // 2, 1.0)] if j % 2 == 0 else [(j // 2, 0.5), (j // 2 + 1, 0.5)]
                for ci, wi in xs:
                    for cj, wj in zs:
                        t = coarse_lookup.get((ci, cj, c))
                        if t is not None:
                            src.append(k)
                            dst.append(t)
                            weight.append(wi * wj)
            src = torch.tensor(src, dtype=torch.long, device=real.device)
            dst = torch.tensor(dst, dtype=torch.long, device=real.device)
            weight = torch.tensor(weight, dtype=torch.float64, device=real.device)
            transfer = BatchedTransfer(
                src, dst, weight, len(fine["dofs"]), len(coarse_dofs), fine["scale"], coarse_scale
            )
            fine["transfer"] = transfer

            def coarse_action(x, parent=fine["action"], mapping=transfer):
                self.stats["prolong_calls"] += 1
                out = parent(mapping.prolong(x))
                self.stats["restrict_calls"] += 1
                return mapping.restrict(out)

            diagonal = diagonal.new_zeros((batch, len(coarse_dofs))).index_add(
                1, dst, transfer.weights.square() * fine["diagonal"][:, src]
            )
            if not bool(torch.all(diagonal > 0)):
                raise RuntimeError("Nonpositive diagonal proxy")
            self.levels.append(
                dict(
                    action=coarse_action,
                    diagonal=diagonal,
                    dofs=coarse_dofs,
                    nr=nr,
                    nz=nz,
                    scale=coarse_scale,
                    transfer=None,
                )
            )
        self.metadata = dict(
            method="axisymmetric indexed positive-block plus inertia V-cycle",
            hierarchy=[dict(nr=l["nr"], nz=l["nz"], free_dofs=len(l["dofs"])) for l in self.levels],
            coupled_outer_action_unchanged=True,
            coarse_action_uses_finest_graph=True,
            coarse_diagonal="positive proxy, not exact Galerkin diagonal",
            only_bare_uniform_r_z=True,
            no_global_matrices=True,
            setup_seconds=time.perf_counter() - start,
        )

    def smooth(self, level, rhs, x, r, count):
        for _ in range(count):
            z = r / level["diagonal"]
            rz = (r.conj() * z).sum(dim=1).real
            az = level["action"](z)
            curvature = (z.conj() * az).sum(dim=1).real
            live = rz > 0
            if bool((live & (curvature <= 0)).any()):
                raise RuntimeError("Positive block curvature failed")
            safe = torch.where(live, curvature, torch.ones_like(curvature))
            alpha = torch.where(live, torch.clamp(rz / safe, max=self.omega), torch.zeros_like(rz))
            x = x + alpha[:, None] * z
            r = r - alpha[:, None] * az
            self.stats["smoothing_steps"] += 1
        return x, r

    def coarse(self, level, rhs):
        x = torch.zeros_like(rhs)
        r = rhs.clone()
        initial = norm_rows(r)
        live = initial > 0
        z = r / level["diagonal"]
        direction = z.clone()
        rz = (r.conj() * z).sum(dim=1).real
        for _ in range(self.coarse_iterations):
            if not bool(live.any()):
                return x
            ad = level["action"](direction)
            curvature = (direction.conj() * ad).sum(dim=1).real
            if bool((live & ((curvature <= 0) | (rz <= 0))).any()):
                raise RuntimeError("Coarse curvature failed")
            safe = torch.where(live, curvature, torch.ones_like(curvature))
            alpha = torch.where(live, rz / safe, torch.zeros_like(rz))
            x = x + alpha[:, None] * direction
            r = r - alpha[:, None] * ad
            self.stats["coarse_steps"] += 1
            live = live & (norm_rows(r) > self.coarse_rtol * initial)
            z = r / level["diagonal"]
            next_rz = (r.conj() * z).sum(dim=1).real
            safe_rz = torch.where(live, rz, torch.ones_like(rz))
            beta = torch.where(live, next_rz / safe_rz, torch.zeros_like(rz))
            direction = torch.where(
                live[:, None], z + beta[:, None] * direction, torch.zeros_like(z)
            )
            rz = next_rz
        self.stats["coarse_budget_exhaustions"] += int(live.sum())
        return x

    def cycle(self, index, rhs):
        level = self.levels[index]
        if level["transfer"] is None:
            return self.coarse(level, rhs)
        x = torch.zeros_like(rhs)
        r = rhs.clone()
        x, r = self.smooth(level, rhs, x, r, self.pre_smooth)
        self.stats["restrict_calls"] += 1
        coarse_rhs = level["transfer"].restrict(r)
        coarse_x = self.cycle(index + 1, coarse_rhs)
        self.stats["prolong_calls"] += 1
        x = x + level["transfer"].prolong(coarse_x)
        r = rhs - level["action"](x)
        x, r = self.smooth(level, rhs, x, r, self.post_smooth)
        return x

    @torch.no_grad()
    def __call__(self, value):
        tick = time.perf_counter()
        self.stats["calls"] += 1
        result = self.cycle(0, value)
        self.stats["apply_seconds"] += time.perf_counter() - tick
        return result


def make_axisym_vcycle_factory(**options):
    def factory(problem, real, imag, frequencies, scale, action):
        instance = AxisymmetricBlockVCycle(
            problem, real, imag, frequencies, scale, action, **options
        )
        problem.last_preconditioner = instance
        return instance

    return factory
