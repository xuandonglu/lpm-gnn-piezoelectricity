"""Geometry-only transfer and auxiliary V-cycle for the rotated frame."""

import numpy as np
import torch
from scipy.spatial import Delaunay, cKDTree

from .fixed_frame import DTYPE, WIDTH, geometry, inside, visibility
from .piezo_graph_multilevel import IndexedTransfer
from .piezo_message_graph import MessageConstraints
from .piezo_message_multigrid import BlockVCyclePreconditioner


def particle_interpolation(coarse, fine):
    triangulation = Delaunay(coarse)
    vertices = coarse[triangulation.simplices]
    valid = inside(vertices.mean(axis=1))
    for k in range(3):
        for row in np.flatnonzero(valid):
            if not visibility(vertices[row, k], vertices[row, (k + 1) % 3][None, :])[0]:
                valid[row] = False
    area = np.abs(np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])) / 2
    assert np.isclose(area[valid].sum(), 0.00032, rtol=1e-10)
    nearest_tree = cKDTree(coarse)
    centers = cKDTree(vertices[valid].mean(axis=1))
    valid_ids = np.flatnonzero(valid)
    source = []
    destination = []
    weights = []
    simplex = triangulation.find_simplex(fine, tol=1e-10)
    nearest_distance, nearest = nearest_tree.query(fine)
    for i, point in enumerate(fine):
        if nearest_distance[i] < 1e-12:
            ids = [int(nearest[i])]
            values = [1.0]
        else:
            candidate = int(simplex[i])
            candidates = [candidate] if candidate >= 0 and valid[candidate] else []
            if not candidates:
                _, near = centers.query(point, k=min(32, len(valid_ids)))
                candidates = list(valid_ids[np.atleast_1d(near)])
            for candidate in candidates:
                transform = triangulation.transform[candidate]
                local = transform[:2] @ (point - transform[2])
                values = np.r_[local, 1 - local.sum()]
                if min(values) >= -1e-9:
                    ids = triangulation.simplices[candidate]
                    break
            else:
                raise RuntimeError("No same-material coarse interpolation simplex")
        for j, w in zip(ids, values):
            if abs(w) > 1e-13:
                source.append(i)
                destination.append(int(j))
                weights.append(float(w))
    return np.array(source), np.array(destination), np.array(weights)


class FixedFrameVCycle(BlockVCyclePreconditioner):
    def __init__(self, problem, theta, scale, action, n, degrees):
        self.pre_smooth = 2
        self.post_smooth = 2
        self.omega = 0.7
        self.coarse_iterations = 48
        self.coarse_rtol = 0.03
        self.dtype = scale.dtype
        self.device = scale.device
        self.size = len(scale)
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
        graph, con = problem.graph, problem.constraints
        mechanical = con.free_mechanical_mask
        electrical = con.free_electrical_mask

        def positive(value):
            self.stats["positive_finest_action_calls"] += 1
            self.stats["finest_saddle_action_calls"] += 2
            return action(value * mechanical) * mechanical - action(value * electrical) * electrical

        diagonal = (
            con.contract(
                torch.cat(
                    (graph.mechanical_diagonal(theta), graph.dielectric_diagonal(theta)[:, None]), 1
                )
            )
            * scale.square()
        )
        self.levels = [
            dict(
                action=positive,
                diagonal=diagonal,
                scale=scale,
                coordinates=graph.coordinates.numpy(),
                constraints=con,
                transfer=None,
            )
        ]
        current_n = n
        while current_n > 20 and current_n % 2 == 0:
            current_n //= 2
            if current_n % 20:
                break
            fine = self.levels[-1]
            xy, vol, q, h, nbulk = geometry(current_n, degrees)
            bottom = np.flatnonzero(abs(xy[:, 1]) < 1e-11)
            top = np.flatnonzero(abs(xy[:, 1] - WIDTH) < 1e-11)
            prescribed = {3 * int(i) + k: 0.0 for i in bottom for k in range(3)}
            prescribed.update({3 * int(i) + 2: 0.0 for i in top})
            coarse_con = MessageConstraints(len(xy), prescribed)
            distance, injection = cKDTree(fine["coordinates"]).query(xy)
            assert max(distance) < 1e-12
            coarse_reps = coarse_con.free_representatives.numpy()
            fine_full = 3 * injection[coarse_reps // 3] + coarse_reps % 3
            fine_free = fine["constraints"].full_to_free.numpy()[fine_full]
            assert min(fine_free) >= 0
            coarse_scale = fine["scale"][torch.tensor(fine_free, dtype=torch.long)]
            fnode, cnode, w = particle_interpolation(xy, fine["coordinates"])
            rows = []
            cols = []
            values = []
            fmap = fine["constraints"].full_to_free.numpy()
            cmap = coarse_con.full_to_free.numpy()
            for channel in range(3):
                f = fmap[3 * fnode + channel]
                c = cmap[3 * cnode + channel]
                use = (f >= 0) & (c >= 0)
                rows.extend(f[use])
                cols.extend(c[use])
                values.extend(w[use])
            rows = torch.tensor(rows, dtype=torch.long)
            cols = torch.tensor(cols, dtype=torch.long)
            values = torch.tensor(values, dtype=DTYPE) * coarse_scale[cols] / fine["scale"][rows]
            mapping = IndexedTransfer(
                rows, cols, values, n_fine=len(fine["scale"]), n_coarse=len(coarse_scale)
            )
            fine["transfer"] = mapping
            coarse_diagonal = torch.zeros_like(coarse_scale).index_add(
                0, cols, values.square() * fine["diagonal"][rows]
            )

            def coarse_action(value, parent=fine["action"], transfer=mapping):
                self.stats["transfer_prolong_calls"] += 1
                lifted = transfer.prolong(value)
                response = parent(lifted)
                self.stats["transfer_restrict_calls"] += 1
                return transfer.restrict(response)

            self.levels.append(
                dict(
                    action=coarse_action,
                    diagonal=coarse_diagonal,
                    scale=coarse_scale,
                    coordinates=xy,
                    constraints=coarse_con,
                    transfer=None,
                )
            )
        self.metadata = dict(
            method="positive-block indexed V-cycle",
            transfer="same-material barycentric interpolation of particle values; geometry only",
            free_sizes=[len(level["scale"]) for level in self.levels],
            mechanical_and_dielectric_blocks="unchanged complete graph actions",
            global_material_matrix_constructed=False,
            reference_solution_used=False,
        )
