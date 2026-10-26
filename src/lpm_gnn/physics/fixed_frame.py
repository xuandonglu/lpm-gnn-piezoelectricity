"""Rotated material lattice inside a fixed double-window domain.

Boundary differences use same-material-side quadratic reconstruction. Particle
weights are exact clipped Voronoi areas. Sparse expansion is a verification
backend for this graph energy, separate from the matrix-free graph iteration.
"""

import json
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import splu
from scipy.spatial import Voronoi, cKDTree

from lpm_gnn.physics.piezo_frame_message_problem import (
    frame_constraints,
    frame_pad_weights,
    make_frame_geometry,
)
from lpm_gnn.physics.piezo_message_geometry import MessageGeometry
from lpm_gnn.physics.piezo_message_graph import (
    _DIRECTIONS,
    _OFFSETS,
    DEFAULT_PHYSICAL,
    DEFAULT_SCALES,
    MessageConstraints,
    PiezoMessageGraph,
)
from lpm_gnn.physics.rotated_lattice import RotatedLatticeGraph
from lpm_gnn.reference.planar_fem import PiezoMaterial, pzt4, q4_reference

DTYPE = torch.float64
WIDTH = 0.02
THICKNESS = 0.001
HOLES = ((0.005, 0.009, 0.005, 0.015), (0.011, 0.015, 0.005, 0.015))


def rotate_reference(material, angle):
    q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    index = np.array([[0, 2], [2, 1]])
    c4 = material.c[index[:, :, None, None], index[None, None, :, :]]
    e3 = material.e[:, index]
    cr = np.einsum("ia,jb,kc,ld,abcd->ijkl", q, q, q, q, c4)
    er = np.einsum("ia,jb,kc,abc->ijk", q, q, q, e3)
    pairs = [(0, 0), (1, 1), (0, 1)]
    c = np.array([[cr[i, j, k, l] for k, l in pairs] for i, j in pairs])
    e = np.array([[er[a, i, j] for i, j in pairs] for a in range(2)])
    return PiezoMaterial(c, e, q @ material.kappa @ q.T, "full-tensor rotated reference")


def relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def inside(points, strict=False):
    x, y = np.asarray(points).T
    eps = 1e-12 if strict else -1e-12
    valid = (x > eps) & (x < WIDTH - eps) & (y > eps) & (y < WIDTH - eps)
    for a, b, c, d in HOLES:
        valid &= ~((x > a + 1e-12) & (x < b - 1e-12) & (y > c + 1e-12) & (y < d - 1e-12))
    return valid


def visibility(center, points):
    # Reject a segment if an open portion lies strictly inside either hole.
    points = np.asarray(points)
    delta = points - center
    keep = np.ones(len(points), bool)
    for a, b, c, d in HOLES:
        lo = np.zeros(len(points))
        hi = np.ones(len(points))
        possible = np.ones(len(points), bool)
        for axis, left, right in ((0, a, b), (1, c, d)):
            nonzero = np.abs(delta[:, axis]) > 1e-15
            first = np.full(len(points), -np.inf)
            second = np.full(len(points), np.inf)
            t1 = (left + 1e-13 - center[axis]) / np.where(nonzero, delta[:, axis], 1.0)
            t2 = (right - 1e-13 - center[axis]) / np.where(nonzero, delta[:, axis], 1.0)
            first[nonzero] = np.minimum(t1, t2)[nonzero]
            second[nonzero] = np.maximum(t1, t2)[nonzero]
            possible &= nonzero | ((center[axis] > left + 1e-13) & (center[axis] < right - 1e-13))
            lo = np.maximum(lo, first)
            hi = np.minimum(hi, second)
        keep &= ~(possible & (hi > lo + 1e-12))
    return keep


def boundary_distance(points):
    x, y = np.asarray(points).T
    distance = np.minimum.reduce((x, y, WIDTH - x, WIDTH - y))
    for a, b, c, d in HOLES:
        dx = np.maximum.reduce((a - x, np.zeros_like(x), x - b))
        dy = np.maximum.reduce((c - y, np.zeros_like(y), y - d))
        distance = np.minimum(distance, np.hypot(dx, dy))
    return distance


def boundary_sites(h):
    points = []
    for a, b, c, d in ((0.0, WIDTH, 0.0, WIDTH),) + HOLES:
        for x0, y0, x1, y1 in ((a, c, b, c), (b, c, b, d), (b, d, a, d), (a, d, a, c)):
            count = int(np.ceil(np.hypot(x1 - x0, y1 - y0) / h - 1e-9))
            points.extend(np.linspace((x0, y0), (x1, y1), count + 1))
    points = np.unique(np.round(np.asarray(points), 12), axis=0)
    return points


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly)
    return abs(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))) / 2


def clip_rect(poly, bounds):
    poly = list(map(np.asarray, poly))
    a, b, c, d = bounds
    for axis, value, sign in ((0, a, 1), (0, b, -1), (1, c, 1), (1, d, -1)):
        if not poly:
            return []
        output = []
        previous = poly[-1]
        before = sign * (previous[axis] - value) >= -1e-15
        for current in poly:
            after = sign * (current[axis] - value) >= -1e-15
            if after != before:
                fraction = (value - previous[axis]) / (current[axis] - previous[axis])
                output.append(previous + fraction * (current - previous))
            if after:
                output.append(current)
            previous, before = current, after
        poly = output
    return poly


def exact_volumes(xy):
    guard = np.array([[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]], float)
    voronoi = Voronoi(np.vstack((xy, guard)))
    volumes = []
    for i in range(len(xy)):
        region = voronoi.regions[voronoi.point_region[i]]
        if -1 in region:
            raise RuntimeError("Unbounded physical Voronoi cell")
        poly = voronoi.vertices[region]
        clipped = clip_rect(poly, (0.0, WIDTH, 0.0, WIDTH))
        area = polygon_area(clipped)
        for hole in HOLES:
            area -= polygon_area(clip_rect(clipped, hole))
        volumes.append(area)
    volumes = np.asarray(volumes)
    if np.any(volumes <= 0) or not np.isclose(volumes.sum(), 0.00032, rtol=1e-10):
        raise RuntimeError("Exact physical volume coverage failed")
    return volumes


def geometry(n, degrees, offset=(0.0, 0.0)):
    h = WIDTH / n
    angle = np.deg2rad(degrees)
    q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    extent = int(np.ceil(n / np.sqrt(2))) + 3
    lattice_index = np.array(
        [(i, j) for j in range(-extent, extent + 1) for i in range(-extent, extent + 1)]
    )
    bulk = (lattice_index + offset) * h @ q.T + WIDTH / 2
    bulk = bulk[inside(bulk, strict=True) & (boundary_distance(bulk) > 0.3 * h)]
    boundary = boundary_sites(h)
    xy = np.vstack((bulk, boundary))
    if len(np.unique(np.round(xy, 12), axis=0)) != len(xy):
        raise RuntimeError("Duplicate particle sites")
    return xy, exact_volumes(xy), q, h, len(bulk)


def basis(r, degree):
    x, y = r.T
    columns = [x, y]
    if degree == 2:
        columns += [x * x, x * y, y * y]
    return np.column_stack(columns)


def graph_geometry(n, degrees, *, degree=2, offset=(0.0, 0.0)):
    xy, volumes, q, h, nbulk = geometry(n, degrees, offset)
    tree = cKDTree(xy)
    nodes = []
    bonds = []
    coefficients = []
    star_nodes = []
    star_node_stars = []
    axial = []
    diagonal = []
    virtual = 0
    maximum_condition = 0.0
    maximum_radius = 0.0
    for i, center in enumerate(xy):
        targets = center + h * _OFFSETS @ q.T
        distances, nearest = tree.query(targets)
        real = (distances < h * 1e-7) & visibility(center, xy[nearest])
        missing = np.flatnonzero(~real)
        recon_nodes = None
        if len(missing):
            virtual += len(missing)
            for radius in (2.4, 3.2, 4.2, 5.5):
                neighbors = np.asarray(tree.query_ball_point(center, radius * h), dtype=int)
                neighbors = neighbors[(neighbors != i) & visibility(center, xy[neighbors])]
                relative_position = (xy[neighbors] - center) @ q / h
                p = basis(relative_position, degree)
                weights = (1 + np.sum(relative_position**2, axis=1)) ** -2
                weighted = p * np.sqrt(weights[:, None])
                if (
                    len(neighbors) >= p.shape[1] + 1
                    and np.linalg.matrix_rank(weighted) == p.shape[1]
                ):
                    condition = np.linalg.cond(weighted)
                    if condition < 1000:
                        break
            else:
                raise RuntimeError(f"Poor boundary reconstruction at particle{i}")
            maximum_condition = max(maximum_condition, float(condition))
            maximum_radius = max(maximum_radius, radius)
            weights = np.linalg.pinv(weighted, rcond=1e-13) * np.sqrt(weights)[None, :]
            reconstructed = basis(_OFFSETS[missing], degree) @ weights
            recon_nodes = neighbors
            recon_lookup = {int(kind): row for row, kind in enumerate(missing)}
        accum_a = {}
        accum_d = {}
        for kind in range(8):
            bi = 8 * i + kind
            if real[kind]:
                ids = [i, int(nearest[kind])]
                values = [-1.0, 1.0]
            else:
                values = reconstructed[recon_lookup[kind]]
                ids = np.r_[i, recon_nodes]
                values = np.r_[-values.sum(), values]
            target_acc = accum_a if kind < 4 else accum_d
            direction = _DIRECTIONS[kind] @ q.T
            for node, value in zip(ids, values):
                if abs(value) < 1e-14:
                    continue
                nodes.append(int(node))
                bonds.append(bi)
                coefficients.append(float(value))
                target_acc[int(node)] = target_acc.get(int(node), np.zeros(2)) + value * direction
        for node in sorted(set(accum_a) | set(accum_d)):
            star_nodes.append(node)
            star_node_stars.append(i)
            axial.append(accum_a.get(node, np.zeros(2)))
            diagonal.append(accum_d.get(node, np.zeros(2)))
    data = dict(
        coordinates=xy,
        material_coordinates=(xy - WIDTH / 2) @ q,
        volumes=volumes,
        rotation=q,
        poling_direction=q[:, 1],
        coefficient_node=np.array(nodes),
        coefficient_bond=np.array(bonds),
        coefficient_value=np.array(coefficients),
        bond_star=np.repeat(np.arange(len(xy)), 8),
        bond_direction=np.tile(_DIRECTIONS @ q.T, (len(xy), 1)),
        bond_kind=np.tile(np.arange(8), len(xy)),
        star_scale=THICKNESS * volumes / h**2,
        star_center=np.arange(len(xy)),
        star_cell=np.full(len(xy), -1),
        star_node=np.array(star_nodes),
        star_node_star=np.array(star_node_stars),
        axial_node_coeff=np.array(axial),
        diagonal_node_coeff=np.array(diagonal),
    )
    metadata = dict(
        particles=len(xy),
        bulk_lattice_particles=nbulk,
        exact_boundary_particles=len(xy) - nbulk,
        virtual_directions=virtual,
        polynomial_completion_degree=degree,
        max_weighted_reconstruction_condition=maximum_condition,
        max_reconstruction_radius_over_h=maximum_radius,
        volume_m2=float(volumes.sum()),
        fixed_exact_geometry=True,
        boundary_completion="same-side weighted polynomial differences",
        material_axes_rotated_degrees=degrees,
        volume_rule="exact clipped Voronoi cells",
    )
    return data, metadata, h


class FixedFrameGraph(RotatedLatticeGraph):
    def __init__(self, n, degrees, *, degree=2, offset=(0.0, 0.0)):
        torch.nn.Module.__init__(self)
        data, metadata, h = graph_geometry(n, degrees, degree=degree, offset=offset)
        self.n = len(data["coordinates"])
        self.ndof = 3 * self.n
        self.nstar = self.n
        self.nbond = 8 * self.n
        self.spacing = h
        self.thickness = THICKNESS
        for name, value in data.items():
            self.register_buffer(
                name,
                torch.as_tensor(
                    value, dtype=torch.long if np.issubdtype(value.dtype, np.integer) else DTYPE
                ),
            )
        self.register_buffer("parameter_scales", torch.tensor(DEFAULT_SCALES, dtype=DTYPE))
        self.register_buffer(
            "initial_theta", torch.tensor(DEFAULT_PHYSICAL / DEFAULT_SCALES, dtype=DTYPE)
        )
        kind = data["bond_kind"]
        for name, value in dict(
            axial=kind < 4,
            diagonal=kind >= 4,
            horizontal=kind < 2,
            vertical=(kind >= 2) & (kind < 4),
        ).items():
            self.register_buffer(name, torch.tensor(value, dtype=DTYPE))
        self.metadata = metadata


def assemble_graph(graph):
    """Exact sparse expansion of graph.energy for bounded verification only."""
    node = graph.coefficient_node.numpy()
    bond = graph.coefficient_bond.numpy()
    value = graph.coefficient_value.numpy()
    direction = graph.bond_direction.numpy()[bond]
    e = sp.coo_matrix(
        (
            np.r_[value * direction[:, 0], value * direction[:, 1]],
            (np.r_[bond, bond], np.r_[3 * node, 3 * node + 1]),
        ),
        shape=(graph.nbond, graph.ndof),
    ).tocsr()
    d = sp.coo_matrix((value, (bond, 3 * node + 2)), shape=(graph.nbond, graph.ndof)).tocsr()
    star = graph.bond_star.numpy()
    kind = graph.bond_kind.numpy()
    select_a = sp.coo_matrix(
        ((kind < 4).astype(float), (star, np.arange(graph.nbond))), shape=(graph.n, graph.nbond)
    ).tocsr()
    select_d = sp.coo_matrix(
        ((kind >= 4).astype(float), (star, np.arange(graph.nbond))), shape=(graph.n, graph.nbond)
    ).tocsr()
    a = graph.bond_direction.numpy() @ graph.poling_direction.numpy()
    sa = select_a @ e
    sd = select_d @ e
    p = select_a @ sp.diags(a) @ d
    spring, tv, eta, alpha, beta = graph._material_coefficients(graph.initial_theta)
    spring, eta, beta = (v.numpy() for v in (spring, eta, beta))
    scale = graph.star_scale.numpy()
    bs = scale[star]
    cross = e.T @ sp.diags(eta * a * bs) @ d + float(alpha) * sa.T @ sp.diags(scale) @ p
    matrix = (
        e.T @ sp.diags(spring * bs) @ e
        + float(tv) * (sa.T @ sp.diags(scale) @ sa + sd.T @ sp.diags(scale) @ sd)
        - d.T @ sp.diags(beta * bs) @ d
        + cross
        + cross.T
    ).tocsr()
    matrix.eliminate_zeros()
    return matrix


def fixed_boundary_data(graph):
    xy = graph.coordinates.numpy()
    eps = 1e-11
    bottom = np.flatnonzero(abs(xy[:, 1]) < eps)
    top = np.flatnonzero(abs(xy[:, 1] - WIDTH) < eps)
    top = top[np.argsort(xy[top, 0])]
    pad = np.zeros(graph.n)
    for i, j in zip(top[:-1], top[1:]):
        a, b = xy[[i, j], 0]
        left, right = max(a, 0.008), min(b, 0.012)
        if right > left:
            t0, t1 = (left - a) / (b - a), (right - a) / (b - a)
            pad[i] += (b - a) * ((t1 - t0) - (t1 * t1 - t0 * t0) / 2)
            pad[j] += (b - a) * (t1 * t1 - t0 * t0) / 2
    assert np.isclose(pad.sum(), 0.004, rtol=1e-11)
    return bottom, top, pad / 0.004


def sparse_cases(matrix, xy, bottom, top, pad):
    ndof = len(xy) * 3
    fixed = np.unique(np.r_[(3 * bottom[:, None] + np.arange(3)).ravel(), 3 * top + 2])
    free = np.setdiff1d(np.arange(ndof), fixed)
    a = matrix[free][:, free].tocsc()
    scale = 1 / np.sqrt(abs(a.diagonal()))
    factor = splu(sp.diags(scale) @ a @ sp.diags(scale))
    states = {}
    outputs = {}
    for name, force, voltage in [("force", 0.01, 0.0), ("voltage", 0.0, 1.0)]:
        rhs = np.zeros(ndof)
        rhs[1::3] = force * pad
        state = np.zeros(ndof)
        state[3 * top + 2] = voltage
        br = rhs[free] - matrix[free][:, fixed] @ state[fixed]
        state[free] = scale * factor.solve(scale * br)
        residual = matrix @ state - rhs
        scaled_relative = float(
            np.linalg.norm(scale * residual[free]) / max(np.linalg.norm(scale * br), 1e-300)
        )
        field = state.reshape(-1, 3)
        reaction = (matrix @ state).reshape(-1, 3)
        states[name] = field
        outputs[name] = dict(
            scaled_relative_residual=scaled_relative,
            pad_displacement_m=float(pad @ field[:, 1]),
            top_free_charge_C=float(-reaction[top, 2].sum()),
            charge_balance_C=float(reaction[top, 2].sum() + reaction[bottom, 2].sum()),
            force_balance_N=float(reaction[bottom, :2].sum(axis=0)[1] + force),
        )
    return states, outputs


def fem_reference(n, degrees, out):
    cache = out / f"fem_n{n}_a{degrees:g}.npz"
    meta = cache.with_suffix(".json")
    if cache.exists():
        raw = np.load(cache)
        return {name: raw[name].copy() for name in raw.files}, json.loads(meta.read_text())
    mesh = make_frame_geometry(n)
    mat = rotate_reference(pzt4(), np.deg2rad(degrees))
    # One independent continuum Q4 element integrated with 2x2 Gaussian rule.
    # All reference cells share its geometry. Reuse is only within this FEM.
    example = MessageGeometry.rectangle(1, width=mesh.spacing)
    klocal = q4_reference(example, mat, THICKNESS).toarray()
    # rectangle(1) ordering: lower-left, lower-right, upper-left, upper-right;
    # reorder into the connectivity order used by make_frame_geometry.
    ids = (3 * np.array([0, 1, 3, 2])[:, None] + np.arange(3)).ravel()
    klocal = klocal[np.ix_(ids, ids)]
    dofs = (3 * mesh.cells[:, :, None] + np.arange(3)).reshape(-1, 12)
    matrix = sp.coo_matrix(
        (
            np.tile(klocal.ravel(), len(dofs)),
            (np.repeat(dofs, 12, axis=1).ravel(), np.tile(dofs, (1, 12)).ravel()),
        ),
        shape=(3 * mesh.n, 3 * mesh.n),
    ).tocsr()
    bottom = np.flatnonzero(abs(mesh.coordinates[:, 1]) < 1e-12)
    top = np.flatnonzero(abs(mesh.coordinates[:, 1] - WIDTH) < 1e-12)
    states, outputs = sparse_cases(matrix, mesh.coordinates, bottom, top, frame_pad_weights(mesh))
    arrays = dict(coordinates=mesh.coordinates, cells=mesh.cells, **states)
    np.savez_compressed(cache, **arrays)
    meta.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return arrays, outputs


def interpolate_reference(reference, points):
    xy = reference["coordinates"]
    cells = reference["cells"]
    h = xy[cells[0, 1], 0] - xy[cells[0, 0], 0]
    centers = xy[cells].mean(axis=1)
    lookup = {
        tuple(np.rint(center / h - 0.5).astype(int)): conn for center, conn in zip(centers, cells)
    }
    result = {name: np.empty((len(points), 3)) for name in ("force", "voltage")}
    for i, point in enumerate(points):
        ix, iy = np.floor(point / h + 1e-10).astype(int)
        for key in ((ix, iy), (ix - 1, iy), (ix, iy - 1), (ix - 1, iy - 1)):
            conn = lookup.get(key)
            if conn is None:
                continue
            ll = xy[conn[0]]
            tx, ty = (point - ll) / h
            if min(tx, ty) >= -1e-8 and max(tx, ty) <= 1 + 1e-8:
                weight = np.array([(1 - tx) * (1 - ty), tx * (1 - ty), tx * ty, (1 - tx) * ty])
                for name in result:
                    result[name][i] = weight @ reference[name][conn]
                break
        else:
            raise RuntimeError(f"No physical reference cell at{point}")
    return result


def run_case(n, degrees, out, reference_n, degree=2, offset=(0.0, 0.0)):
    import time

    tick = time.perf_counter()
    graph = FixedFrameGraph(n, degrees, degree=degree, offset=offset)
    print("geometry", n, degrees, graph.metadata, flush=True)
    matrix = assemble_graph(graph)
    probe = np.random.default_rng(42).normal(size=(graph.n, 3)) * [1e-8, 1e-8, 1.0]
    action = graph.action(torch.tensor(probe, dtype=DTYPE), graph.initial_theta).numpy()
    assembled = (matrix @ probe.ravel()).reshape(-1, 3)
    parity = {
        name: relative(action[:, s], assembled[:, s])
        for name, s in [("mechanical", slice(0, 2)), ("electrical", slice(2, 3))]
    }
    assert max(parity.values()) < 1e-10
    bottom, top, pad = fixed_boundary_data(graph)
    states, ports = sparse_cases(matrix, graph.coordinates.numpy(), bottom, top, pad)
    ref, refports = fem_reference(reference_n, degrees, out)
    projected = interpolate_reference(ref, graph.coordinates.numpy())
    weights = np.sqrt(graph.volumes.numpy())[:, None]
    errors = {}
    for name in ("force", "voltage"):
        errors[name] = {
            field: relative(
                weights * states[name][:, section], weights * projected[name][:, section]
            )
            for field, section in [("displacement", slice(0, 2)), ("potential", slice(2, 3))]
        }
        for label in ("pad_displacement_m", "top_free_charge_C"):
            errors[name][label + "_relative_error"] = abs(
                ports[name][label] / refports[name][label] - 1
            )
    suffix = f"n{n}_a{degrees:g}_p{degree}"
    if offset != (0.0, 0.0):
        suffix += f"_shift{offset[0]:g}_{offset[1]:g}"
    arrays = dict(
        coordinates=graph.coordinates.numpy(),
        volumes=graph.volumes.numpy(),
        bulk_lattice_particles=np.array(graph.metadata["bulk_lattice_particles"]),
        poling_direction=graph.poling_direction.numpy(),
        pad=pad,
        bottom=bottom,
        top=top,
        **{f"{name}_state": value for name, value in states.items()},
        **{f"{name}_FEM": value for name, value in projected.items()},
    )
    np.savez_compressed(out / f"{suffix}.npz", **arrays)
    row = dict(
        n=n,
        degrees=degrees,
        metadata=graph.metadata,
        reference_n=reference_n,
        errors=errors,
        ports=ports,
        reference_ports=refports,
        graph_sparse_action_parity=parity,
        solver="scaled sparse LU of exactly expanded graph energy; research verification only",
        elapsed_seconds=time.perf_counter() - tick,
        nonzeros=matrix.nnz,
        boundary_unknowns="exact physical boundary particle sites",
        material_frame_coefficients="unchanged PZT4; no fitted weights or FEM force correction",
    )
    (out / f"{suffix}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print("RESULT", n, degrees, errors, flush=True)
    return row
