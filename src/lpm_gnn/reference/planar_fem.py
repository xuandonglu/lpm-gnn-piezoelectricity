"""Independent Gaussian-Q4 continuum reference, using SI units.

The reference uses continuum constitutive tensors and Gaussian quadrature;
it does not use the particle enthalpy or graph residual.
"""

import time
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import MatrixRankWarning, spsolve

EPS0 = 8.8541878128e-12


@dataclass
class PiezoMaterial:
    c: np.ndarray
    e: np.ndarray
    kappa: np.ndarray
    label: str = "specified continuum coefficients"

    def __post_init__(self):
        self.c = np.asarray(self.c, float).reshape(3, 3)
        self.e = np.asarray(self.e, float).reshape(2, 3)
        self.kappa = np.asarray(self.kappa, float).reshape(2, 2)
        for a in (self.c, self.e, self.kappa):
            if not np.all(np.isfinite(a)):
                raise ValueError("Nonfinite constitutive coefficient")
        for a in (self.c, self.kappa):
            if not np.allclose(a, a.T, rtol=1e-12, atol=0):
                raise ValueError("C and kappa must be symmetric")
            if np.linalg.eigvalsh(a).min() <= 0:
                raise ValueError("C and fixed-strain kappa must be positive definite")

    @property
    def enthalpy(self):
        return np.block([[self.c, -self.e.T], [-self.e, -self.kappa]])

    def rotated(self, angle):
        """Rotate material axes; engineering shear transformed with its factor 2."""
        co, si = np.cos(angle), np.sin(angle)
        q = np.array([[co, -si], [si, co]])
        transform = np.empty((3, 3))
        for k, v in enumerate(np.eye(3)):
            strain = np.array([[v[0], v[2] / 2], [v[2] / 2, v[1]]])
            local = q.T @ strain @ q
            transform[:, k] = [local[0, 0], local[1, 1], 2 * local[0, 1]]
        return PiezoMaterial(
            transform.T @ self.c @ transform,
            q @ self.e @ transform,
            q @ self.kappa @ q.T,
            self.label + "; rotated",
        )

    def response(self, strain, electric_field):
        return (
            self.c @ strain - self.e.T @ electric_field,
            self.e @ strain + self.kappa @ electric_field,
        )


def pzt4():
    """Literature example, not specimen-calibrated coefficients.

    In-plane axes are material (1,3), with plane strain in axis 2.
    Source: 10.11993/j.issn.2096-3920.2022-0084, Table 1; epsilon^S/epsilon0.
    Do not mix this fixed-strain permittivity with a free-stress data sheet.
    """
    return PiezoMaterial(
        1e9 * np.array([[139.0, 74.3, 0.0], [74.3, 115.0, 0.0], [0.0, 0.0, 25.6]]),
        np.array([[0.0, 0.0, 12.7], [-5.2, 15.1, 0.0]]),
        EPS0 * np.diag([730.0, 635.0]),
        "literature PZT-4 plane strain (1,3)",
    )


def quad_gradients(xy, xi, eta):
    dref = 0.25 * np.array(
        [[-(1 - eta), -(1 - xi)], [1 - eta, -(1 + xi)], [1 + eta, 1 + xi], [-(1 + eta), 1 - xi]]
    )
    jac = xy.T @ dref
    det = np.linalg.det(jac)
    if det <= 0:
        raise ValueError("Inverted Q4 cell")
    return dref @ np.linalg.inv(jac), det


def field_map(grad):
    b = np.zeros((5, 3 * len(grad)))
    b[0, 0::3] = grad[:, 0]
    b[1, 1::3] = grad[:, 1]
    b[2, 0::3] = grad[:, 1]
    b[2, 1::3] = grad[:, 0]
    b[3, 2::3] = -grad[:, 0]
    b[4, 2::3] = -grad[:, 1]
    return b


def assemble_local(blocks, ndof):
    rr, cc, vv = [], [], []
    for ids, block in blocks:
        row, col = np.meshgrid(ids, ids, indexing="ij")
        rr.extend(row.ravel())
        cc.extend(col.ravel())
        vv.extend(block.ravel())
    matrix = sp.coo_matrix((vv, (rr, cc)), shape=(ndof, ndof)).tocsr()
    matrix.eliminate_zeros()
    return matrix


def q4_reference(mesh, material=None, thickness=0.001):
    """Independent 2x2 Gaussian Q4 continuum reference (no star residual energy)."""
    material = material or pzt4()
    blocks = []
    for conn in mesh.cells:
        xy = mesh.coordinates[conn]
        center = xy.mean(axis=0)
        mat = material(center) if callable(material) else material
        local = np.zeros((12, 12))
        for xi in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
            for eta in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
                grad, det = quad_gradients(xy, xi, eta)
                b = field_map(grad)
                local += thickness * det * b.T @ mat.enthalpy @ b
        ids = (3 * conn[:, None] + np.arange(3)).ravel()
        blocks.append((ids, local))
    return assemble_local(blocks, 3 * mesh.n)


def solve_coupled(
    matrix, mesh, prescribed, rhs=None, crack_electric="impermeable", electrode_groups=()
):
    """Scaled saddle solve with exact voltage/essential elimination and MPC ties.

    Mechanical crack topology is already in the mesh. Ideal permeable faces
    are equipotential pairs with independently moving displacements.
    Each electrode group is a sequence of particle IDs with a common potential.
    A floating group has zero NET charge unless its contracted RHS is nonzero;
    local surface charge need not vanish. Mechanical DOFs are never tied.
    Electrical RHS denotes outward material flux, NOT electrode free charge.
    """
    if crack_electric not in ("impermeable", "permeable"):
        raise ValueError("Unsupported crack electrical model")
    n = matrix.shape[0]
    electrode_groups = tuple(np.asarray(nodes, int).ravel() for nodes in electrode_groups)
    if any(not 0 <= int(dof) < n for dof in prescribed):
        raise ValueError("Prescribed DOF outside the model")
    rhs = np.zeros(n) if rhs is None else np.asarray(rhs, float).reshape(n)
    group = np.arange(n)

    def root(k):
        while group[k] != k:
            group[k] = group[group[k]]
            k = group[k]
        return k

    def tie(a, b):
        ra, rb = root(a), root(b)
        group[max(ra, rb)] = min(ra, rb)

    if crack_electric == "permeable":
        for a, b in mesh.face_pairs:
            tie(3 * int(a) + 2, 3 * int(b) + 2)
    for nodes in electrode_groups:
        nodes = np.asarray(nodes, int).ravel()
        if not len(nodes) or np.any((nodes < 0) | (nodes >= mesh.n)):
            raise ValueError("Invalid electrode particle IDs")
        for node in nodes[1:]:
            tie(3 * int(nodes[0]) + 2, 3 * int(node) + 2)
    group = np.array([root(i) for i in range(n)])
    # A finite answer from a nearly singular direct solve does not establish
    # uniqueness. Check rigid modes and one electrical gauge per component.
    i, j = mesh.pairs.T
    adjacency = sp.coo_matrix(
        (np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(mesh.n, mesh.n)
    ).tocsr()
    components, labels = sp.csgraph.connected_components(adjacency, directed=False)
    prescribed_ids = np.array(list(prescribed), int)
    for component in range(components):
        nodes = np.flatnonzero(labels == component)
        center = mesh.coordinates[nodes].mean(axis=0)
        rows = []
        for dof in prescribed_ids:
            node, direction = divmod(int(dof), 3)
            if labels[node] != component or direction == 2:
                continue
            p = (mesh.coordinates[node] - center) / mesh.spacing
            rows.append([1.0, 0.0, -p[1]] if direction == 0 else [0.0, 1.0, p[0]])
        if not rows or np.linalg.matrix_rank(np.array(rows)) < 3:
            raise ValueError(
                "Mechanical supports do not remove all three rigid modes of each component"
            )
    electrical_edges = list(map(tuple, mesh.pairs))
    if crack_electric == "permeable":
        electrical_edges.extend(map(tuple, mesh.face_pairs))
    for nodes in electrode_groups:
        electrical_edges.extend((int(nodes[0]), int(node)) for node in nodes[1:])
    ei, ej = np.array(electrical_edges, int).T
    electro_adj = sp.coo_matrix(
        (np.ones(2 * len(ei)), (np.r_[ei, ej], np.r_[ej, ei])), shape=(mesh.n, mesh.n)
    ).tocsr()
    ecomponents, elabels = sp.csgraph.connected_components(electro_adj, directed=False)
    grounded = {int(elabels[int(dof) // 3]) for dof in prescribed_ids if dof % 3 == 2}
    if len(grounded) != ecomponents:
        raise ValueError("Missing electric potential gauge on a connected electrical component")
    representatives, inverse = np.unique(group, return_inverse=True)
    prolongation = sp.coo_matrix(
        (np.ones(n), (np.arange(n), inverse)), shape=(n, len(representatives))
    ).tocsr()
    reduced = (prolongation.T @ matrix @ prolongation).tocsr()
    load = prolongation.T @ rhs
    fixed = {}
    for dof, value in prescribed.items():
        k = int(inverse[int(dof)])
        if k in fixed and not np.isclose(fixed[k], value, rtol=1e-12, atol=1e-15):
            raise ValueError("Conflicting potential values on tied crack faces")
        fixed[k] = float(value)
    fixed_ids = np.array(sorted(fixed), int)
    x = np.zeros(len(representatives))
    x[fixed_ids] = [fixed[k] for k in fixed_ids]
    free = np.setdiff1d(np.arange(len(x)), fixed_ids)
    reduced_rhs = load[free] - reduced[free][:, fixed_ids] @ x[fixed_ids]
    a = reduced[free][:, free].tocsr()
    diagonal = abs(a.diagonal())
    if len(free) and np.any(diagonal <= 0):
        raise ValueError("Zero diagonal: missing support/gauge or invalid topology")
    scale = 1 / np.sqrt(diagonal)
    d = sp.diags(scale)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        x[free] = scale * spsolve((d @ a @ d).tocsc(), scale * reduced_rhs)
    solve_seconds = time.perf_counter() - started
    state = prolongation @ x
    reaction = matrix @ state - rhs
    contracted = prolongation.T @ reaction
    # Unit-aware residuals. Electrical and mechanical residuals are never mixed
    # in a single unscaled physical norm.
    residuals = {}
    for label, selector in (
        ("mechanical", representatives % 3 != 2),
        ("electrical", representatives % 3 == 2),
    ):
        ids = free[selector[free]]
        action = (prolongation.T @ (matrix @ state))[ids]
        # Sum |Kij xj| captures cancellation even for a zero applied field/load.
        reference = (abs(reduced[ids]) @ abs(x)) + abs(load[ids])
        residuals[label] = float(
            np.linalg.norm(contracted[ids]) / max(np.linalg.norm(reference), np.finfo(float).tiny)
        )
    if not np.all(np.isfinite(state)):
        raise ValueError("Nonfinite coupled solution")
    return dict(
        state=state.reshape(mesh.n, 3),
        reaction=reaction.reshape(mesh.n, 3),
        solve_seconds=solve_seconds,
        residuals=residuals,
        electrical_mpcs=n - len(representatives),
        free_dofs=len(free),
    )
