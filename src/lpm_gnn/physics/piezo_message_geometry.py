"""Pure geometric particle/cell incidence. No stiffness matrices or solvers.

The square-cell construction follows the existing project's QuadParticleMesh
geometry; this module does not instantiate its material or assembled models.
Cells define area, same-side support, and crack topology, not FEM quadrature.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class MessageGeometry:
    coordinates: np.ndarray
    cells: np.ndarray
    spacing: float
    face_pairs: np.ndarray

    def __post_init__(self):
        self.coordinates = np.asarray(self.coordinates, float).reshape(-1, 2)
        self.cells = np.asarray(self.cells, np.int64).reshape(-1, 4)
        self.face_pairs = np.asarray(self.face_pairs, np.int64).reshape(-1, 2)
        self.n = len(self.coordinates)
        self.volumes = np.zeros(self.n)
        pairs = set()
        for cell in self.cells:
            xy = self.coordinates[cell]
            area = 0.5 * abs(
                np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
            )
            if area <= 0:
                raise ValueError("Nonpositive geometric cell area")
            self.volumes[cell] += area / 4
            pairs.update(tuple(sorted((int(i), int(j)))) for i, j in combinations(cell, 2))
        if np.any(self.volumes <= 0):
            raise ValueError("Unused particle")
        self.pairs = np.asarray(sorted(pairs), np.int64).reshape(-1, 2)

    @classmethod
    def rectangle(cls, nx, ny=None, width=0.02, height=None, crack_cells=0):
        ny = nx if ny is None else ny
        height = width if height is None else height
        if nx < 1 or ny < 1 or not np.isclose(width / nx, height / ny):
            raise ValueError("Positive square lattice spacing required")
        if crack_cells and (ny % 2 or not 0 < crack_cells < nx):
            raise ValueError("Edge crack requires even ny and 0<a<nx")
        points = [(i * width / nx, j * height / ny) for j in range(ny + 1) for i in range(nx + 1)]
        duplicates, faces = {}, []
        for i in range(crack_cells):
            node = (ny // 2) * (nx + 1) + i
            duplicates[node] = len(points)
            points.append(points[node])
            faces.append((node, duplicates[node]))
        cells = []
        for j in range(ny):
            for i in range(nx):
                ll = j * (nx + 1) + i
                conn = [ll, ll + 1, ll + nx + 2, ll + nx + 1]
                if crack_cells and j >= ny // 2:
                    conn = [duplicates.get(k, k) for k in conn]
                cells.append(conn)
        return cls(points, cells, width / nx, faces)

    @classmethod
    def window(cls, n, width=0.02, crack_length=0.002):
        """Two rectangular windows in a square: a synthetic engineering proxy."""
        if n < 10 or n % 10:
            raise ValueError("n must be a positive multiple of 10")
        h = width / n
        base = cls.rectangle(n, crack_cells=int(round(crack_length / h)))
        keep = []
        for cell in base.cells:
            x, y = base.coordinates[cell].mean(axis=0)
            inside_y = 0.25 * width < y < 0.75 * width
            inside_x = 0.2 * width < x < 0.4 * width or 0.6 * width < x < 0.8 * width
            if not (inside_x and inside_y):
                keep.append(cell)
        cells = np.asarray(keep, np.int64)
        used = np.unique(cells)
        ids = np.full(base.n, -1, np.int64)
        ids[used] = np.arange(len(used))
        faces = [(ids[a], ids[b]) for a, b in base.face_pairs if ids[a] >= 0 and ids[b] >= 0]
        return cls(base.coordinates[used], ids[cells], h, faces)
