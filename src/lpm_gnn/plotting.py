"""Compact result plots with shared scales and mathematical axis labels."""

import numpy as np


def planar_fields(coordinates, state, reference, output, cells=None, labels=("LPM-GNN", "FEM")):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    plt.rcParams.update({"font.size": 10, "mathtext.fontset": "stix", "font.family": "serif"})
    xy = np.asarray(coordinates) * 1000
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1])
    if cells is not None:
        cell_edges = {
            tuple(sorted((int(a), int(b)))) for cell in cells for a in cell for b in cell if a != b
        }
        triangulation.set_mask(
            [
                any(
                    tuple(sorted((int(t[i]), int(t[(i + 1) % 3])))) not in cell_edges
                    for i in range(3)
                )
                for t in triangulation.triangles
            ]
        )
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5), constrained_layout=True)
    for row, (column, scale, title) in enumerate(
        ((1, 1e6, r"$u_y$ ($\mu$m)"), (2, 1.0, r"$\phi$ (V)"))
    ):
        low = min(state[:, column].min(), reference[:, column].min()) * scale
        high = max(state[:, column].max(), reference[:, column].max()) * scale
        if low == high:
            high = low + 1e-15
        for j, values in enumerate((state, reference)):
            ax = axes[row, j]
            image = ax.tripcolor(
                triangulation,
                values[:, column] * scale,
                shading="gouraud",
                cmap="viridis",
                vmin=low,
                vmax=high,
            )
            ax.set_title(labels[j])
            ax.set_aspect("equal")
            ax.set_xlabel(r"$x$ (mm)")
            ax.set_ylabel(r"$y$ (mm)")
        fig.colorbar(image, ax=list(axes[row]), label=title, shrink=0.85)
    fig.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
