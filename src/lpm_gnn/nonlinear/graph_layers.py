"""Collective-neighborhood graph support and geometry-only transfers."""

from .physics import PiezoMessageGraph, nn, np, sp, tensor_sparse, torch

DT = torch.float64


def make_level(mesh, device):
    """Full star-dependency support, including collective nonlocal pairs."""
    graph = PiezoMessageGraph(mesh)
    nodes = graph.star_node.cpu().numpy()
    stars = graph.star_node_star.cpu().numpy()
    edges = set()
    order = np.argsort(stars)
    for group in np.split(order, np.flatnonzero(np.diff(stars[order])) + 1):
        ids = np.unique(nodes[group])
        edges.update((int(a), int(b)) for a in ids for b in ids if a != b)
    edges = np.array(sorted(edges), dtype=np.int64).T
    delta = (mesh.coordinates[edges[0]] - mesh.coordinates[edges[1]]) / mesh.spacing
    attributes = np.column_stack((delta, np.linalg.norm(delta, axis=1))) / 2.0
    degree = np.bincount(edges[1], minlength=mesh.n)
    norm = 1.0 / np.sqrt(degree[edges[0]] * degree[edges[1]])
    return dict(
        edge=torch.tensor(edges, dtype=torch.long, device=device),
        attributes=torch.tensor(attributes, dtype=DT, device=device),
        norm=torch.tensor(norm, dtype=DT, device=device),
        n=mesh.n,
    )


def make_transfer(fine, coarse, device):
    """Bilinear node transfer restricted to an actual material cell."""
    fg = np.rint(fine.coordinates / fine.spacing).astype(int)
    cg = np.rint(coarse.coordinates / coarse.spacing).astype(int)
    lookup = {tuple(v): i for i, v in enumerate(cg)}
    cells = {(int(cg[c, 0].min()), int(cg[c, 1].min())) for c in coarse.cells}
    fi, ci, w = [], [], []
    for i, (ix, iy) in enumerate(fg):
        cx, cy = ix // 2, iy // 2
        candidates = sorted(
            (x, y)
            for x in ([cx - 1, cx] if ix % 2 == 0 else [cx])
            for y in ([cy - 1, cy] if iy % 2 == 0 else [cy])
            if (x, y) in cells
        )
        assert candidates, "Transfer cannot cross a window"
        x, y = candidates[0]
        tx, ty = ix / 2 - x, iy / 2 - y
        for dx, dy, weight in (
            (0, 0, (1 - tx) * (1 - ty)),
            (1, 0, tx * (1 - ty)),
            (1, 1, tx * ty),
            (0, 1, (1 - tx) * ty),
        ):
            if weight > 0:
                fi.append(i)
                ci.append(lookup[(x + dx, y + dy)])
                w.append(weight)
    fi, ci, w = np.array(fi), np.array(ci), np.array(w)
    assert np.max(abs(np.bincount(fi, weights=w) - 1)) < 1e-14
    vw = w * fine.volumes[fi]
    restriction = vw / np.bincount(ci, weights=vw)[ci]
    assert np.max(abs(np.bincount(ci, weights=restriction) - 1)) < 1e-14
    return dict(
        fine=torch.tensor(fi, dtype=torch.long, device=device),
        coarse=torch.tensor(ci, dtype=torch.long, device=device),
        prolong=torch.tensor(w, dtype=DT, device=device),
        restrict=torch.tensor(restriction, dtype=DT, device=device),
        nf=fine.n,
        nc=coarse.n,
        crosses_window=False,
    )


class GraphLevel(nn.Module):
    def __init__(self, a, device, dtype):
        super().__init__()
        self.n = a["mesh"].n
        level = make_level(a["mesh"], "cpu")
        src, dst = level["edge"].numpy()
        delta = (a["mesh"].coordinates[src] - a["mesh"].coordinates[dst]) / a["mesh"].spacing
        norm = level["norm"].numpy()
        for i, factor in enumerate((np.ones(len(src)), delta[:, 0] / 2, delta[:, 1] / 2)):
            adj = sp.coo_matrix((norm * factor, (dst, src)), shape=(self.n, self.n)).tocsr()
            self.register_buffer("adj" + str(i), tensor_sparse(adj, device, dtype))
        self.register_buffer(
            "static",
            torch.tensor(
                np.column_stack((2 * a["mesh"].coordinates / 0.02 - 1, a["mask"])),
                dtype=dtype,
                device=device,
            ),
        )

    def aggregate(self, h, k):
        # One sparse graph operation for the whole batch/hidden channel set.
        b, n, w = h.shape
        value = torch.sparse.mm(getattr(self, "adj" + str(k)), h.permute(1, 0, 2).reshape(n, -1))
        return value.reshape(n, b, w).permute(1, 0, 2)


class SharedKernel(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.width = width

        def mlp(a, b, c):
            return nn.Sequential(nn.Linear(a, b), nn.SiLU(), nn.Linear(b, c))

        self.encoder = mlp(21, width, width)
        self.messages = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(3)])
        self.global_encoder = mlp(2 * width + 4, width, width)
        self.update = mlp(3 * width, 2 * width, width)
        self.norm = nn.LayerNorm(width)
        self.decoder = mlp(width, width, 3)
        self.stage = nn.Parameter(torch.zeros(3, width))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def propagate(self, h, level, params, stage, count=2):
        for _ in range(count):
            aggregate = sum(m(level.aggregate(h, k)) for k, m in enumerate(self.messages))
            pooled = torch.cat((h.mean(1), h.amax(1), params), -1)
            global_state = self.global_encoder(pooled)[:, None].expand_as(h)
            update = self.update(torch.cat((h, aggregate, global_state), -1)) + self.stage[stage]
            h = self.norm(h + update)
        return h
