"""Equivalent float64 NumPy/SciPy inference for the compared graph networks."""

from .physics import np, sp


def numpy(t):
    return t.detach().cpu().numpy().copy()


class Kernel:
    def __init__(self, kernel):
        self.state = {k: numpy(v).astype(np.float64) for k, v in kernel.state_dict().items()}

    def linear(self, x, name):
        s = self.state
        return x @ s[name + ".weight"].T + s.get(name + ".bias", 0)

    def mlp(self, x, name):
        x = self.linear(x, name + ".0")
        return self.linear(x / (1 + np.exp(-x)), name + ".2")

    def propagate(self, h, adj, params, stage, count=2):
        for _ in range(count):
            aggregate = sum(self.linear(a @ h, "messages." + str(k)) for k, a in enumerate(adj))
            glob = self.mlp(np.concatenate((h.mean(0), h.max(0), params)), "global_encoder")
            update = (
                self.mlp(
                    np.concatenate((h, aggregate, np.broadcast_to(glob, h.shape)), -1), "update"
                )
                + self.state["stage"][stage]
            )
            value = h + update
            mean = value.mean(-1, keepdims=True)
            h = (value - mean) / np.sqrt(((value - mean) ** 2).mean(-1, keepdims=True) + 1e-5)
            h = h * self.state["norm.weight"] + self.state["norm.bias"]
        return h


def scipy_csr(t):
    t = t.to_sparse_coo().coalesce()
    index = numpy(t.indices())
    return sp.coo_matrix((numpy(t.values()), index), shape=t.shape).tocsr()


class CompactCPU:
    def __init__(self, net, env):
        self.env = env
        self.net = net
        self.kernel = Kernel(net.kernel)
        self.c = env.cycle
        self.adj = [scipy_csr(getattr(env.graph, "adj" + str(i))) for i in range(3)]
        self.p = scipy_csr(env.p)
        self.average = scipy_csr(env.average)
        self.static = numpy(env.static)
        self.field_scale = numpy(env.field_scale)
        self.diag = numpy(env.diag)
        self.output_scale = numpy(env.output_scale)
        self.mask = numpy(env.mask)
        self.basis = numpy(env.force_basis).reshape(len(self.c.top), -1)
        self.voltage = numpy(env.voltage).ravel()
        self.local = {k: numpy(v).astype(np.float64) for k, v in net.local.state_dict().items()}

    def features(self, y, b, lift, params, r=None):
        if r is None:
            rn = np.zeros_like(y)
        else:
            scale = np.sqrt((r * r * self.mask).sum(0) / self.mask.sum(0) + 1e-24)
            rn = r / scale
        return np.column_stack(
            (
                self.static,
                b / self.diag,
                lift,
                np.broadcast_to(params, (len(y), 4)),
                y / self.field_scale,
                rn,
            )
        )

    def __call__(self, b, lift, params):
        y = (b[self.c.top, 1] @ self.basis + params[1] * self.voltage).reshape(-1, 3)
        if self.net.kind in ("CompactData", "CompactPI"):
            f = self.features(y, b, lift, params)
            h = self.kernel.mlp(self.average @ f, "encoder")
            h = self.kernel.propagate(h, self.adj, params, 1, 3)
            x = np.column_stack((self.p @ h, f))
            x = x @ self.local["0.weight"].T + self.local["0.bias"]
            x = x / (1 + np.exp(-x))
            delta = x @ self.local["2.weight"].T + self.local["2.bias"]
            return y + delta * self.output_scale * self.mask
        for stage in range(self.net.cycles):
            r = ((b.ravel() - self.c.physical.action(y.ravel())).reshape(-1, 3)) * self.mask
            f = self.average @ self.features(y, b, lift, params, r)
            h = self.kernel.mlp(f, "encoder")
            h = self.kernel.propagate(h, self.adj, params, min(stage, 2), 2)
            weight = 0.8 + 0.35 * np.tanh(self.p @ self.kernel.mlp(h, "decoder"))
            full = np.zeros_like(y).ravel()
            full[self.c.free] = self.c.apply((r * weight).ravel()[self.c.free])
            y = y + full.reshape(-1, 3)
        return y


class ScheduleCPU(CompactCPU):
    def __call__(self, b, lift, params):
        y = (b[self.c.top, 1] @ self.basis + params[1] * self.voltage).reshape(-1, 3)
        r = ((b.ravel() - self.c.physical.action(y.ravel())).reshape(-1, 3)) * self.mask
        h = self.kernel.mlp(self.average @ self.features(y, b, lift, params, r), "encoder")
        h = self.kernel.propagate(h, self.adj, params, 1, 1)
        weights = (0.8 + 0.35 * np.tanh(self.p @ self.kernel.mlp(h, "decoder"))).reshape(
            len(y), self.net.cycles, 3
        )
        for stage in range(self.net.cycles):
            if stage:
                r = ((b.ravel() - self.c.physical.action(y.ravel())).reshape(-1, 3)) * self.mask
            full = np.zeros_like(y).ravel()
            full[self.c.free] = self.c.apply((r * weights[:, stage]).ravel()[self.c.free])
            y = y + full.reshape(-1, 3)
        return y
