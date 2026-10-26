"""Fully coupled 40 -> 20 -> 40 Galerkin V-cycle."""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from .physics import SparsePhysics, physical_arrays, transfer_arrays


class CoupledCycle:
    def __init__(self, basis_path, pre=1, post=1, omega=0.65):
        tic = time.perf_counter()
        self.physical = SparsePhysics()
        a = self.physical.a
        self.free = self.physical.free
        self.k = self.physical.kfree
        self.coarse = physical_arrays(20)
        p, _ = transfer_arrays(a, self.coarse)
        self.p = sp.kron(p, sp.eye(3), format="csr")[self.free][
            :, np.flatnonzero(self.coarse["mask"].ravel())
        ]
        self.pt = self.p.T.tocsr()
        self.kc = (self.pt @ self.k @ self.p).tocsc()
        self.lu = sla.splu(self.kc)
        self.diag = self.k.diagonal()
        self.pre = pre
        self.post = post
        self.omega = omega
        z = np.load(basis_path)
        self.basis = z["force"].reshape(len(z["top"]), -1)
        self.voltage = z["voltage"].ravel()
        self.top = z["top"]
        self.setup_seconds = time.perf_counter() - tic

    def initial(self, b, params):
        return b[self.top, 1] @ self.basis + (params[1] / 100) * self.voltage

    def apply(self, r):
        x = np.zeros_like(r)
        for _ in range(self.pre):
            x += self.omega * (r - self.k @ x) / self.diag
        x += self.p @ self.lu.solve(self.pt @ (r - self.k @ x))
        for _ in range(self.post):
            x += self.omega * (r - self.k @ x) / self.diag
        return x

    def predict(self, b, params, cycles, alpha=1.0):
        y = self.initial(b, params)
        for _ in range(cycles):
            r = (b.ravel() - self.physical.action(y))[self.free]
            y[self.free] += alpha * self.apply(r)
        return y.reshape(-1, 3)
