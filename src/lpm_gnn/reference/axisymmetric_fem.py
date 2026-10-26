"""Independent tensor Q1/Q2 axisymmetric harmonic FEM, NOT an LPM implementation.

SI units. Polarization z. Strain [rr,tt,zz,gamma_rz], electric gradient [r,z].
Electric enthalpy: .5 eps C eps + eps e.T grad(phi) - .5 grad(phi) k grad(phi).
exp(+i*w*t); electrode charge is minus its internal potential residual.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

EPS0 = 8.8541878128e-12
NAMES = ("c11", "c12", "c13", "c33", "c44", "e15", "e31", "e33", "eps11", "eps33")
U_SCALE = 1e-9
ENERGY_SCALE = 1e-9


@dataclass(frozen=True)
class Material:
    c11: float = 1.38999e11
    c12: float = 7.78366e10
    c13: float = 7.42836e10
    c33: float = 1.15412e11
    c44: float = 2.5641e10
    e15: float = 12.718
    e31: float = -5.200
    e33: float = 15.080
    eps11: float = 762.5
    eps33: float = 663.2
    rho: float = 7500.0
    loss_m: float = 0.002
    loss_d: float = 0.0

    def matrices(self):
        c = np.array(
            [
                [self.c11, self.c12, self.c13, 0],
                [self.c12, self.c11, self.c13, 0],
                [self.c13, self.c13, self.c33, 0],
                [0, 0, 0, self.c44],
            ],
            dtype=float,
        )
        e = np.array([[0, 0, 0, self.e15], [self.e31, self.e31, self.e33, 0]], dtype=float)
        k = EPS0 * np.diag([self.eps11, self.eps33])
        return c, e, k

    def check(self):
        c, _, k = self.matrices()
        if np.linalg.eigvalsh(c).min() <= 0 or np.linalg.eigvalsh(k).min() <= 0:
            raise ValueError("Non-positive real material stiffness/permittivity")
        if self.rho <= 0 or self.loss_m < 0 or self.loss_d < 0:
            raise ValueError("Non-passive material")


class AxisymmetricDisk:
    def __init__(self, nr=40, nz=8, radius=0.0125, height=0.00202, order=2):
        self.nr, self.nz = int(nr), int(nz)
        if order not in (1, 2):
            raise ValueError("Only Q1/Q2 supported")
        self.order = order
        self.radius, self.height = float(radius), float(height)
        rr, zz = np.meshgrid(
            np.linspace(0, radius, nr * order + 1),
            np.linspace(0, height, nz * order + 1),
            indexing="ij",
        )
        self.xy = np.column_stack([rr.ravel(), zz.ravel()])
        grid = np.arange((nr * order + 1) * (nz * order + 1)).reshape(
            nr * order + 1, nz * order + 1
        )
        self.cells = np.array(
            [
                [
                    grid[order * i + a, order * j + b]
                    for a in range(order + 1)
                    for b in range(order + 1)
                ]
                for i in range(nr)
                for j in range(nz)
            ],
            dtype=int,
        )
        self.nnode = len(self.xy)
        self.ndof = 3 * self.nnode
        # Unknown mechanical DOFs first, potential last. All top nodes share one port.
        self.bottom = np.flatnonzero(np.isclose(self.xy[:, 1], 0, atol=1e-15))
        self.top = np.flatnonzero(np.isclose(self.xy[:, 1], height, atol=1e-15))
        self.axis = np.flatnonzero(np.isclose(self.xy[:, 0], 0, atol=1e-15))
        mapping = np.full(self.ndof, -1, dtype=int)
        forbidden = np.r_[2 * self.axis, 2 * self.nnode + self.bottom, 2 * self.nnode + self.top]
        free = np.setdiff1d(np.arange(self.ndof), forbidden)
        mapping[free] = np.arange(len(free))
        mapping[2 * self.nnode + self.top] = len(free)
        mapping[2 * self.nnode + self.bottom] = len(free) + 1
        self.mapping, self.free_global = mapping, free
        self.port = len(free)
        self.ground_port = len(free) + 1
        self.nunknown = len(free)
        self.nreduced = len(free) + 2
        self._assemble()

    def _assemble(self):
        ne = len(self.cells)
        nl = self.cells.shape[1]
        km = np.zeros((5, ne, 2 * nl, 2 * nl))
        ke = np.zeros((3, ne, 2 * nl, nl))
        kd = np.zeros((2, ne, nl, nl))
        mass = np.zeros((ne, 2 * nl, 2 * nl))
        cunit = np.zeros((5, 4, 4))
        cunit[0, 0, 0] = cunit[0, 1, 1] = 1
        cunit[1, 0, 1] = cunit[1, 1, 0] = 1
        cunit[2, 0, 2] = cunit[2, 2, 0] = cunit[2, 1, 2] = cunit[2, 2, 1] = 1
        cunit[3, 2, 2] = cunit[4, 3, 3] = 1
        eunit = np.zeros((3, 2, 4))
        eunit[0, 0, 3] = 1
        eunit[1, 1, 0] = eunit[1, 1, 1] = 1
        eunit[2, 1, 2] = 1
        dunit = np.zeros((2, 2, 2))
        dunit[0, 0, 0] = dunit[1, 1, 1] = 1
        volume = 0.0
        points, weights = np.polynomial.legendre.leggauss(self.order + 1)

        def lagrange(t):
            if self.order == 1:
                return np.array([(1 - t) / 2, (1 + t) / 2]), np.array([-0.5, 0.5])
            return np.array([t * (t - 1) / 2, 1 - t * t, t * (t + 1) / 2]), np.array(
                [t - 0.5, -2 * t, t + 0.5]
            )

        for xi, wi in zip(points, weights):
            for eta, wj in zip(points, weights):
                lx, dx = lagrange(xi)
                ly, dy = lagrange(eta)
                n = np.kron(lx, ly)
                deriv = np.column_stack([np.kron(dx, ly), np.kron(lx, dy)])
                jac = np.einsum("eaD,ai->eDi", self.xy[self.cells], deriv)
                det = np.linalg.det(jac)
                if np.any(det <= 0):
                    raise ValueError("Nonpositive Jacobian")
                grad = np.einsum("ai,eiD->eaD", deriv, np.linalg.inv(jac))
                rq = self.xy[self.cells][:, :, 0] @ n
                w = 2 * np.pi * rq * det * wi * wj
                volume += w.sum()
                b = np.zeros((ne, 4, 2 * nl))
                b[:, 0, 0::2] = grad[:, :, 0]
                b[:, 1, 0::2] = n[None, :] / rq[:, None]
                b[:, 2, 1::2] = grad[:, :, 1]
                b[:, 3, 0::2] = grad[:, :, 1]
                b[:, 3, 1::2] = grad[:, :, 0]
                g = grad.transpose(0, 2, 1)
                km += np.einsum("eai,kab,ebj,e->keij", b, cunit, b, w) * U_SCALE**2 / ENERGY_SCALE
                ke += np.einsum("eai,kba,ebj,e->keij", b, eunit, g, w) * U_SCALE / ENERGY_SCALE
                kd -= np.einsum("eai,kab,ebj,e->keij", g, dunit, g, w) * EPS0 / ENERGY_SCALE
                nn = n[:, None] * n[None, :]
                mass[:, 0::2, 0::2] += w[:, None, None] * nn
                mass[:, 1::2, 1::2] += w[:, None, None] * nn
        udofs = np.stack([2 * self.cells, 2 * self.cells + 1], axis=-1).reshape(ne, 2 * nl)
        pdofs = 2 * self.nnode + self.cells
        self.bases = []
        for local in km:
            self.bases.append(self._scatter(local, udofs, udofs))
        for local in ke:
            matrix = self._scatter(local, udofs, pdofs)
            self.bases.append((matrix + matrix.T).tocsc())
        for local in kd:
            self.bases.append(self._scatter(local, pdofs, pdofs))
        self.mass = self._scatter(mass * U_SCALE**2 / ENERGY_SCALE, udofs, udofs)
        self.volume = float(volume)
        self.uz_translation = np.zeros(self.nreduced)
        axial = 2 * np.arange(self.nnode) + 1
        self.uz_translation[self.mapping[axial]] = 1.0

    def _scatter(self, local, rowdofs, coldofs):
        rows = np.broadcast_to(self.mapping[rowdofs][:, :, None], local.shape).ravel()
        cols = np.broadcast_to(self.mapping[coldofs][:, None, :], local.shape).ravel()
        values = local.ravel()
        ok = (rows >= 0) & (cols >= 0)
        n = self.nreduced
        result = sparse.coo_matrix((values[ok], (rows[ok], cols[ok])), shape=(n, n)).tocsc()
        result.eliminate_zeros()
        return result

    def matrix(self, material, frequency=0.0):
        material.check()
        weights = np.array([getattr(material, n) for n in NAMES], dtype=complex)
        weights[:5] *= 1 + 1j * material.loss_m
        weights[8:] *= 1 - 1j * material.loss_d
        result = sum(
            (value * b for value, b in zip(weights, self.bases)),
            sparse.csc_matrix(self.mass.shape, dtype=complex),
        )
        return (result - (2 * np.pi * frequency) ** 2 * material.rho * self.mass).tocsc()

    def solve(self, material, frequency, voltage=1.0, diagnostics=False):
        a = self.matrix(material, frequency)
        unknown = np.arange(self.nunknown)
        if frequency == 0:
            # One point only removes the exact axial rigid translation at static equilibrium.
            unknown = unknown[unknown != self.mapping[1]]
        aff = a[unknown, :][:, unknown].tocsc()
        afp = a[unknown, self.port].toarray().ravel()
        lu = splu(aff)
        x = np.zeros(self.nreduced, dtype=complex)
        x[self.port] = voltage
        x[unknown] = lu.solve(-afp * voltage)
        residual = a @ x
        charge = -residual[self.port] * ENERGY_SCALE
        admittance = 1j * 2 * np.pi * frequency * charge / voltage
        impedance = 1 / admittance if frequency else complex(np.inf)
        output = dict(charge=charge, admittance=admittance, impedance=impedance)
        if diagnostics:
            qh_a_q = np.vdot(x, a @ x)
            input_power = 0.5 * np.real(voltage * np.conj(admittance * voltage))
            dissipated = 0.5 * 2 * np.pi * frequency * np.imag(qh_a_q) * ENERGY_SCALE
            row_scale = np.asarray(abs(a[unknown, :]) @ abs(x)).ravel()
            relative = np.linalg.norm(residual[unknown]) / max(np.linalg.norm(row_scale), 1e-300)
            full = np.zeros(self.ndof, dtype=complex)
            active = self.mapping >= 0
            full[active] = x[self.mapping[active]]
            full[: 2 * self.nnode] *= U_SCALE
            ground_charge = -residual[self.ground_port] * ENERGY_SCALE
            centroid = np.vdot(self.uz_translation, self.mass @ x)
            centroid_denominator = max(
                np.linalg.norm(self.mass @ self.uz_translation) * np.linalg.norm(x), 1e-300
            )
            output.update(
                x_scaled=x,
                full=full,
                relative_free_residual=float(relative),
                input_power=float(input_power),
                dissipated_power=float(dissipated),
                charge_balance_relative_error=float(
                    abs(charge + ground_charge) / max(abs(charge), 1e-300)
                ),
                axial_centroid_relative_error=float(abs(centroid) / centroid_denominator),
                power_relative_error=float(
                    abs(input_power - dissipated) / max(abs(input_power), 1e-30)
                ),
                axial_total_mass=float(
                    material.rho
                    * ENERGY_SCALE
                    / U_SCALE**2
                    * np.vdot(self.uz_translation, self.mass @ self.uz_translation).real
                ),
            )
        return output

    def spectrum(self, material, frequencies, diagnostics=False):
        data = [self.solve(material, float(f), diagnostics=diagnostics) for f in frequencies]
        result = {
            key: np.array([d[key] for d in data]) for key in ("charge", "admittance", "impedance")
        }
        if diagnostics:
            result.update(
                {
                    key: np.array([d[key] for d in data])
                    for key in (
                        "relative_free_residual",
                        "input_power",
                        "power_relative_error",
                        "charge_balance_relative_error",
                        "axial_centroid_relative_error",
                    )
                }
            )
        return result


def thickness_impedance(material, frequencies, radius, height):
    """Exact 1D free thickness plate when e31=c13=e15=0; finite circular area."""
    c = material.c33 * (1 + 1j * material.loss_m)
    kappa = EPS0 * material.eps33 * (1 - 1j * material.loss_d)
    cd = c + material.e33**2 / kappa
    x = 2 * np.pi * np.asarray(frequencies) * np.sqrt(material.rho / cd) * height / 2
    coupling = material.e33**2 / (kappa * cd)
    capacitance = kappa * np.pi * radius**2 / height / (1 - coupling * np.tan(x) / x)
    return 1 / (1j * 2 * np.pi * np.asarray(frequencies) * capacitance)
