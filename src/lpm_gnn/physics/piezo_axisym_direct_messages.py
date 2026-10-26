"""Axisymmetric adaptation of a direct eight-bond, many-body particle star.

NEW MODEL, not the old four-neighbour reconstructed/GMLS candidate or Q2 FEM.
The real meridional mechanical energy is the direct bond/star energy.  Its
energy-orthogonal affine projection supplies [er,ez,gamma]; the positive real
non-affine remainder is retained when adding the full TI hoop energy.  The
piezoelectric coupling is explicitly a NEW projected-strain/axial-gradient
choice, not a claim to preserve every old planar non-affine G coefficient.

All physical actions are closed-form local chain-rule messages; autograd of a
full-field energy is NEVER used to compute an action.  Local coefficient
vectors/incidences are not global system matrices.  No dense/sparse matrix,
inverse, factorization, linear solve or old FEM/model import is used here.

real coefficients (physical SI):
 [C11,C12,C13,C33,C44,e15,e31,e33,kappa11,kappa33].
imag coefficients, optional (physical SI, +i*omega*t):
 [ImC11,ImC12,ImC13,ImC33,ImC44,Imkappa11,Imkappa33].
Piezoelectric coefficients are real.  Passive Cimag is PSD; kappa imag<=0.
eta_nonaffine>=0 multiplies the REAL non-affine remainder as extra imaginary
energy.  It defaults to zero; it is a declared discretization control, not a
fitted material parameter.  Projection depends on real C only.

Geometry is the exact R,H rectangular meridional grid (dr need not equal dz),
with positive exact truncated dual ring volumes, including the axis.  Real
inside bonds are retained.  Negative-r neighbours are parity-reflected when
their z coordinate exists (ur odd; uz/phi even).  Other absent directions use
same-side affine difference completion.  At axis/top-bottom corners that
completion uses the regularity-aware radial derivative (ur slope; uz/phi zero).
Hoop strain is ur/r off-axis and er on-axis.  ur(axis)=0 and face potentials
MUST be imposed by the caller's indexed constraints, not by this action.

Energy is a complex NONCONJUGATE dynamic functional, not stored mean energy.
``dissipated_power`` separately uses conjugated quadratic products.  Frequency
zero is allowed for patches/gauges; this module does not solve any system.
No patch, dispersion, convergence or experimental-validity claim is made.
"""

import math

import torch

REAL_NAMES = ("c11", "c12", "c13", "c33", "c44", "e15", "e31", "e33", "kappa11", "kappa33")
IMAG_NAMES = ("c11_im", "c12_im", "c13_im", "c33_im", "c44_im", "kappa11_im", "kappa33_im")
OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + " must be a positive Python integer")
    return value


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return value


def _eta(value):
    if isinstance(value, torch.Tensor):
        if value.ndim or value.requires_grad or value.is_complex():
            raise ValueError(
                "eta_nonaffine is a fixed real scalar control, not an inferred parameter"
            )
        value = float(value.item())
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("eta_nonaffine must be finite and nonnegative")
    return value


class AxisymmetricDirectMessageGraph(torch.nn.Module):
    """Local messages for full fields [batch,nnode,(ur,uz,phi)].

    Public geometry buffers: coordinates[n,2], volumes[n], node_mass_kg[n],
    axis_nodes, bottom_nodes, top_nodes and outer_nodes.  ``n==nnode`` and
    ``ndof==3*n``.  All state tensors must be complex128 on the graph device.
    Materials may be shared [10]/[7] or independent [B,10]/[B,7] rows, allowing
    reciprocal parameter contractions without cross-frequency derivatives.

    Call ``validate_parameters`` once per material change BEFORE a solve.  The
    hot action checks shape/dtype/device but does not synchronize material-domain
    checks on every iteration.  Unsupported real star domains are rejected by
    validation rather than stabilized/fitted away.
    """

    def __init__(self, nr=48, nz=8, radius=0.0125, height=0.00202, rho=7862.0, device="cpu"):
        super().__init__()
        self.nr, self.nz = _integer(nr, "nr"), _integer(nz, "nz")
        self.radius, self.height, self.rho = (
            _positive(radius, "radius"),
            _positive(height, "height"),
            _positive(rho, "rho"),
        )
        self.dr, self.dz = self.radius / self.nr, self.height / self.nz
        self.n = self.nnode = (self.nr + 1) * (self.nz + 1)
        self.ndof = 3 * self.n
        length_squared = self.dr * self.dr + self.dz * self.dz
        self.p, self.q = self.dr * self.dr / length_squared, self.dz * self.dz / length_squared
        self.tau = math.sqrt(self.p * self.q)
        coordinates, volumes = [], []
        for ir in range(self.nr + 1):
            r = ir * self.dr
            rlo, rhi = max(0.0, r - self.dr / 2), min(self.radius, r + self.dr / 2)
            for iz in range(self.nz + 1):
                z = iz * self.dz
                zlo, zhi = max(0.0, z - self.dz / 2), min(self.height, z + self.dz / 2)
                coordinates.append((r, z))
                volumes.append(math.pi * (rhi - rlo) * (rhi + rlo) * (zhi - zlo))
        directions, lengths = [], []
        for ox, oy in OFFSETS:
            dx, dz = ox * self.dr, oy * self.dz
            distance = math.hypot(dx, dz)
            directions.append((dx / distance, dz / distance))
            lengths.append(distance)
        entry_node, entry_bond, entry_coeff = [], [], []
        reflected, virtual, real_bonds = 0, 0, 0

        def node(ir, iz):
            return ir * (self.nz + 1) + iz

        def add(coefficients, index, value):
            previous = coefficients.setdefault(index, [0.0, 0.0, 0.0])
            for channel in range(3):
                previous[channel] += value[channel]

        def derivative(ir, iz, axis):
            # Component-aware scalar derivative incidence, not a gradient matrix.
            coeff = {}
            center = node(ir, iz)
            if axis == 0:
                if ir == 0:
                    add(coeff, center, [-1 / self.dr, 0.0, 0.0])
                    add(coeff, node(1, iz), [1 / self.dr, 0.0, 0.0])
                elif ir == self.nr:
                    add(coeff, center, [1 / self.dr] * 3)
                    add(coeff, node(ir - 1, iz), [-1 / self.dr] * 3)
                else:
                    add(coeff, node(ir + 1, iz), [0.5 / self.dr] * 3)
                    add(coeff, node(ir - 1, iz), [-0.5 / self.dr] * 3)
            elif iz == 0:
                add(coeff, center, [-1 / self.dz] * 3)
                add(coeff, node(ir, 1), [1 / self.dz] * 3)
            elif iz == self.nz:
                add(coeff, center, [1 / self.dz] * 3)
                add(coeff, node(ir, iz - 1), [-1 / self.dz] * 3)
            else:
                add(coeff, node(ir, iz + 1), [0.5 / self.dz] * 3)
                add(coeff, node(ir, iz - 1), [-0.5 / self.dz] * 3)
            return coeff

        for ir in range(self.nr + 1):
            for iz in range(self.nz + 1):
                center = node(ir, iz)
                drule, zrule = derivative(ir, iz, 0), derivative(ir, iz, 1)
                for direction, (ox, oz) in enumerate(OFFSETS):
                    rr, zz = ir + ox, iz + oz
                    coefficients = {}
                    if 0 <= rr <= self.nr and 0 <= zz <= self.nz:
                        add(coefficients, node(rr, zz), [1.0, 1.0, 1.0])
                        add(coefficients, center, [-1.0, -1.0, -1.0])
                        real_bonds += 1
                    elif rr < 0 and 0 <= zz <= self.nz:
                        add(coefficients, node(-rr, zz), [-1.0, 1.0, 1.0])
                        add(coefficients, center, [-1.0, -1.0, -1.0])
                        reflected += 1
                    else:
                        for rules, factor in ((drule, ox * self.dr), (zrule, oz * self.dz)):
                            for index, values in rules.items():
                                add(coefficients, index, [factor * v for v in values])
                        virtual += 1
                    bond = 8 * center + direction
                    for index, values in sorted(coefficients.items()):
                        if any(value != 0.0 for value in values):
                            entry_node.append(index)
                            entry_bond.append(bond)
                            entry_coeff.append(values)
        tensors = dict(
            coordinates=torch.tensor(coordinates, dtype=torch.float64, device=device),
            volumes=torch.tensor(volumes, dtype=torch.float64, device=device),
            coefficient_node=torch.tensor(entry_node, dtype=torch.long, device=device),
            coefficient_bond=torch.tensor(entry_bond, dtype=torch.long, device=device),
            coefficient_value=torch.tensor(entry_coeff, dtype=torch.float64, device=device),
            bond_direction=torch.tensor(directions, dtype=torch.float64, device=device),
            bond_length=torch.tensor(lengths, dtype=torch.float64, device=device),
            axis_nodes=torch.tensor(
                [node(0, j) for j in range(self.nz + 1)], dtype=torch.long, device=device
            ),
            bottom_nodes=torch.tensor(
                [node(i, 0) for i in range(self.nr + 1)], dtype=torch.long, device=device
            ),
            top_nodes=torch.tensor(
                [node(i, self.nz) for i in range(self.nr + 1)], dtype=torch.long, device=device
            ),
            outer_nodes=torch.tensor(
                [node(self.nr, j) for j in range(self.nz + 1)], dtype=torch.long, device=device
            ),
        )
        tensors["node_mass_kg"] = self.rho * tensors["volumes"]
        radius_values = tensors["coordinates"][:, 0]
        tensors["axis_mask"] = radius_values == 0
        tensors["inverse_radius"] = torch.where(
            radius_values > 0,
            torch.reciprocal(radius_values.clamp_min(self.dr)),
            torch.zeros_like(radius_values),
        )
        for name, value in tensors.items():
            self.register_buffer(name, value)
        self.metadata = dict(
            model="axisymmetric energy-projected direct eight-bond star adaptation",
            original_published_LPM_reproduction=False,
            old_GMLS_or_FEM_used=False,
            particles=self.n,
            directions_per_star=8,
            nr=self.nr,
            nz=self.nz,
            radius_m=self.radius,
            height_m=self.height,
            density_kg_m3=self.rho,
            dr_m=self.dr,
            dz_m=self.dz,
            p=self.p,
            q=self.q,
            real_names=list(REAL_NAMES),
            imag_names=list(IMAG_NAMES),
            units="C Pa; e C/m^2; kappa absolute F/m; displacement m; phi V",
            volume_rule="exact positive truncated dual annular volumes including axis",
            physical_inside_directions=real_bonds,
            axis_reflected_directions=reflected,
            same_side_virtual_directions=virtual,
            axis_parity="ur odd; uz/phi even; missing-z corners use regular affine completion",
            hoop_rule="ur/r off axis; projected er on axis",
            projection="real-direct-star energy-orthogonal meridional affine projection",
            mechanical_nonaffine_remainder_retained=True,
            coupling="new projected strain and axial electric gradients; not full old planar G",
            electric_nonaffine_remainder="axial opposite-bond sum squares",
            phasor="exp(+i*omega*t)",
            mass="positive nodal rho*volume; no electric mass",
            constraints_in_module=False,
            axis_ur_must_be_fixed_zero_by_caller=True,
            frequency_zero_allowed=True,
            global_material_matrix_constructed=False,
            sparse_backend_used=False,
            autograd_fullfield_action_used=False,
            trained_network=False,
            patch_or_convergence_claimed=False,
        )

    def _material_rows(self, values, size, batch, name):
        if (
            not isinstance(values, torch.Tensor)
            or values.dtype != torch.float64
            or values.device != self.coordinates.device
        ):
            raise TypeError(name + " must be real float64 on the graph device")
        if values.shape == (size,):
            return values[None, :].expand(batch, size)
        if values.shape == (batch, size):
            return values
        raise ValueError(name + " has wrong shared/batched material shape")

    def _materials(self, real, imag, batch):
        real = self._material_rows(real, 10, batch, "real")
        if imag is None:
            imag = real.new_zeros((batch, 7))
        else:
            imag = self._material_rows(imag, 7, batch, "imag")
        return real, imag

    def _star_coefficients(self, real):
        c11, _, c13, c33, c44 = real[:, :5].unbind(dim=1)
        ad = c44 / (4 * self.p * self.q)
        ba = (c13 - c44) / (4 + 32 * self.p * self.q)
        bd = 2 * ba
        ee = 2 * ad + 8 * bd
        ar = 0.5 * c11 - 2 * ba - ee * self.p * self.p
        az = 0.5 * c33 - 2 * ba - ee * self.q * self.q
        brh, brv, brd = ar + 2 * ba, 2 * ba, 0.5 * ee * self.p
        bzh, bzv, bzd = 2 * ba, az + 2 * ba, 0.5 * ee * self.q
        determinant = c11 * c33 - c13.square()
        prh, prv, prd = (
            (c33 * brh - c13 * bzh) / determinant,
            (c33 * brv - c13 * bzv) / determinant,
            (c33 * brd - c13 * bzd) / determinant,
        )
        pzh, pzv, pzd = (
            (c11 * bzh - c13 * brh) / determinant,
            (c11 * bzv - c13 * brv) / determinant,
            (c11 * bzd - c13 * brd) / determinant,
        )
        pr = torch.stack((prh, prh, prv, prv, prd, prd, prd, prd), dim=1)
        pz = torch.stack((pzh, pzh, pzv, pzv, pzd, pzd, pzd, pzd), dim=1)
        zero = torch.zeros_like(c11)
        shear = torch.ones_like(c11) / (4 * self.tau)
        pg = torch.stack((zero, zero, zero, zero, shear, -shear, shear, -shear), dim=1)
        return ar, az, ad, ba, bd, pr, pz, pg

    def validate_parameters(self, real, imag=None, eta_nonaffine=0.0):
        """Explicit scalar/vector-domain checks; no eigenproblem or linear solve.

        For real direct-star PSD, positive Ar/Az/Ad are required.  Negative
        collective coefficients are allowed only when their exact rank-one
        criteria pass.  Full real TI C must be SPD, loss TI Cimag PSD.  A small
        relative tolerance allows rounding at loss-semidefinite boundaries,
        not a silently repaired/passivity-projected production material.
        """
        _eta(eta_nonaffine)
        batch = (
            real.shape[0]
            if isinstance(real, torch.Tensor) and real.ndim == 2
            else (imag.shape[0] if isinstance(imag, torch.Tensor) and imag.ndim == 2 else 1)
        )
        real, imag = self._materials(real, imag, batch)
        if not bool(torch.isfinite(real).all().item()) or not bool(
            torch.isfinite(imag).all().item()
        ):
            raise ValueError("Materials must be finite")
        a, b, c, d, s = real[:, :5].unbind(dim=1)
        if not bool(
            torch.all(
                (a - b > 0)
                & (a + b > 0)
                & (d > 0)
                & (s > 0)
                & ((a + b) * d - 2 * c.square() > 0)
                & (real[:, 8] > 0)
                & (real[:, 9] > 0)
            ).item()
        ):
            raise ValueError(
                "Real full TI elasticity and dielectric entries must be positive definite"
            )
        ar, az, ad, ba, bd, _, _, _ = self._star_coefficients(real)
        if not bool(torch.all((ar > 0) & (az > 0) & (ad > 0)).item()):
            raise ValueError("Rectangle/material lies outside positive Ar/Az/Ad direct-star domain")
        axial_criterion = 1 + ba * (2 / ar + 2 / az)
        diagonal_criterion = 1 + 4 * bd / ad
        if not bool(torch.all((axial_criterion >= 0) & (diagonal_criterion >= 0)).item()):
            raise ValueError("Direct-star collective metric is not PSD")
        a, b, c, d, s = imag[:, :5].unbind(dim=1)
        magnitude = imag[:, :5].abs().amax(dim=1)
        tol = 64 * torch.finfo(torch.float64).eps * magnitude
        determinant_tolerance = 64 * torch.finfo(torch.float64).eps * magnitude.square()
        if not bool(
            torch.all(
                (a - b >= -tol)
                & (a + b >= -tol)
                & (d >= -tol)
                & (s >= -tol)
                & ((a + b) * d - 2 * c.square() >= -determinant_tolerance)
                & (imag[:, 5] <= 0)
                & (imag[:, 6] <= 0)
            ).item()
        ):
            raise ValueError(
                "Full TI imaginary elasticity must be PSD; imaginary dielectric must be nonpositive"
            )
        return dict(
            real_TI_SPD=True,
            real_direct_star_PSD=True,
            loss_TI_PSD_with_roundoff_tolerance=True,
            dielectric_imag_nonpositive=True,
            eta_nonaffine=float(_eta(eta_nonaffine)),
            production_coefficients_repaired=False,
        )

    def _inputs(self, state, real, imag, frequencies, eta_nonaffine):
        if (
            not isinstance(state, torch.Tensor)
            or state.ndim != 3
            or state.shape[1:] != (self.n, 3)
            or state.shape[0] < 1
            or state.dtype != torch.complex128
            or state.device != self.coordinates.device
        ):
            raise ValueError("state must be complex128 [batch,nnode,3] on graph device")
        real, imag = self._materials(real, imag, state.shape[0])
        if isinstance(frequencies, torch.Tensor) and frequencies.requires_grad:
            raise ValueError("Frequency is a fixed input, not an inferred parameter")
        if torch.as_tensor(frequencies, device=state.device).is_complex():
            raise ValueError("Frequency must be real Hz, not complex")
        frequency = torch.as_tensor(frequencies, dtype=torch.float64, device=state.device)
        if frequency.ndim == 0 and state.shape[0] == 1:
            frequency = frequency[None]
        if frequency.shape != (state.shape[0],):
            raise ValueError("Frequency must have one real Hz entry per batch row")
        # Domain checks are intentionally explicit outside hot iteration, but
        # frequency sign/finite must never silently reinterpret a dynamic input.
        if not bool(torch.all(torch.isfinite(frequency) & (frequency >= 0)).item()):
            raise ValueError("Frequency must be finite nonnegative Hz")
        return real, imag, frequency, _eta(eta_nonaffine)

    def _features(self, state, real):
        batch = state.shape[0]
        difference = state.new_zeros((batch, 8 * self.n, 3)).index_add(
            1,
            self.coefficient_bond,
            self.coefficient_value[None, :, :] * state[:, self.coefficient_node, :],
        )
        difference = difference.reshape(batch, self.n, 8, 3)
        stretch = (difference[:, :, :, :2] * self.bond_direction[None, None, :, :]).sum(
            dim=3
        ) / self.bond_length[None, None, :]
        axial_phi = difference[:, :, :4, 2] / self.bond_length[None, None, :4]
        gr, gz = (
            0.5 * (axial_phi[:, :, 0] - axial_phi[:, :, 1]),
            0.5 * (axial_phi[:, :, 2] - axial_phi[:, :, 3]),
        )
        ar, az, ad, ba, bd, pr, pz, pg = self._star_coefficients(real)
        er = (pr[:, None, :] * stretch).sum(dim=2)
        ez = (pz[:, None, :] * stretch).sum(dim=2)
        gamma = (pg[:, None, :] * stretch).sum(dim=2)
        hoop = torch.where(
            self.axis_mask[None, :], er, state[:, :, 0] * self.inverse_radius[None, :]
        )
        return dict(
            stretch=stretch,
            axial_phi=axial_phi,
            er=er,
            et=hoop,
            ez=ez,
            gamma=gamma,
            gr=gr,
            gz=gz,
            ar=ar,
            az=az,
            ad=ad,
            ba=ba,
            bd=bd,
            pr=pr,
            pz=pz,
            pg=pg,
        )

    @staticmethod
    def _native_density(features, hermitian=False):
        s = features["stretch"]
        squared = (lambda x: x.abs().square()) if hermitian else (lambda x: x.square())
        axis_sum, diag_sum = s[:, :, :4].sum(dim=2), s[:, :, 4:].sum(dim=2)
        return (
            0.5 * features["ar"][:, None] * squared(s[:, :, :2]).sum(dim=2)
            + 0.5 * features["az"][:, None] * squared(s[:, :, 2:4]).sum(dim=2)
            + 0.5 * features["ad"][:, None] * squared(s[:, :, 4:]).sum(dim=2)
            + 0.5 * features["ba"][:, None] * squared(axis_sum)
            + 0.5 * features["bd"][:, None] * squared(diag_sum)
        )

    @staticmethod
    def _plane_density(features, real, hermitian=False):
        er, ez, g = features["er"], features["ez"], features["gamma"]
        c11, c13, c33, c44 = real[:, 0, None], real[:, 2, None], real[:, 3, None], real[:, 4, None]
        if hermitian:
            return (
                0.5 * c11 * er.abs().square()
                + c13 * (er.conj() * ez).real
                + 0.5 * c33 * ez.abs().square()
                + 0.5 * c44 * g.abs().square()
            )
        return (
            0.5 * c11 * er.square()
            + c13 * er * ez
            + 0.5 * c33 * ez.square()
            + 0.5 * c44 * g.square()
        )

    def constitutive_fields(self, state, real):
        """Projected strain/gradient fields; no constraint or equilibrium solve.

        ``nonaffine_density`` uses nonconjugate products and can be complex.
        Its Hermitian counterpart is available as ``nonaffine_H_density``.
        """
        if (
            not isinstance(state, torch.Tensor)
            or state.ndim != 3
            or state.shape[1:] != (self.n, 3)
            or state.dtype != torch.complex128
            or state.device != self.coordinates.device
        ):
            raise ValueError("Expected complex128 physical fields [B,n,3]")
        real = self._material_rows(real, 10, state.shape[0], "real")
        features = self._features(state, real)
        result = {
            key: features[key]
            for key in ("er", "et", "ez", "gamma", "gr", "gz", "stretch", "axial_phi")
        }
        result["nonaffine_density"] = self._native_density(features) - self._plane_density(
            features, real
        )
        result["nonaffine_H_density"] = self._native_density(features, True) - self._plane_density(
            features, real, True
        )
        return result

    def action(self, state, real, imag=None, frequencies=0.0, eta_nonaffine=0.0):
        """Closed-form local derivative plus elementwise mechanical inertia.

        The complex transpose-symmetric action is the derivative of ``energy``.
        No autograd field-gradient call is used.  Material derivatives MAY be
        taken through these scalar formulas by the caller's reciprocal layer.
        """
        real, imag, frequency, eta = self._inputs(state, real, imag, frequencies, eta_nonaffine)
        f = self._features(state, real)
        s, er, et, ez, g, gr, gz = (
            f[key] for key in ("stretch", "er", "et", "ez", "gamma", "gr", "gz")
        )
        c11, c12, c13, c33, c44 = (real[:, j, None] + 1j * imag[:, j, None] for j in range(5))
        e15, e31, e33 = (real[:, j, None] for j in (5, 6, 7))
        kr, kz = real[:, 8, None] + 1j * imag[:, 5, None], real[:, 9, None] + 1j * imag[:, 6, None]
        axis_sum, diag_sum = (
            s[:, :, :4].sum(dim=2, keepdim=True),
            s[:, :, 4:].sum(dim=2, keepdim=True),
        )
        native_force = torch.cat(
            (
                f["ar"][:, None, None] * s[:, :, :2] + f["ba"][:, None, None] * axis_sum,
                f["az"][:, None, None] * s[:, :, 2:4] + f["ba"][:, None, None] * axis_sum,
                f["ad"][:, None, None] * s[:, :, 4:] + f["bd"][:, None, None] * diag_sum,
            ),
            dim=2,
        )
        plane_r = real[:, 0, None] * er + real[:, 2, None] * ez
        plane_z = real[:, 2, None] * er + real[:, 3, None] * ez
        plane_g = real[:, 4, None] * g
        remainder_force = (
            native_force
            - f["pr"][:, None, :] * plane_r[:, :, None]
            - f["pz"][:, None, :] * plane_z[:, :, None]
            - f["pg"][:, None, :] * plane_g[:, :, None]
        )
        sigma_r = c11 * er + c12 * et + c13 * ez + e31 * gz
        sigma_t = c12 * er + c11 * et + c13 * ez + e31 * gz
        sigma_z = c13 * (er + et) + c33 * ez + e33 * gz
        tau_rz = c44 * g + e15 * gr
        stretch_force = (
            (1 + 1j * eta) * remainder_force
            + f["pr"][:, None, :] * (sigma_r + self.axis_mask[None, :] * sigma_t)[:, :, None]
            + f["pz"][:, None, :] * sigma_z[:, :, None]
            + f["pg"][:, None, :] * tau_rz[:, :, None]
        )
        phi_r = e15 * g - kr * gr
        phi_z = e31 * (er + et) + e33 * ez - kz * gz
        sum_r = f["axial_phi"][:, :, 0] + f["axial_phi"][:, :, 1]
        sum_z = f["axial_phi"][:, :, 2] + f["axial_phi"][:, :, 3]
        potential_force = torch.stack(
            (
                0.5 * phi_r - 0.25 * kr * sum_r,
                -0.5 * phi_r - 0.25 * kr * sum_r,
                0.5 * phi_z - 0.25 * kz * sum_z,
                -0.5 * phi_z - 0.25 * kz * sum_z,
            ),
            dim=2,
        )
        potential_force = torch.cat((potential_force, torch.zeros_like(potential_force)), dim=2)
        mechanical = (
            stretch_force[:, :, :, None]
            * self.bond_direction[None, None, :, :]
            / self.bond_length[None, None, :, None]
        )
        response = torch.cat(
            (mechanical, potential_force[:, :, :, None] / self.bond_length[None, None, :, None]),
            dim=3,
        )
        response = (response * self.volumes[None, :, None, None]).reshape(
            state.shape[0], 8 * self.n, 3
        )
        output = torch.zeros_like(state).index_add(
            1,
            self.coefficient_node,
            self.coefficient_value[None, :, :] * response[:, self.coefficient_bond, :],
        )
        hoop_nodal = self.volumes[None, :] * sigma_t * self.inverse_radius[None, :]
        omega_squared = (2 * math.pi * frequency).square()
        inertia = omega_squared[:, None] * self.node_mass_kg[None, :]
        return torch.stack(
            (
                output[:, :, 0] + hoop_nodal - inertia * state[:, :, 0],
                output[:, :, 1] - inertia * state[:, :, 1],
                output[:, :, 2],
            ),
            dim=2,
        )

    def energy(self, state, real, imag=None, frequencies=0.0, eta_nonaffine=0.0):
        """Complex nonconjugate dynamic functional [B] in joules."""
        real, imag, frequency, eta = self._inputs(state, real, imag, frequencies, eta_nonaffine)
        f = self._features(state, real)
        er, et, ez, g, gr, gz = (f[key] for key in ("er", "et", "ez", "gamma", "gr", "gz"))
        c11, c12, c13, c33, c44 = (real[:, j, None] + 1j * imag[:, j, None] for j in range(5))
        e15, e31, e33 = (real[:, j, None] for j in (5, 6, 7))
        kr, kz = real[:, 8, None] + 1j * imag[:, 5, None], real[:, 9, None] + 1j * imag[:, 6, None]
        remainder = self._native_density(f) - self._plane_density(f, real)
        density = (
            (1 + 1j * eta) * remainder
            + 0.5 * c11 * (er.square() + et.square())
            + c12 * er * et
            + c13 * (er + et) * ez
            + 0.5 * c33 * ez.square()
            + 0.5 * c44 * g.square()
            + e15 * g * gr
            + (e31 * (er + et) + e33 * ez) * gz
            - 0.5 * kr * gr.square()
            - 0.5 * kz * gz.square()
            - 0.125 * kr * (f["axial_phi"][:, :, 0] + f["axial_phi"][:, :, 1]).square()
            - 0.125 * kz * (f["axial_phi"][:, :, 2] + f["axial_phi"][:, :, 3]).square()
        )
        omega_squared = (2 * math.pi * frequency).square()
        kinetic = (
            0.5
            * omega_squared
            * (self.node_mass_kg[None, :] * state[:, :, :2].square().sum(dim=2)).sum(dim=1)
        )
        return (self.volumes[None, :] * density).sum(dim=1) - kinetic

    def dissipated_power(self, state, real, imag=None, frequencies=0.0, eta_nonaffine=0.0):
        """Conjugated mean loss powers [B], for peak phasors, not stored energy.

        No loss contribution is clipped.  Negative outputs expose a domain or
        numerical problem rather than silently projecting a fitted material.
        """
        real, imag, frequency, eta = self._inputs(state, real, imag, frequencies, eta_nonaffine)
        f = self._features(state, real)
        er, et, ez, g, gr, gz = (f[key] for key in ("er", "et", "ez", "gamma", "gr", "gz"))
        a, b, c, d, s = (imag[:, j, None] for j in range(5))
        mechanical = (
            0.5 * a * (er.abs().square() + et.abs().square())
            + b * (er.conj() * et).real
            + c * ((er + et).conj() * ez).real
            + 0.5 * d * ez.abs().square()
            + 0.5 * s * g.abs().square()
        )
        remainder = self._native_density(f, True) - self._plane_density(f, real, True)
        kr, kz = imag[:, 5, None], imag[:, 6, None]
        electric = (
            -0.5 * kr * gr.abs().square()
            - 0.5 * kz * gz.abs().square()
            - 0.125 * kr * (f["axial_phi"][:, :, 0] + f["axial_phi"][:, :, 1]).abs().square()
            - 0.125 * kz * (f["axial_phi"][:, :, 2] + f["axial_phi"][:, :, 3]).abs().square()
        )
        omega = 2 * math.pi * frequency
        pm = omega * (self.volumes[None, :] * mechanical).sum(dim=1)
        pn = omega * eta * (self.volumes[None, :] * remainder).sum(dim=1)
        pe = omega * (self.volumes[None, :] * electric).sum(dim=1)
        return dict(mechanical_W=pm, nonaffine_W=pn, dielectric_W=pe, total_W=pm + pn + pe)
