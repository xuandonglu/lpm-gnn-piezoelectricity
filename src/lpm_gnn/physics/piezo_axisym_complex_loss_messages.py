"""Complex-piezoelectric-loss axisymmetric particle messages.

This class adds imaginary electromechanical coupling to the base real model
with mechanical and dielectric losses.

real[10] (absolute SI, unchanged):
 [C11,C12,C13,C33,C44,e15,e31,e33,kappaR,kappaZ].
imag[10] (absolute SI, exp(+i*omega*t)):
 [C11im,C12im,C13im,C33im,C44im,e15im,e31im,e33im,kappaRim,kappaZim].
The reduced imag[7]=[C5im,kappaRim,kappaZim] is also accepted, with eimag=0.
Both shared and independent batch materials are supported as in the base graph.

The extra density is i*(f15*gamma*gr+[f31*(er+et)+f33*ez]*gz).
The extra force uses explicit local chain-rule gather/scatter messages in
BOTH reciprocal directions; it is not autograd of a full-field energy.
The conjugated mean-power cross term can be negative, but the complete loss
block must be PSD.  No material or power contribution is clipped/repaired.

For nonzero normal coupling the current validator requires the 2x2 normal
elastic-loss block to be strictly positive definite.  Rank-deficient normal
blocks with nonzero normal eimag are explicitly REJECTED (even when a special
range-compatible semidefinite limit could be passive).  Zero normal coupling
retains the base semidefinite domain.  Similarly, nonzero e15im requires
C44im>0.  No epsilon is inserted into a singular Schur denominator.

This is an unvalidated candidate module, not a demonstrated fit improvement,
learned network or identified intrinsic loss mechanism.  No solve is provided
here.  The old AxisymmetricMessageProblem accepts imag7 ONLY and must NOT be
used with imag10 until a separately authorized adapter is implemented/tested.
"""

import math

import torch

from .piezo_axisym_direct_messages import REAL_NAMES, AxisymmetricDirectMessageGraph

IMAG_NAMES = (
    "c11_im",
    "c12_im",
    "c13_im",
    "c33_im",
    "c44_im",
    "e15_im",
    "e31_im",
    "e33_im",
    "kappa11_im",
    "kappa33_im",
)


class AxisymmetricComplexLossMessageGraph(AxisymmetricDirectMessageGraph):
    """Same geometry/physical fields, with explicit reciprocal complex e loss.

    ``action``, ``energy``, ``dissipated_power`` and ``validate_parameters``
    have the same argument names/defaults as the base graph, but imag accepts
    either the old seven or the documented new ten entries.  All coefficients
    are REAL float64 tensors; e*=e_real+i*e_imag is built internally.

    The parent's real constitutive projection/axis/volume/mass rules are
    inherited unchanged.  Parameter derivatives through explicit messages are
    allowed; no derivative through the field solver is implemented here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = dict(
            self.metadata,
            model="axisymmetric projected direct-star adaptation with explicit complex piezo loss",
            real_names=list(REAL_NAMES),
            imag_names=list(IMAG_NAMES),
            old_imag7_accepted_as_zero_piezo_imag=True,
            real_model_geometry_mass_projection_unchanged=True,
            imaginary_piezo_coupling="explicit reciprocal node/bond/star messages",
            extra_energy="i*(e15im*gamma*gr+[e31im*(er+et)+e33im*ez]*gz)",
            extra_mean_power="omega*integral Re(conj(strain)*eimag*gradphi)",
            full_loss_passivity_check="TI normal/shear scalar Schur; no inverse/clip/repair",
            nonzero_coupling_rank_deficient_normal_loss_domain="explicitly rejected",
            generic_old_imag7_solver_compatible_with_imag10=False,
            numerical_or_experimental_improvement_claimed=False,
        )

    def _split_imag(self, real, imag, batch):
        """Return real rows, base imag7 rows and eimag3 rows, with AD intact."""
        real_rows = self._material_rows(real, 10, batch, "real")
        if imag is None:
            base = real_rows.new_zeros((batch, 7))
            piezo = real_rows.new_zeros((batch, 3))
        else:
            if not isinstance(imag, torch.Tensor) or imag.ndim not in (1, 2):
                raise ValueError(
                    "imag must be shared/batched real-double seven or ten coefficients"
                )
            size = imag.shape[-1]
            if size not in (7, 10):
                raise ValueError("imag has seven legacy entries or ten complex-loss entries")
            values = self._material_rows(imag, size, batch, "imag")
            if size == 7:
                base = values
                piezo = values.new_zeros((batch, 3))
            else:
                base = torch.cat((values[:, :5], values[:, 8:]), dim=1)
                piezo = values[:, 5:8]
        return real_rows, base, piezo

    def _state_materials(self, state, real, imag):
        if (
            not isinstance(state, torch.Tensor)
            or state.ndim != 3
            or state.shape[1:] != (self.n, 3)
            or state.shape[0] < 1
            or state.dtype != torch.complex128
            or state.device != self.coordinates.device
        ):
            raise ValueError("Expected physical complex128 state[B,nnode,3] on graph device")
        return self._split_imag(real, imag, state.shape[0])

    def validate_parameters(self, real, imag=None, eta_nonaffine=0.0):
        """Validate base material and the COMPLETE coupled imaginary loss form.

        The normal Schur requirement for Delta=(a+b)d-2c^2>0 is evaluated as
          2*f31^2/A + A*(f33-2c*f31/A)^2/Delta, A=a+b,
        algebraically identical to
          [2d*f31^2-4c*f31*f33+A*f33^2]/Delta,
        but without cancellation between three large numerator terms.
        Degenerate nonzero-coupling sectors are refused explicitly, not repaired.
        """
        batch = (
            real.shape[0]
            if isinstance(real, torch.Tensor) and real.ndim == 2
            else (imag.shape[0] if isinstance(imag, torch.Tensor) and imag.ndim == 2 else 1)
        )
        real_rows, base, piezo = self._split_imag(real, imag, batch)
        certificate = super().validate_parameters(real_rows, base, eta_nonaffine)
        if not bool(torch.isfinite(piezo).all().item()):
            raise ValueError("Imaginary piezoelectric coefficients must be finite")
        a, b, c, d, s = base[:, :5].unbind(dim=1)
        f15, f31, f33 = piezo.unbind(dim=1)
        dr, dz = -base[:, 5], -base[:, 6]
        normal_active = (f31 != 0) | (f33 != 0)
        shear_active = f15 != 0
        normal_required = real_rows.new_zeros(batch)
        shear_required = real_rows.new_zeros(batch)
        if bool(normal_active.any().item()):
            aa = (a + b)[normal_active]
            cc, dd = c[normal_active], d[normal_active]
            determinant = aa * dd - 2 * cc.square()
            if not bool(torch.all((aa > 0) & (dd > 0) & (determinant > 0)).item()):
                raise ValueError(
                    "Nonzero normal eimag requires strictly positive normal loss block; rank-deficient coupled domain is not supported"
                )
            first, second = f31[normal_active], f33[normal_active]
            needed = (
                2 * first.square() / aa + aa * (second - 2 * cc * first / aa).square() / determinant
            )
            actual = dz[normal_active]
            tolerance = (
                64 * torch.finfo(torch.float64).eps * torch.maximum(actual.abs(), needed.abs())
            )
            if not bool(torch.all(torch.isfinite(needed) & (actual - needed >= -tolerance)).item()):
                raise ValueError("Normal coupled imaginary loss fails full PSD Schur condition")
            normal_required[normal_active] = needed
        if bool(shear_active.any().item()):
            ss = s[shear_active]
            if not bool(torch.all(ss > 0).item()):
                raise ValueError(
                    "Nonzero e15imag requires positive C44imag; no singular shear regularization"
                )
            needed = f15[shear_active].square() / ss
            actual = dr[shear_active]
            tolerance = (
                64 * torch.finfo(torch.float64).eps * torch.maximum(actual.abs(), needed.abs())
            )
            if not bool(torch.all(torch.isfinite(needed) & (actual - needed >= -tolerance)).item()):
                raise ValueError("Shear coupled imaginary loss fails full PSD Schur condition")
            shear_required[shear_active] = needed
        certificate = dict(certificate)
        certificate.update(
            complete_complex_electromechanical_loss_PSD_with_roundoff_tolerance=True,
            normal_piezo_loss_required_dielectric_F_per_m=normal_required.detach().cpu().tolist(),
            shear_piezo_loss_required_dielectric_F_per_m=shear_required.detach().cpu().tolist(),
            nonzero_coupling_semidefinite_normal_limit="rejected, not falsely inverted",
            imaginary_parameter_order=list(IMAG_NAMES),
            legacy_imag7_input=False if imag is None else imag.shape[-1] == 7,
            piezoelectric_imaginary_coefficients_all_zero=bool(torch.all(piezo == 0).item()),
            production_coefficients_repaired=False,
        )
        return certificate

    def action(self, state, real, imag=None, frequencies=0.0, eta_nonaffine=0.0):
        real_rows, base, piezo = self._state_materials(state, real, imag)
        original = super().action(state, real_rows, base, frequencies, eta_nonaffine)
        # Deliberately do not skip exactly-zero eimag: its derivative at beta=0
        # must remain available to the later reciprocal parameter contraction.
        f = self._features(state, real_rows)
        f15, f31, f33 = (piezo[:, j, None] for j in range(3))
        sigma_t = 1j * f31 * f["gz"]
        sigma_r = sigma_t
        sigma_z = 1j * f33 * f["gz"]
        shear = 1j * f15 * f["gr"]
        stretch_force = (
            f["pr"][:, None, :] * (sigma_r + self.axis_mask[None, :] * sigma_t)[:, :, None]
            + f["pz"][:, None, :] * sigma_z[:, :, None]
            + f["pg"][:, None, :] * shear[:, :, None]
        )
        phi_r = 1j * f15 * f["gamma"]
        phi_z = 1j * (f31 * (f["er"] + f["et"]) + f33 * f["ez"])
        phi_force = torch.stack((0.5 * phi_r, -0.5 * phi_r, 0.5 * phi_z, -0.5 * phi_z), dim=2)
        phi_force = torch.cat((phi_force, torch.zeros_like(phi_force)), dim=2)
        mechanical = (
            stretch_force[:, :, :, None]
            * self.bond_direction[None, None, :, :]
            / self.bond_length[None, None, :, None]
        )
        response = torch.cat(
            (mechanical, phi_force[:, :, :, None] / self.bond_length[None, None, :, None]), dim=3
        )
        response = (response * self.volumes[None, :, None, None]).reshape(
            state.shape[0], 8 * self.n, 3
        )
        extra = torch.zeros_like(state).index_add(
            1,
            self.coefficient_node,
            self.coefficient_value[None, :, :] * response[:, self.coefficient_bond, :],
        )
        hoop_nodal = self.volumes[None, :] * sigma_t * self.inverse_radius[None, :]
        extra = torch.stack((extra[:, :, 0] + hoop_nodal, extra[:, :, 1], extra[:, :, 2]), dim=2)
        # Inertia and all original material terms are already in original.
        return original + extra

    def energy(self, state, real, imag=None, frequencies=0.0, eta_nonaffine=0.0):
        real_rows, base, piezo = self._state_materials(state, real, imag)
        original = super().energy(state, real_rows, base, frequencies, eta_nonaffine)
        f = self._features(state, real_rows)
        f15, f31, f33 = (piezo[:, j, None] for j in range(3))
        coupling = (
            f15 * f["gamma"] * f["gr"] + (f31 * (f["er"] + f["et"]) + f33 * f["ez"]) * f["gz"]
        )
        return original + 1j * (self.volumes[None, :] * coupling).sum(dim=1)

    def dissipated_power(self, state, real, imag=None, frequencies=0.0, eta_nonaffine=0.0):
        """Peak-phasor mean powers, including the SIGNED piezo cross term."""
        real_rows, base, piezo = self._state_materials(state, real, imag)
        result = dict(super().dissipated_power(state, real_rows, base, frequencies, eta_nonaffine))
        # Base input validation supplies the real, fixed, nonnegative frequencies.
        _, _, frequency, _ = self._inputs(state, real_rows, base, frequencies, eta_nonaffine)
        f = self._features(state, real_rows)
        f15, f31, f33 = (piezo[:, j, None] for j in range(3))
        cross = (
            f15 * f["gamma"].conj() * f["gr"]
            + (f31 * (f["er"] + f["et"]).conj() + f33 * f["ez"].conj()) * f["gz"]
        ).real
        power = 2 * math.pi * frequency * (self.volumes[None, :] * cross).sum(dim=1)
        result["piezoelectric_cross_W"] = power
        result["total_W"] = result["total_W"] + power
        return result
