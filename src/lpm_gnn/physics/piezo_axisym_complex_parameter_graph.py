"""Six-coordinate passive shared complex-piezo-loss material/port adapter.

The first five coordinates specify shared tensor scales and losses:
  [log Cscale,log escale,log kappascale,log eta_m,log eta_d,xi_beta].
One additional signed ratio beta links ALL three imaginary piezo coefficients
to their real values, rather than identifying three independent intrinsic losses.

With full real TI C SPD and positive kappa, current uniform C/dielectric loss,
  chi_z=e_z^T C_normal^-1 e_z,
  beta_max=sqrt(eta_m*eta_d*min(kappaZ/chi_z,C44*kappaR/e15^2)),
  beta=.98*tanh(xi_beta)*beta_max.
This keeps the complete normal/shear imaginary loss strictly inside its Schur
limits.  All formulas use current physical coefficients, including derivatives
of beta_max.  No physical coefficient is clamped, clipped, inverted as a matrix
or projected after evaluation.  The finite-domain family explicitly requires
nonzero nominal e15 and nonzero normal piezo coupling; no fake inverse handles
degenerate cases.  Grid-specific real direct-star admissibility remains the
physical graph validator's responsibility.

Output imag10 order is C5im,e15im,e31im,e33im,kappaRim,kappaZim.
The inherited base port's evaluate must use len(parameters.names), not a literal
five, and its problem must accept imag10.  This adapter does NOT change that
base implementation or silently route an old imag7 solver to new physics.
Importing this file performs no model instantiation/solve/fit/training.
"""

import torch

from .piezo_axisym_parameter_graph import (
    AxisymmetricMaterialParameters,
    AxisymmetricPortParameterGraph,
)

REAL_NAMES = ("c11", "c12", "c13", "c33", "c44", "e15", "e31", "e33", "kappa11", "kappa33")
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


class AxisymmetricComplexMaterialParameters(AxisymmetricMaterialParameters):
    """Six shared/material-batch coordinates; no independent tensor recovery.

    ``material(theta[...,6]) -> real[...,10], imag[...,10], eta_m[...]``
    preserves the parent's three-result convention.  ``decode`` is a JSON-ready
    report for one shared theta[6].  No numerical field solve is invoked.
    """

    names = (
        "log_C_scale",
        "log_e_scale",
        "log_kappa_scale",
        "log_eta_mechanical",
        "log_eta_dielectric",
        "xi_beta_shared_piezo_imag_ratio",
    )
    passive_safety_multiplier = 0.98

    def __init__(self, nominal_real):
        if not isinstance(nominal_real, torch.Tensor):
            raise TypeError("Nominal SI real material must be a float64 tensor[10]")
        super().__init__(nominal_real)
        if not bool(torch.isfinite(self.nominal).all().item()):
            raise ValueError("Nominal material must be finite")
        a, b, c, d, s = self.nominal[:5].unbind()
        normal_determinant = (a + b) * d - 2 * c.square()
        if not bool(
            (
                (a - b > 0)
                & (a + b > 0)
                & (d > 0)
                & (s > 0)
                & (normal_determinant > 0)
                & (self.nominal[8] > 0)
                & (self.nominal[9] > 0)
            ).item()
        ):
            raise ValueError(
                "This family requires full real TI SPD and positive clamped permittivities"
            )
        if not bool((self.nominal[5] != 0).item()):
            raise ValueError("This bounded family explicitly requires nonzero nominal e15")
        if not bool(((self.nominal[6] != 0) | (self.nominal[7] != 0)).item()):
            raise ValueError("This bounded family requires nonzero normal piezo coupling")

    def _mapping(self, theta):
        if (
            not isinstance(theta, torch.Tensor)
            or theta.ndim < 1
            or theta.shape[-1] != 6
            or theta.dtype != torch.float64
            or theta.device != self.nominal.device
        ):
            raise TypeError("Need real float64 theta[...,6] on the nominal material device")
        if not bool(torch.isfinite(theta).all().item()):
            raise ValueError("Material coordinates must be finite")
        # Reuse the first-five family verbatim; xi_beta is NOT exponentiated.
        real, legacy_imag, eta_m = super().material(theta[..., :5])
        eta_d = theta[..., 4].exp()
        if (
            not bool(torch.isfinite(real).all().item())
            or not bool(torch.isfinite(legacy_imag).all().item())
            or not bool(torch.all((eta_m > 0) & (eta_d > 0)).item())
        ):
            raise ValueError("Log scales produced nonfinite/underflowed physical coefficients")
        c11, c12, c13, c33, c44 = real[..., :5].unbind(dim=-1)
        e15, e31, e33 = real[..., 5:8].unbind(dim=-1)
        kr, kz = real[..., 8], real[..., 9]
        aa = c11 + c12
        determinant = aa * c33 - 2 * c13.square()
        if not bool(
            torch.all(
                torch.isfinite(aa)
                & torch.isfinite(determinant)
                & (aa > 0)
                & (determinant > 0)
                & (e15 != 0)
            ).item()
        ):
            raise ValueError(
                "Current finite-precision physical coefficients left the strict family domain"
            )
        # Same normal susceptibility as the expanded Schur numerator, written
        # as a sum of nonnegative squares to avoid numerator cancellation.
        chi_z = 2 * e31.square() / aa + aa * (e33 - 2 * c13 * e31 / aa).square() / determinant
        normal_ratio = kz / chi_z
        shear_ratio = c44 * kr / e15.square()
        limiting_ratio = torch.minimum(normal_ratio, shear_ratio)
        bound_squared = eta_m * eta_d * limiting_ratio
        if not bool(
            torch.all(
                torch.isfinite(chi_z)
                & (chi_z > 0)
                & torch.isfinite(normal_ratio)
                & (normal_ratio > 0)
                & torch.isfinite(shear_ratio)
                & (shear_ratio > 0)
                & torch.isfinite(bound_squared)
                & (bound_squared > 0)
            ).item()
        ):
            raise ValueError(
                "Passive beta bound is not finite/strictly positive; no clipping or fake inverse is applied"
            )
        beta_max = torch.sqrt(bound_squared)
        beta = self.passive_safety_multiplier * torch.tanh(theta[..., 5]) * beta_max
        piezo_imag = beta[..., None] * real[..., 5:8]
        imag = torch.cat((legacy_imag[..., :5], piezo_imag, legacy_imag[..., 5:]), dim=-1)
        if not bool(torch.isfinite(imag).all().item()):
            raise ValueError("Nonfinite complex-loss material")
        details = dict(
            beta=beta,
            beta_max=beta_max,
            eta_m=eta_m,
            eta_d=eta_d,
            chi_z=chi_z,
            normal_ratio=normal_ratio,
            shear_ratio=shear_ratio,
            limiting_ratio=limiting_ratio,
        )
        return real, imag, eta_m, details

    def material(self, theta):
        real, imag, eta, _ = self._mapping(theta)
        return real, imag, eta

    def decode(self, theta):
        if not isinstance(theta, torch.Tensor) or theta.shape != (6,):
            raise ValueError("decode requires one shared theta[6], not a material batch")
        real, imag, eta, details = self._mapping(theta)
        number = lambda value: float(value.detach().cpu().item())
        rz, rr = number(details["normal_ratio"]), number(details["shear_ratio"])
        sector = "normal_z" if rz < rr else ("shear_r" if rr < rz else "equal_normal_and_shear")
        real_values = real.detach().cpu().tolist()
        imag_values = imag.detach().cpu().tolist()
        return dict(
            C_scale=number(theta[0].exp()),
            e_scale=number(theta[1].exp()),
            kappa_scale=number(theta[2].exp()),
            eta_mechanical=number(eta),
            eta_dielectric=number(details["eta_d"]),
            xi_beta=number(theta[5]),
            beta=number(details["beta"]),
            beta_max=number(details["beta_max"]),
            beta_fraction_of_passive_bound=number(details["beta"] / details["beta_max"]),
            passive_safety_multiplier=self.passive_safety_multiplier,
            normal_coupling_susceptibility_F_per_m=number(details["chi_z"]),
            normal_beta_bound_ratio=rz,
            shear_beta_bound_ratio=rr,
            limiting_passivity_sector=sector,
            real_SI=real_values,
            imag_SI=imag_values,
            real_SI_named=dict(zip(REAL_NAMES, real_values)),
            imag_SI_named=dict(zip(IMAG_NAMES, imag_values)),
            real_parameter_order=list(REAL_NAMES),
            imaginary_parameter_order=list(IMAG_NAMES),
            material_coordinate_names=list(self.names),
            material_coordinate_dimension=6,
            piezo_imaginary_family="one shared signed beta times current real (e15,e31,e33)",
            nonaffine_mechanical_loss_equals_eta_mechanical=True,
            strict_passive_domain="positive uniform eta_m/eta_d and nominal real TI SPD; .98*tanh interior",
            grid_specific_real_direct_star_domain_still_requires_graph_validation=True,
            conditional_scope=[
                "Fixed nominal TI elastic/piezo/dielectric tensor shapes; only shared scales.",
                "Known fixed geometry, density, poling and linear small-signal convention.",
                "One common complex-piezo ratio; not three independent loss mechanisms.",
                "Uniform mechanical/dielectric loss families; not a broadband causal material law.",
                "Fixture, electrodes and circuit assumptions remain those of the chosen forward model.",
                "Response fit and passive mapping do not establish intrinsic truth or identifiability.",
            ],
            all_intrinsic_constants_independently_identified=False,
            three_imaginary_piezo_coefficients_independently_identified=False,
            trained_network=False,
        )


class AxisymmetricComplexPortParameterGraph(AxisymmetricPortParameterGraph):
    """Inherited reciprocal port with the six-coordinate material object.

    Requires the separately updated base evaluate using len(parameters.names)
    and a separately verified problem accepting imag10.  No inherited method
    is duplicated or patched here; the old five-family mapper remains intact.
    """

    names = AxisymmetricComplexMaterialParameters.names

    def __init__(self, problem, nominal_real):
        super().__init__(problem, nominal_real)
        self.parameters = AxisymmetricComplexMaterialParameters(nominal_real)
