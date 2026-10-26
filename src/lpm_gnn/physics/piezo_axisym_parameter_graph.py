"""Material-parameter autograd bridge for the direct particle-message graph.

Five shared parameters: logarithmic C/e/kappa scales, log mechanical loss,
log dielectric loss.  This is a deliberately restricted material family,
not ten individually identified intrinsic constants. All forward fields come
from message iterations. No iteration tape, stiffness matrix, or adjoint
matrix solve is used. For this SINGLE fixed-voltage reciprocal port only,
dQ/dtheta = -u.T (dA/dtheta) u. Different observations require their own
adjoint and cannot silently reuse this identity.

The parameter contraction uses the nonconjugate physical energy, with the
full-field solution DETACHED. Autograd differentiates local scalar material
and star formulas, never an exact analytic continuum response.
"""

import math

import torch


class AxisymmetricMaterialParameters:
    names = (
        "log_C_scale",
        "log_e_scale",
        "log_kappa_scale",
        "log_eta_mechanical",
        "log_eta_dielectric",
    )

    def __init__(self, nominal_real):
        if nominal_real.shape != (10,) or nominal_real.dtype != torch.float64:
            raise TypeError("Nominal SI real material must be float64 [10]")
        self.nominal = nominal_real.detach().clone()

    def material(self, theta):
        if theta.shape[-1:] != (5,) or theta.dtype != torch.float64:
            raise TypeError("Need real float64 theta[...,5]")
        factors = theta.exp()
        real = torch.cat(
            (
                self.nominal[:5] * factors[..., 0, None],
                self.nominal[5:8] * factors[..., 1, None],
                self.nominal[8:] * factors[..., 2, None],
            ),
            dim=-1,
        )
        imag = torch.cat(
            (real[..., :5] * factors[..., 3, None], -real[..., 8:] * factors[..., 4, None]), dim=-1
        )
        return real, imag, factors[..., 3]

    def decode(self, theta):
        real, imag, eta = self.material(theta)
        return dict(
            C_scale=float(theta[0].exp()),
            e_scale=float(theta[1].exp()),
            kappa_scale=float(theta[2].exp()),
            eta_mechanical=float(eta),
            eta_dielectric=float(theta[4].exp()),
            real_SI=real.detach().cpu().tolist(),
            imag_SI=imag.detach().cpu().tolist(),
            all_intrinsic_constants_independently_identified=False,
        )


class AxisymmetricPortParameterGraph:
    def __init__(self, problem, nominal_real):
        self.problem = problem
        self.graph = problem.graph
        self.parameters = AxisymmetricMaterialParameters(nominal_real)
        self.last_records = None
        self.previous_state = None
        self.previous_frequencies = None

    def evaluate(self, theta, frequencies, with_derivatives=True):
        """Return log|Y|, arg(Y), and exact fixed-topology port derivatives.

        Frequency rows are independent; material PARTIALS use independent
        copies of theta so a single scalar contraction per real/imag channel
        produces one derivative row per frequency. This does not claim new
        independent measurements. The actual forward solve uses shared real
        material and evaluates the original prescribed voltage boundary.
        """
        parameter_count = len(self.parameters.names)
        if theta.shape != (parameter_count,):
            raise ValueError("Shared material parameter count does not match the explicit family")
        with torch.no_grad():
            real, imag, eta = self.parameters.material(theta.detach())
            warm = (
                self.previous_state
                if self.previous_frequencies is not None
                and torch.equal(frequencies, self.previous_frequencies)
                else None
            )
            forward = self.problem.solve_batch(
                real, imag, frequencies, eta_nonaffine=float(eta), initial_state=warm
            )
        self.last_records = forward["records"]
        for record in self.last_records:
            if (
                not record["info"]["converged"]
                or record["diagnostics"]["fresh_scaled_relative_true_residual"]
                > 2 * self.problem.rtol
            ):
                raise RuntimeError("Unconverged message field is not accepted by inverse")
            d = record["diagnostics"]
            if d["power_relative_error"] > 2e-5 or d["charge_balance_relative_error"] > 2e-5:
                raise RuntimeError(
                    "Port conservation gate failed; response not accepted by inverse"
                )
        q = forward["charge"].detach()
        if not bool(torch.isfinite(q).all()) or not bool((q.abs() > 0).all()):
            raise FloatingPointError("A finite nonzero port charge is required")
        self.previous_state = forward["state"].detach()
        self.previous_frequencies = frequencies.detach().clone()
        omega = 2 * math.pi * frequencies
        y = 1j * omega * q
        prediction = torch.stack((torch.log(y.abs()), torch.angle(y)), dim=1)
        if not with_derivatives:
            return prediction, None, forward
        with torch.enable_grad():
            repeated = (
                theta.detach()[None, :]
                .expand(len(frequencies), parameter_count)
                .clone()
                .requires_grad_(True)
            )
            reals, imags, etas = self.parameters.material(repeated)
            state = forward["state"].detach()
            # Entire uniform mechanical loss includes the direct-star
            # non-affine remainder. eta here is MATERIAL loss, not a fitted
            # stabilization. Derivative of eta*R is explicit and exact.
            energy = self.graph.energy(state, reals, imags, frequencies, eta_nonaffine=0.0)
            fields = self.graph.constitutive_fields(state, reals)
            remainder = (self.graph.volumes[None, :] * fields["nonaffine_density"]).sum(dim=1)
            reciprocal_q = -2 * (energy + 1j * etas * remainder)
            dqr = torch.autograd.grad(reciprocal_q.real.sum(), repeated, retain_graph=True)[0]
            dqi = torch.autograd.grad(reciprocal_q.imag.sum(), repeated)[0]
            dq = torch.complex(dqr, dqi)
            relative = dq / q[:, None]
            jacobian = torch.stack((relative.real, relative.imag), dim=1).detach()
        identity_error = (reciprocal_q.detach() - q).abs() / q.abs().clamp_min(1e-30)
        forward["parameter_contraction_charge_identity_relative_error"] = identity_error
        forward["parameter_derivative_method"] = (
            "single voltage port reciprocity; local physical energy partials; no solve tape"
        )
        return prediction.detach(), jacobian, forward

    def predict(self, theta, frequencies):
        return _PortPrediction.apply(theta, self, frequencies)


class _PortPrediction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, model, frequencies):
        prediction, jacobian, _ = model.evaluate(theta, frequencies, True)
        ctx.save_for_backward(jacobian)
        return prediction

    @staticmethod
    def backward(ctx, gradient):
        (jacobian,) = ctx.saved_tensors
        return (gradient[:, :, None] * jacobian).sum(dim=(0, 1)), None, None
