"""Indexed, batched pure-message harmonic wrapper for an axisymmetric graph.

No material, mass, T, P or transfer matrix is constructed.  The graph supplies
closed-form local actions in physical SI units, including mechanical inertia.
This wrapper imposes ur(axis)=0, phi(bottom)=0, phi(top)=1 V.  ALL uz remain
free at positive frequency; no axial pin or artificial damping is introduced.

The static diagonal is extracted by 27 simultaneous geometry-color message
inputs (3x3 node colors times 3 channels).  This is exact only for the declared
structured graph's action half-width <=2 grid intervals.  These are physical
batch states, not columns of a stored global operator.  Positive elementwise
congruence scaling uses |static diagonal|+omega^2*mechanical nodal mass.

Each frequency is an independent complex-symmetric system.  The batched scalar-
Givens FGMRES utility owns row masks and true-residual stopping.  This module
provides forward fields/diagnostics only: no parameter backward/feature layer,
matrix oracle, old FEM import or full-field autograd action is used.
"""

import math
import time

import torch

from .piezo_graph_batch_iterations import solve as batch_fgmres


def _norm_rows(value):
    if value.shape[1] == 0:
        return value.real.new_zeros(value.shape[0])
    scale = value.abs().amax(dim=1)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale * torch.sqrt((value / safe[:, None]).abs().square().sum(dim=1))


class AxisymmetricMessageProblem(torch.nn.Module):
    """Exact indexed free space and an optional right preconditioner factory.

    The graph API is action(state[B,N,3],real[10],imag[7],frequencies[B],
    eta_nonaffine=0), coordinates[N,2], node_mass_kg[N], nr/nz and boundary IDs.
    Graph actions ALREADY subtract omega^2*m*u; do not add inertia twice.

    ``preconditioner_factory(self,real,imag,frequencies,scale,scaled_action)``
    returns an optional batched vector map, never a matrix.  Factory operators
    can use ``_state_action`` and ``reduced_action``; current_eta_nonaffine is
    available when their optional eta_nonaffine argument is omitted.
    """

    imaginary_parameter_sizes = (7,)

    def __init__(self, graph, rtol=1e-9, restart=80, maxiter=1600, preconditioner_factory=None):
        super().__init__()
        for name in (
            "coordinates",
            "node_mass_kg",
            "nr",
            "nz",
            "n",
            "action",
            "validate_parameters",
            "axis_nodes",
            "bottom_nodes",
            "top_nodes",
        ):
            if not hasattr(graph, name):
                raise TypeError("Graph is missing " + name)
        if (
            graph.coordinates.shape != (graph.n, 2)
            or graph.coordinates.dtype != torch.float64
            or graph.node_mass_kg.shape != (graph.n,)
            or graph.node_mass_kg.dtype != torch.float64
            or graph.n != (graph.nr + 1) * (graph.nz + 1)
        ):
            raise ValueError("Expected the declared float64 rectangular node-major graph")
        if not math.isfinite(float(rtol)) or rtol < 0:
            raise ValueError("rtol must be finite and nonnegative")
        if isinstance(restart, bool) or not isinstance(restart, int) or restart < 1:
            raise ValueError("restart must be a positive integer")
        if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 0:
            raise ValueError("maxiter must be a nonnegative integer")
        if preconditioner_factory is not None and not callable(preconditioner_factory):
            raise TypeError("preconditioner_factory must be callable or None")
        self.graph, self.n, self.ndof = graph, int(graph.n), 3 * int(graph.n)
        self.rtol, self.restart, self.maxiter = float(rtol), restart, maxiter
        self.preconditioner_factory = preconditioner_factory
        self.current_eta_nonaffine = 0.0
        self._action_call_count = self._action_row_count = 0
        device = graph.coordinates.device
        fixed = torch.zeros(self.ndof, dtype=torch.bool, device=device)
        fixed[3 * graph.axis_nodes] = True
        fixed[3 * graph.bottom_nodes + 2] = True
        fixed[3 * graph.top_nodes + 2] = True
        free = (~fixed).nonzero(as_tuple=False).flatten()
        self.nfree = int(free.numel())
        self.register_buffer("free_indices", free)
        self.register_buffer("free_mechanical_mask", free.remainder(3) != 2)
        self.register_buffer("free_electrical_mask", free.remainder(3) == 2)
        self.register_buffer("axis", graph.axis_nodes.detach().clone())
        self.register_buffer("bottom", graph.bottom_nodes.detach().clone())
        self.register_buffer("top", graph.top_nodes.detach().clone())
        node = torch.arange(self.n, dtype=torch.long, device=device)
        ir, iz = torch.div(node, graph.nz + 1, rounding_mode="floor"), node.remainder(graph.nz + 1)
        colors = (
            3 * (3 * ir.remainder(3) + iz.remainder(3))[:, None]
            + torch.arange(3, device=device)[None, :]
        )
        self.register_buffer("_diagonal_colors", colors.reshape(-1))
        self.register_buffer(
            "_full_dof_ids", torch.arange(self.ndof, dtype=torch.long, device=device)
        )
        self.metadata = dict(
            constraints="axis ur=0; full bottom/top phi=0/1 V peak",
            all_axial_displacements_free=True,
            nonzero_frequency_only=True,
            constraint_representation="pure indexed lift/prolong/transpose restriction",
            global_constraint_matrix_constructed=False,
            global_material_matrix_constructed=False,
            static_diagonal="27 geometry-color batch messages; signed unconstrained diagonal",
            required_action_halfwidth_grid_intervals=2,
            parameter_backward_implemented=False,
            no_old_FEM_import=True,
            no_fullfield_autograd_action=True,
        )

    def _check_reduced(self, z):
        if (
            not isinstance(z, torch.Tensor)
            or z.ndim != 2
            or z.shape[1] != self.nfree
            or z.shape[0] < 1
            or z.dtype not in (torch.float64, torch.complex128)
            or z.device != self.free_indices.device
        ):
            raise ValueError("Expected float64/complex128 reduced batch [B,nfree] on graph device")

    def prolong(self, z):
        """Homogeneous indexed Tz; no lift is included."""
        self._check_reduced(z)
        return (
            z.new_zeros((z.shape[0], self.ndof))
            .index_add(1, self.free_indices, z)
            .reshape(z.shape[0], self.n, 3)
        )

    def contract(self, full):
        """T-transpose restriction; it is not a mean or a potential gauge fix."""
        if (
            not isinstance(full, torch.Tensor)
            or full.ndim != 3
            or full.shape[1:] != (self.n, 3)
            or full.dtype not in (torch.float64, torch.complex128)
            or full.device != self.free_indices.device
        ):
            raise ValueError("Expected full batch [B,N,3] on graph device")
        return full.reshape(full.shape[0], self.ndof)[:, self.free_indices]

    def lift(self):
        """Physical complex128 [N,3] unit-voltage lift; no mechanical clamp."""
        result = self.graph.coordinates.new_zeros((self.n, 3), dtype=torch.complex128)
        result[self.top, 2] = 1.0
        return result

    def _frequency_vector(self, frequencies, *, positive):
        if isinstance(frequencies, torch.Tensor) and frequencies.requires_grad:
            raise ValueError("Frequencies are fixed inputs")
        raw = torch.as_tensor(frequencies, device=self.free_indices.device)
        if raw.is_complex() or raw.ndim > 1:
            raise ValueError("Frequencies must be a real scalar or vector of Hz")
        value = torch.as_tensor(
            frequencies, dtype=torch.float64, device=self.free_indices.device
        ).reshape(-1)
        if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
            raise ValueError("A nonempty finite frequency batch is required")
        if not bool(((value > 0) if positive else (value >= 0)).all().item()):
            raise ValueError("Positive frequencies are required for the unpinned harmonic solve")
        return value

    def _state_action(self, state, real, imag, frequencies, eta_nonaffine=None):
        """One closed-form graph action; supplied graph already includes inertia."""
        if state.dtype == torch.float64:
            state = state.to(dtype=torch.complex128)
        frequency = torch.as_tensor(frequencies, dtype=torch.float64, device=state.device)
        if frequency.ndim == 0:
            frequency = frequency.expand(state.shape[0])
        eta = self.current_eta_nonaffine if eta_nonaffine is None else eta_nonaffine
        self._action_call_count += 1
        self._action_row_count += int(state.shape[0])
        return self.graph.action(state, real, imag, frequency, eta_nonaffine=eta)

    def reduced_action(self, z, real, imag, frequencies, eta_nonaffine=None):
        return self.contract(
            self._state_action(self.prolong(z), real, imag, frequencies, eta_nonaffine)
        )

    def scaled_action(self, y, real, imag, frequencies, scale, eta_nonaffine=None):
        return scale * self.reduced_action(scale * y, real, imag, frequencies, eta_nonaffine)

    @torch.no_grad()
    def static_diagonal(self, real, eta_nonaffine=0.0):
        """Exact SIGNED [N,3] static diagonal by a single 27-row action batch.

        Requires the graph's declared half-width <=2 in both grid coordinates.
        Standalone callers must validate the material domain explicitly.  This
        method does not call validate_parameters again within solve_batch.
        """
        probe = self.graph.coordinates.new_zeros((27, self.ndof), dtype=torch.complex128)
        probe[self._diagonal_colors, self._full_dof_ids] = 1.0
        imaginary = real.new_zeros(7)
        frequency = real.new_zeros(27)
        acted = self._state_action(
            probe.reshape(27, self.n, 3), real, imaginary, frequency, eta_nonaffine
        )
        picked = (
            acted.reshape(27, self.ndof)
            .gather(0, self._diagonal_colors[None, :])
            .reshape(self.n, 3)
        )
        # With real material, zero imaginary input and eta=0 this is real.  The
        # explicitly requested signed real diagonal is still returned for eta>0;
        # that control adds only imaginary non-affine stiffness.
        return picked.real

    def rhs_batch(self, real, imag, frequencies, eta_nonaffine=None):
        lift = self.lift()[None, :, :].expand(frequencies.numel(), self.n, 3)
        return -self.contract(self._state_action(lift, real, imag, frequencies, eta_nonaffine))

    @torch.no_grad()
    def solve_batch(self, real, imag, frequencies, eta_nonaffine=0.0, initial_state=None):
        """Return physical batch fields, electrode ports, and per-row records.

        This is forward-only.  Nonconverged rows and their true-residual traces
        are returned, not relabelled success or silently pinned/regularized.
        Diagnostics report absolute SI residuals and dimensionless S-scaled
        residuals; neither is advertised as a physical field-error bound.
        """
        tick = time.perf_counter()
        self._action_call_count = self._action_row_count = 0
        if (
            not isinstance(real, torch.Tensor)
            or real.shape != (10,)
            or real.dtype != torch.float64
            or real.device != self.free_indices.device
        ):
            raise ValueError("Shared real material must be float64 [10] on graph device")
        if imag is None:
            imag = real.new_zeros(7)
        if (
            not isinstance(imag, torch.Tensor)
            or imag.ndim != 1
            or imag.numel() not in self.imaginary_parameter_sizes
            or imag.dtype != torch.float64
            or imag.device != real.device
        ):
            raise ValueError(
                "Shared imaginary material must be float64 with an explicitly supported parameter count on graph device"
            )
        frequency = self._frequency_vector(frequencies, positive=True)
        # Exactly one explicit parameter-domain validation per call/material.
        validation = self.graph.validate_parameters(real, imag, eta_nonaffine=eta_nonaffine)
        self.current_eta_nonaffine = (
            float(eta_nonaffine.item())
            if isinstance(eta_nonaffine, torch.Tensor)
            else float(eta_nonaffine)
        )
        batch = int(frequency.numel())
        diagonal = self.static_diagonal(real, eta_nonaffine=self.current_eta_nonaffine).reshape(-1)
        mass = self.graph.node_mass_kg
        mechanical_mass = torch.stack((mass, mass, torch.zeros_like(mass)), dim=1).reshape(-1)[
            self.free_indices
        ]
        omega_squared = (2 * math.pi * frequency).square()
        magnitude = (
            diagonal[self.free_indices].abs()[None, :]
            + omega_squared[:, None] * mechanical_mass[None, :]
        )
        if not bool(torch.isfinite(magnitude).all().item()) or not bool(
            (magnitude > 0).all().item()
        ):
            raise ValueError("Zero/nonfinite positive scaling magnitude; no diagonal shift applied")
        scale = torch.rsqrt(magnitude)
        rhs = self.rhs_batch(real, imag, frequency)

        def action(y):
            return self.scaled_action(y, real, imag, frequency, scale)

        preconditioner = (
            None
            if self.preconditioner_factory is None
            else self.preconditioner_factory(self, real, imag, frequency, scale, action)
        )
        initial_y = None
        if initial_state is not None:
            if (
                initial_state.shape != (batch, self.graph.n, 3)
                or initial_state.dtype != torch.complex128
                or initial_state.device != real.device
                or not bool(torch.isfinite(initial_state).all())
            ):
                raise ValueError("Warm initial field must be finite complex128 [B,nnode,3]")
            # Only an initial guess. Essential constraints are reconstructed
            # from the new lift, and the full fresh residual still decides.
            initial_y = self.contract(initial_state.detach() - self.lift()[None, :, :]) / scale
        y, batch_info = batch_fgmres(
            action,
            scale * rhs,
            x0=initial_y,
            preconditioner=preconditioner,
            rtol=self.rtol,
            atol=0.0,
            restart=self.restart,
            maxiter=self.maxiter,
        )
        z = scale * y
        state = self.lift()[None, :, :] + self.prolong(z)
        internal = self._state_action(state, real, imag, frequency)
        free_residual = self.contract(internal)
        # This fresh S-residual includes the inhomogeneous lift through the
        # full state.  It is independent of the Krylov recurrence estimate.
        scaled_residual = scale * free_residual
        scaled_rhs = scale * rhs
        rhs_norm = _norm_rows(scaled_rhs)
        residual_norm = _norm_rows(scaled_residual)
        reference = torch.where(rhs_norm > 0, rhs_norm, torch.ones_like(rhs_norm))
        charge = -internal[:, self.top, 2].sum(dim=1)
        ground = -internal[:, self.bottom, 2].sum(dim=1)
        omega = 2 * math.pi * frequency
        admittance = 1j * omega * charge
        impedance = torch.reciprocal(admittance)
        static = self._state_action(state, real, imag, frequency.new_zeros(batch))
        source_power = 0.5 * admittance.real
        conjugated_static = (state.conj() * static).sum(dim=(1, 2))
        dissipated = 0.5 * omega * conjugated_static.imag
        bilinear_charge = -(state * internal).sum(dim=(1, 2))
        centroid = (mass[None, :] * state[:, :, 1]).sum(dim=1)
        total_mass = float(mass.sum().item())
        tiny = torch.finfo(torch.float64).tiny
        records = []
        for row in range(batch):
            qscale = max(float(charge[row].abs().item()), float(ground[row].abs().item()), tiny)
            pscale = max(
                abs(float(source_power[row].item())), abs(float(dissipated[row].item())), 1e-30
            )
            d = dict(
                frequency_Hz=float(frequency[row].item()),
                unit_voltage_V_peak=1.0,
                fresh_scaled_true_residual_norm=float(residual_norm[row].item()),
                fresh_scaled_relative_true_residual=float(
                    (residual_norm[row] / reference[row]).item()
                ),
                scaled_rhs_norm=float(rhs_norm[row].item()),
                relative_reference="norm(S*rhs) if positive, otherwise 1",
                residuals_are_not_physical_field_error_bounds=True,
                charge_balance_relative_error=float((charge[row] + ground[row]).abs().item())
                / qscale,
                reciprocal_charge_identity_relative_error=float(
                    (charge[row] - bilinear_charge[row]).abs().item()
                )
                / qscale,
                source_power_W=float(source_power[row].item()),
                dissipated_power_W=float(dissipated[row].item()),
                power_absolute_difference_W=float(
                    (source_power[row] - dissipated[row]).abs().item()
                ),
                power_relative_error=float((source_power[row] - dissipated[row]).abs().item())
                / pscale,
                power_reference_is_tiny=(pscale < 1e-20),
                mass_weighted_axial_displacement_abs_kg_m=float(centroid[row].abs().item()),
                mass_weighted_mean_axial_displacement_abs_m=float(centroid[row].abs().item())
                / total_mass,
                mass_weighted_axial_displacement_real_kg_m=float(centroid[row].real.item()),
                mass_weighted_axial_displacement_imag_kg_m=float(centroid[row].imag.item()),
                total_mass_kg=total_mass,
                axis_radial_displacement_max_m=float(state[row, self.axis, 0].abs().max().item()),
                bottom_potential_max_V=float(state[row, self.bottom, 2].abs().max().item()),
                top_potential_error_max_V=float((state[row, self.top, 2] - 1).abs().max().item()),
                finite_state=bool(torch.isfinite(state[row]).all().item()),
                finite_impedance=bool(torch.isfinite(impedance[row]).item()),
            )
            for name, mask, unit in (
                ("mechanical", self.free_mechanical_mask, "N"),
                ("electrical", self.free_electrical_mask, "C"),
            ):
                raw = free_residual[row : row + 1, mask]
                sr = scaled_residual[row : row + 1, mask]
                sb = scaled_rhs[row : row + 1, mask]
                sbnorm = float(_norm_rows(sb)[0].item())
                srnorm = float(_norm_rows(sr)[0].item())
                d[name + "_free_residual_l2_" + unit] = float(_norm_rows(raw)[0].item())
                d[name + "_scaled_residual_l2"] = srnorm
                d[name + "_scaled_rhs_l2"] = sbnorm
                d[name + "_scaled_relative_residual"] = srnorm / (sbnorm if sbnorm > 0 else 1.0)
            info = dict(batch_info["row_infos"][row])
            info["fresh_full_state_scaled_relative_residual"] = d[
                "fresh_scaled_relative_true_residual"
            ]
            records.append(dict(info=info, diagnostics=d))
        batch_info = dict(batch_info)
        batch_info.update(
            wrapper_elapsed_seconds=time.perf_counter() - tick,
            wrapper_graph_action_calls=self._action_call_count,
            wrapper_graph_rows_evaluated=self._action_row_count,
            action_count_scope="only calls routed through wrapper; includes 27 diagonal probe rows",
            parameter_domain_validation=validation,
            positive_scaling="abs(exact signed real static diagonal)+omega^2 mechanical nodal mass",
            all_uz_free=True,
            mechanical_pin_applied=False,
            global_operator_constructed=False,
            warm_initial_field_supplied=initial_state is not None,
            fullfield_autograd_action_used=False,
            preconditioner_supplied=preconditioner is not None,
        )
        return dict(
            state=state,
            charge=charge,
            ground_charge=ground,
            impedance=impedance,
            admittance=admittance,
            records=records,
            batch_info=batch_info,
        )
