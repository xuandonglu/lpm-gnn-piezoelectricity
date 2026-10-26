"""In-plane material-aligned lattice rotation.

The unchanged direct-star coefficients are specified in material axes. Particle
positions, bond directions, collective displacement coefficients, and the polar
axis rotate together. Global displacements and forces remain global vectors;
potential and electrical residuals remain scalars. No constitutive tensor is
rotated inside this graph. The existing boundary-completion construction is
performed on the material-aligned square lattice before rotation.

This class represents one homogeneous material orientation.
"""

import numpy as np
import torch

from lpm_gnn.physics.piezo_message_geometry import MessageGeometry
from lpm_gnn.physics.piezo_message_graph import PiezoMessageGraph


class RotatedLatticeGraph(PiezoMessageGraph):
    def __init__(self, material_mesh, angle, *, origin=(0.0, 0.0), **kwargs):
        angle = float(angle)
        origin = np.asarray(origin, float)
        if not np.isfinite(angle) or origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("Finite angle in radians and planar origin required")
        super().__init__(material_mesh, **kwargs)
        q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotation = torch.as_tensor(q, dtype=self.coordinates.dtype, device=self.coordinates.device)
        self.register_buffer("material_coordinates", self.coordinates.clone())
        self.register_buffer("rotation", rotation)
        self.register_buffer("poling_direction", rotation[:, 1].clone())
        shift = torch.as_tensor(origin, dtype=rotation.dtype, device=rotation.device)
        self.coordinates = self.material_coordinates @ rotation.T + shift
        self.bond_direction = self.bond_direction @ rotation.T
        self.axial_node_coeff = self.axial_node_coeff @ rotation.T
        self.diagonal_node_coeff = self.diagonal_node_coeff @ rotation.T
        self.global_mesh = MessageGeometry(
            self.coordinates.detach().cpu().numpy(),
            material_mesh.cells.copy(),
            material_mesh.spacing,
            material_mesh.face_pairs.copy(),
        )
        self.metadata.update(
            material_orientation_radians=angle,
            orientation_representation="rotated particle lattice and polar direction",
            constitutive_coefficients_frame="unchanged material axes",
            global_constitutive_tensor_rotation_in_graph=False,
            homogeneous_material_orientation=True,
        )

    def energy(self, state, theta):
        spring, tv, eta, alpha, beta = self._material_coefficients(theta)
        ell, delta = self._differences(state)
        directional_polarity = self.bond_direction @ self.poling_direction
        sa = self._sum_star(self.axial * ell)
        sd = self._sum_star(self.diagonal * ell)
        p = self._sum_star(self.axial * directional_polarity * delta)
        local = self._sum_star(
            0.5 * spring * ell.square()
            + eta * directional_polarity * ell * delta
            - 0.5 * beta * delta.square()
        )
        return (
            self.star_scale * (local + 0.5 * tv * (sa.square() + sd.square()) + alpha * sa * p)
        ).sum()

    def action(self, state, theta):
        spring, tv, eta, alpha, beta = self._material_coefficients(theta)
        ell, delta = self._differences(state)
        directional_polarity = self.bond_direction @ self.poling_direction
        si = self.bond_star
        sa = self._sum_star(self.axial * ell)
        sd = self._sum_star(self.diagonal * ell)
        p = self._sum_star(self.axial * directional_polarity * delta)
        fl = (
            spring * ell
            + tv * (self.axial * sa[si] + self.diagonal * sd[si])
            + eta * directional_polarity * delta
            + alpha * self.axial * p[si]
        )
        fp = (
            -beta * delta
            + eta * directional_polarity * ell
            + alpha * self.axial * directional_polarity * sa[si]
        )
        response = torch.cat((fl[:, None] * self.bond_direction, fp[:, None]), 1)
        response = response * self.star_scale[si, None]
        return state.new_zeros((self.n, 3)).index_add(
            0,
            self.coefficient_node,
            self.coefficient_value[:, None] * response[self.coefficient_bond],
        )
