"""Small physical checks requiring no downloaded data or saved checkpoints."""

import unittest

import numpy as np
import torch

from lpm_gnn.examples.disk import NOMINAL
from lpm_gnn.nonlinear.data import normalize_parameters
from lpm_gnn.nonlinear.physics import SparsePhysics, load_case, sample_cases
from lpm_gnn.nonlinear.training import predictor
from lpm_gnn.physics.piezo_axisym_complex_parameter_graph import (
    AxisymmetricComplexMaterialParameters,
)
from lpm_gnn.physics.piezo_message_geometry import MessageGeometry
from lpm_gnn.physics.piezo_message_graph import PiezoMessageGraph
from lpm_gnn.physics.rotated_lattice import RotatedLatticeGraph
from lpm_gnn.reference.planar_fem import pzt4, q4_reference


class PhysicalConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.mesh = MessageGeometry.rectangle(3)
        cls.graph = PiezoMessageGraph(cls.mesh)

    def test_enthalpy_gradient_matches_message_action(self):
        rng = np.random.default_rng(20261025)
        state = torch.tensor(
            rng.normal(size=(self.mesh.n, 3)) * [1e-8, 1e-8, 1.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        (gradient,) = torch.autograd.grad(self.graph.energy(state, self.graph.initial_theta), state)
        action = self.graph.action(state, self.graph.initial_theta)
        for channel in range(3):
            self.assertLess(
                float(
                    torch.norm(gradient[:, channel] - action[:, channel])
                    / torch.norm(action[:, channel])
                ),
                2e-12,
            )

    def test_affine_enthalpy_agrees_with_continuum_q4(self):
        xy = self.mesh.coordinates
        state = np.column_stack(
            (2e-6 * xy[:, 0] + 1e-6 * xy[:, 1], -3e-6 * xy[:, 1], 20 * xy[:, 0] - 50 * xy[:, 1])
        )
        q4 = q4_reference(self.mesh, pzt4(), 0.001)
        expected = 0.5 * state.ravel() @ q4 @ state.ravel()
        actual = float(self.graph.energy(torch.tensor(state), self.graph.initial_theta))
        self.assertAlmostEqual(actual / expected, 1.0, places=11)

    def test_rigid_rotation_preserves_enthalpy(self):
        angle = np.pi / 6
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        graph = RotatedLatticeGraph(self.mesh, angle)
        rng = np.random.default_rng(30)
        local = rng.normal(size=(self.mesh.n, 3)) * [1e-8, 1e-8, 1.0]
        global_state = local.copy()
        global_state[:, :2] = local[:, :2] @ rotation.T
        before = float(self.graph.energy(torch.tensor(local), self.graph.initial_theta))
        after = float(graph.energy(torch.tensor(global_state), graph.initial_theta))
        self.assertAlmostEqual(after / before, 1.0, places=11)

    def test_disk_coordinates_produce_strictly_bounded_loss(self):
        mapper = AxisymmetricComplexMaterialParameters(torch.tensor(NOMINAL, dtype=torch.float64))
        theta = torch.tensor([0.0, 0.0, 0.0, np.log(0.01), np.log(0.015), 1.0], dtype=torch.float64)
        recovered = mapper.decode(theta)
        self.assertLess(abs(recovered["beta"]), recovered["beta_max"])
        self.assertEqual(recovered["material_coordinate_dimension"], 6)

    def test_random_splits_and_input_ranges(self):
        training = sample_cases(8, 20261021, "training")
        repeated = sample_cases(8, 20261021, "training")
        test = sample_cases(8, 20261101, "test")
        self.assertEqual(training, repeated)
        keys = ("force_N", "voltage_V", "center_mm", "sigma_mm")
        a = np.array([[case[k] for k in keys] for case in training])
        b = np.array([[case[k] for k in keys] for case in test])
        self.assertFalse(np.array_equal(a, b))
        self.assertTrue(np.all(abs(normalize_parameters(a)) <= 1.0))

    def test_architectures_match_paper_parameter_counts(self):
        self.assertEqual(sum(p.numel() for p in predictor("data_gnn").parameters()), 11382)
        self.assertEqual(sum(p.numel() for p in predictor("pi_gnn").parameters()), 11382)
        self.assertEqual(sum(p.numel() for p in predictor("lpm_gnn").parameters()), 5516)


class NonlinearConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.physics = SparsePhysics(20)

    def test_exact_tangent_matches_finite_difference(self):
        physics = self.physics
        rng = np.random.default_rng(71)
        y = rng.normal(size=physics.kh.shape[0]) * 0.1
        direction = np.zeros_like(y)
        direction[physics.free] = rng.normal(size=len(physics.free))
        step = 1e-5
        finite = (physics.action(y + step * direction) - physics.action(y - step * direction))[
            physics.free
        ] / (2 * step)
        exact = physics.jacobian(y) @ direction[physics.free]
        self.assertLess(np.linalg.norm(exact - finite) / np.linalg.norm(exact), 1e-8)

    def test_generated_reference_satisfies_equilibrium_and_load(self):
        physics = self.physics
        case = sample_cases(1, 20261021, "check")[0]
        state, info, _ = physics.solve(case)
        applied, b, lift = load_case(physics.a, case)
        self.assertAlmostEqual(applied[:, 1].sum(), case["force_N"], places=13)
        self.assertTrue(info["converged"])
        self.assertLess(info["relative_residual"], 1e-10)
        np.testing.assert_allclose(state[~physics.a["mask"]], lift[~physics.a["mask"]], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
