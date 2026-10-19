"""CPU geometry checks using synthetic points and Gaussian checkpoints only.

Expected rotations and displacements come from analytic geometry, rather than
from another copy of the implementation. No gsplat renderer or GPU is needed.
"""

import contextlib
import io
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import anisotropic_kernel as kernel  # noqa: E402
import apply_transform_with_anisotropic_kernel as mapping  # noqa: E402
import merged_gaussian_transform as geometry  # noqa: E402


def quaternion_matrix_wxyz(quaternion):
    """Independent matrix oracle for a unit Hamilton quaternion."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    vector = np.array([x, y, z])
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        (w * w - vector @ vector) * np.eye(3)
        + 2.0 * np.outer(vector, vector)
        + 2.0 * w * cross
    )


def tensor(values):
    return torch.tensor(values, dtype=torch.float32, device="cpu")


class ImportSafetyTest(unittest.TestCase):
    def test_import_preserves_gradient_state_and_does_not_create_output_files(self):
        # A fresh process catches accidental CLI argument parsing and global
        # torch state changes that an already imported module would conceal.
        code = """
import importlib
import pathlib
import sys
import torch

sys.path.insert(0, sys.argv[1])
torch.set_grad_enabled(sys.argv[2] == "enabled")
before_gradient_state = torch.is_grad_enabled()
before_cwd = pathlib.Path.cwd()
before_files = set(before_cwd.iterdir())
sys.argv = ["import-check", "--not-a-valid-cli-option"]
for name in (
    "merged_gaussian_transform",
    "anisotropic_kernel",
    "apply_transform_with_anisotropic_kernel",
):
    importlib.import_module(name)
assert torch.is_grad_enabled() == before_gradient_state
assert pathlib.Path.cwd() == before_cwd
assert set(before_cwd.iterdir()) == before_files
print("import-safe")
"""
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        for gradient_state in ("enabled", "disabled"):
            with self.subTest(gradient_state=gradient_state), tempfile.TemporaryDirectory() as temp_dir:
                result = subprocess.run(
                    [sys.executable, "-c", code, str(REPO_ROOT), gradient_state],
                    cwd=temp_dir,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertEqual(result.stdout.strip(), "import-safe")


class QuaternionAndPolarTest(unittest.TestCase):
    def test_vector_rotations_include_antiparallel_case(self):
        source = tensor([1.0, 0.0, 0.0])
        for target in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]):
            with self.subTest(target=target):
                quaternion = geometry.quaternion_from_vectors(source, tensor(target))
                rotation = quaternion_matrix_wxyz(quaternion.numpy())
                np.testing.assert_allclose(rotation @ source.numpy(), target, atol=2e-6)
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=2e-6)
                self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=5)

    def test_rotation_matrix_conversion_preserves_quarter_and_half_turns(self):
        matrices = np.array(
            [
                np.eye(3),
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                np.diag([1.0, -1.0, -1.0]),
                np.diag([-1.0, 1.0, -1.0]),
                np.diag([-1.0, -1.0, 1.0]),
            ]
        )
        quaternions = geometry.rotation_matrix_to_quaternion(tensor(matrices))
        for matrix, quaternion in zip(matrices, quaternions.numpy()):
            np.testing.assert_allclose(quaternion_matrix_wxyz(quaternion), matrix, atol=2e-6)

    def test_hamilton_products_agree_with_matrix_composition_in_both_layouts(self):
        half = math.sqrt(0.5)
        qx = tensor([[half, half, 0.0, 0.0]])
        qz = tensor([[half, 0.0, 0.0, half]])
        expected = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        product_wxyz = geometry.quaternion_multiply(qz, qx)
        product_xyzw = mapping.quaternion_multiply(qz[:, [1, 2, 3, 0]], qx[:, [1, 2, 3, 0]])
        np.testing.assert_allclose(quaternion_matrix_wxyz(product_wxyz[0].numpy()), expected, atol=2e-6)
        np.testing.assert_allclose(
            quaternion_matrix_wxyz(product_xyzw[0, [3, 0, 1, 2]].numpy()), expected, atol=2e-6
        )

    def test_rotated_ellipsoid_covariance_matches_analytic_axes(self):
        half = math.sqrt(0.5)
        quaternion = tensor([[half, 0.0, 0.0, half]])
        covariance = geometry.build_covariance_from_quats_scales(quaternion, torch.log(tensor([[2.0, 3.0, 4.0]])))
        np.testing.assert_allclose(covariance[0].numpy(), np.diag([9.0, 4.0, 16.0]), atol=3e-6)

    def test_zero_jacobian_produces_identity_deformation(self):
        deformation, rotation, stretch = geometry.build_deformation_matrix_polar(torch.zeros((2, 3, 3)))
        expected = np.repeat(np.eye(3)[None], 2, axis=0)
        np.testing.assert_allclose(deformation.numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(rotation.numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(stretch.numpy(), np.ones((2, 3)), atol=1e-6)

    def test_right_polar_recovers_known_rotation_and_positive_stretch(self):
        rotation = tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        stretch = torch.diag_embed(tensor([[1.5, 0.7, 1.1]]))
        deformation = rotation @ stretch
        actual_rotation, actual_stretch = geometry.polar_decompose(deformation)
        np.testing.assert_allclose(actual_rotation.numpy(), rotation.numpy(), atol=2e-6)
        np.testing.assert_allclose(actual_stretch.numpy(), stretch.numpy(), atol=2e-6)
        np.testing.assert_allclose((actual_rotation @ actual_stretch).numpy(), deformation.numpy(), atol=2e-6)

    def test_constant_translation_has_no_displacement_gradient(self):
        points = tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        edges = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        for displacement in (torch.zeros_like(points), tensor([[0.2, -0.3, 0.4]]).repeat(4, 1)):
            actual = geometry.estimate_jacobian_batch(points, displacement, edges)
            np.testing.assert_allclose(actual.numpy(), np.zeros((4, 3, 3)), atol=2e-6)

    def test_nonzero_member_gradient_is_kept_for_oblique_member(self):
        # This member is perpendicular to [1, 1, 1], but it still has an edge.
        # The observable axial map sends [1,-1,0] to [0.2,0,0]. The full
        # minimum-norm gradient is checked in float64 because one edge leaves
        # two directions unobserved and the regularized system is ill-conditioned.
        points = tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]])
        displacement = tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        edges = torch.tensor([[0, 1]])
        actual = geometry.estimate_jacobian_batch(points, displacement, edges)
        axial_map = actual @ (points[1] - points[0])
        np.testing.assert_allclose(axial_map.numpy(), [[0.2, 0.0, 0.0]] * 2, atol=2e-6)

        actual_double = geometry.estimate_jacobian_batch(points.double(), displacement.double(), edges)
        expected = np.repeat(np.array([[[0.1, -0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]), 2, axis=0)
        np.testing.assert_allclose(actual_double.numpy(), expected, atol=2e-6)


class AnisotropicPropagationTest(unittest.TestCase):
    def test_kernel_prefers_axial_over_equal_radial_separation(self):
        points = tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
        directions = tensor([[1.0, 0.0, 0.0]]).repeat(3, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            weights = kernel.compute_anisotropic_weights(points, directions, sigma_para=1.0, sigma_perp=0.2)
        self.assertGreater(float(weights[0, 1]), float(weights[0, 2]))
        self.assertGreater(float(weights[0, 0]), float(weights[0, 1]))
        np.testing.assert_allclose(weights.sum(dim=1).numpy(), np.ones(3), atol=1e-6)
        self.assertTrue(bool(torch.isfinite(weights).all()))

    def test_direction_penalty_preserves_antiparallel_alignment(self):
        points = tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [-0.5, 0.0, 0.0]])
        directions = tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        with contextlib.redirect_stdout(io.StringIO()):
            unpenalized = kernel.compute_anisotropic_weights(points, directions, sigma_para=1.0, sigma_perp=1.0, beta=0.0)
            penalized = kernel.compute_anisotropic_weights(points, directions, sigma_para=1.0, sigma_perp=1.0, beta=5.0)
        self.assertAlmostEqual(float(unpenalized[0, 1]), float(unpenalized[0, 2]), places=6)
        self.assertGreater(float(penalized[0, 1]), 100.0 * float(penalized[0, 2]))
        self.assertAlmostEqual(float(penalized[0, 1]), float(penalized[0, 3]), places=6)

    def propagate(self, targets, displacement):
        sources = tensor([[-0.3, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.3, 0.0, 0.0]])
        directions = tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)
        neutral_rotation = tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(4, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            return mapping.compute_anisotropic_interpolation(
                targets, sources, directions, displacement.repeat(4, 1),
                neutral_rotation, torch.ones((4, 3)), radius=1.0,
                sigma_ratio=4.0, beta=5.0, batch_size=2,
            )

    def test_neutral_field_stays_neutral_inside_and_outside_support(self):
        targets = tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        displacement, rotation, stretch, directions = self.propagate(targets, tensor([[0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(displacement.numpy(), np.zeros((2, 3)), atol=1e-6)
        np.testing.assert_allclose(rotation.numpy(), [[0.0, 0.0, 0.0, 1.0]] * 2, atol=1e-6)
        np.testing.assert_allclose(stretch.numpy(), np.ones((2, 3)), atol=1e-6)
        np.testing.assert_allclose(directions.numpy(), [[1.0, 0.0, 0.0]] * 2, atol=1e-6)

    def test_constant_translation_is_preserved_with_full_support(self):
        translation = tensor([[0.2, -0.3, 0.4]])
        actual, rotation, stretch, _ = self.propagate(tensor([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]]), translation)
        np.testing.assert_allclose(actual.numpy(), translation.repeat(2, 1).numpy(), atol=1e-6)
        np.testing.assert_allclose(rotation.numpy(), [[0.0, 0.0, 0.0, 1.0]] * 2, atol=1e-6)
        np.testing.assert_allclose(stretch.numpy(), np.ones((2, 3)), atol=1e-6)

    def test_file_mapping_preserves_gaussian_parameters_for_translation(self):
        # Only small synthetic checkpoints are written, inside a temporary dir.
        half = math.sqrt(0.5)
        means = tensor([[-0.3, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.3, 0.0, 0.0]])
        splats = {
            "means": means,
            "quats": tensor([[half, half, 0.0, 0.0]]).repeat(4, 1),
            "scales": torch.log(tensor([[0.05, 0.02, 0.01]])).repeat(4, 1),
            "opacities": torch.full((4,), 0.7),
            "sh0": torch.full((4, 1, 3), 0.2),
            "shN": torch.zeros((4, 0, 3)),
        }
        for translation in ([0.0, 0.0, 0.0], [0.2, -0.3, 0.4]):
            with self.subTest(translation=translation), tempfile.TemporaryDirectory() as temp_dir:
                directory = Path(temp_dir)
                checkpoint = directory / "synthetic.pt"
                transform = directory / "synthetic_transform.npz"
                output = directory / "mapped.pt"
                output_transform = directory / "mapped_transform.npz"
                torch.save({"splats": splats}, checkpoint)
                np.savez(
                    transform,
                    displacement_vectors=np.tile(translation, (4, 1)).astype(np.float32),
                    rotation_delta=np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)).astype(np.float32),
                    scale_factors=np.ones((4, 3), dtype=np.float32),
                    deformation_mode="delta",
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    mapping.apply_transform_with_anisotropic_kernel(
                        checkpoint, checkpoint, transform, output, output_transform,
                        radius=1.0, device=torch.device("cpu"), use_keops=False,
                        k_neighbors=3, batch_size=2,
                    )
                mapped = torch.load(output, map_location="cpu")["splats"]
                np.testing.assert_allclose(mapped["means"].numpy(), means.numpy() + translation, atol=1e-6)
                for key in ("quats", "scales", "opacities", "sh0", "shN"):
                    np.testing.assert_allclose(mapped[key].numpy(), splats[key].numpy(), atol=1e-6)
                with np.load(output_transform) as relation:
                    np.testing.assert_allclose(relation["displacement_vectors"], np.tile(translation, (4, 1)), atol=1e-6)
                    np.testing.assert_allclose(relation["rotation_delta"], [[1.0, 0.0, 0.0, 0.0]] * 4, atol=1e-6)
                    np.testing.assert_allclose(relation["scale_factors"], np.ones((4, 3)), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
