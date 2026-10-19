"""Physical display-frame checks using small synthetic transforms."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from workflow.coordinate_frames import display_basis, physical_up_in_gaussian_frame  # noqa: E402


class CoordinateFrameTest(unittest.TestCase):
    def test_inverse_registration_maps_fe_up_back_to_gaussian_frame(self):
        transform = np.array(
            [[0.0, 0.0, -2.0, 0.0], [0.0, 2.0, 0.0, 0.0],
             [2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "gs_to_fe.txt"
            np.savetxt(path, transform)
            up = physical_up_in_gaussian_frame(path)
        np.testing.assert_allclose(up, [1.0, 0.0, 0.0], atol=1e-7)

    def test_display_basis_is_right_handed_and_uses_physical_up(self):
        basis = display_basis(np.array([1.0, 2.0, -3.0]))
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(basis)), 1.0, places=6)
        np.testing.assert_allclose(basis[:, 2], np.array([1.0, 2.0, -3.0]) / np.sqrt(14), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
