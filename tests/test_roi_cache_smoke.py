import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

from build_fe_irregular_roi_cache import (  # noqa: E402
    apply_transform,
    build_roi_cache,
    sample_fe_centerlines,
)


class RoiCacheSmokeTest(unittest.TestCase):
    def test_transform_and_centerline_sampling(self):
        transform = np.eye(4)
        transform[:3, 3] = [2.0, -1.0, 0.5]
        points = apply_transform(np.array([[0.0, 0.0, 0.0]]), transform)
        np.testing.assert_allclose(points, [[2.0, -1.0, 0.5]])

        nodes = {
            1: np.array([0.0, 0.0, 0.0]),
            2: np.array([1.0, 0.0, 0.0]),
        }
        elements = [(1, 1, 2, 0, 1)]
        sections = {1: ("R", [0.1, 0.2, 0.0, 0.0])}
        samples, radii = sample_fe_centerlines(nodes, elements, sections, axial_step=0.5)

        np.testing.assert_allclose(samples[:, 0], [0.0, 0.5, 1.0])
        self.assertEqual(len(radii), 3)
        self.assertTrue(np.isfinite(samples).all())

    def test_roi_cache_contains_valid_voxel_keys(self):
        points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        keys, origin, dims = build_roi_cache(points, roi_radius=0.05, roi_voxel_size=0.05)

        self.assertGreater(len(keys), 0)
        self.assertEqual(origin.shape, (3,))
        self.assertEqual(dims.shape, (3,))
        self.assertTrue(np.all(keys >= 0))
        self.assertTrue(np.isfinite(origin).all())


if __name__ == "__main__":
    unittest.main()
