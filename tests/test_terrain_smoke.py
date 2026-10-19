import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.neighbors import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extended_more_layers_param as terrain


class TerrainSmokeTest(unittest.TestCase):
    def setUp(self):
        terrain._kriging_cache.clear()

    def test_flat_reference_height_and_z_rotation(self):
        reference = np.array([[0., 0., 2.], [1., 0., 2.],
                              [0., 1., 2.], [1., 1., 2.]])
        height = terrain.query_kriging_height(.5, .5, KDTree(reference[:, :2]), reference, k=4)
        self.assertAlmostEqual(height, 2.)
        rotated = terrain.rotate_points_around_center(
            np.array([[1., 0., 2.]]), np.array([0., 0., 0.]), np.pi / 2)
        np.testing.assert_allclose(rotated, [[0., 1., 2.]], atol=1e-12)

    def test_facing_correction_and_quaternion_agree(self):
        angle = terrain.calculate_rotation_for_consistent_facing(
            (1., 0.), (0., 1.), (0., 0.))
        self.assertAlmostEqual(angle, np.pi / 2)
        q = terrain.rotate_quaternion_around_z(np.array([1., 0., 0., 0.]), angle)
        np.testing.assert_allclose(q, [np.sqrt(.5), 0., 0., np.sqrt(.5)], atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.)


if __name__ == '__main__':
    unittest.main()
