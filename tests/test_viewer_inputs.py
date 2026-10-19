"""Synthetic vibration-file checks; these never invoke the CUDA renderer."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_play_highres import load_vibration


class VibrationInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.displacement = np.arange(36, dtype=np.float32).reshape(4, 3, 3) / 100

    def save_npy(self, values):
        path = self.root / "vibration.npy"
        np.save(path, values)
        return path

    def test_npy_and_dat_have_identical_frames_and_scaled_values(self):
        npy_path = self.save_npy(self.displacement)
        dat_path = self.root / "vibration.dat"
        self.displacement.tofile(dat_path)
        expected = self.displacement * 2.5
        for path in (npy_path, dat_path):
            with self.subTest(format=path.suffix):
                actual = load_vibration(path, moving_count=3, scale=2.5)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.float32)

    def test_flat_npy_retains_frame_order(self):
        actual = load_vibration(self.save_npy(self.displacement.ravel()), 3)
        np.testing.assert_array_equal(actual, self.displacement)

    def test_single_complete_frame_is_playable(self):
        actual = load_vibration(self.save_npy(self.displacement[:1]), 3)
        self.assertEqual(actual.shape, (1, 3, 3))

    def test_npy_rejects_empty_wrong_count_and_incomplete_frames(self):
        for values in (np.empty((0, 3, 3)), np.empty((4, 2, 3)),
                       np.empty((4, 9)), self.displacement.ravel()[:-1], np.array([])):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                load_vibration(self.save_npy(values), 3)

    def test_dat_rejects_empty_partial_scalar_and_partial_frame(self):
        path = self.root / "vibration.dat"
        for contents in (b"", self.displacement.tobytes()[:-1], self.displacement.tobytes()[:-4]):
            path.write_bytes(contents)
            with self.subTest(bytes=len(contents)), self.assertRaises(ValueError):
                load_vibration(path, 3)

    def test_both_formats_reject_non_finite_values(self):
        for value in (np.nan, np.inf, -np.inf):
            data = self.displacement.copy()
            data[0, 0, 0] = value
            npy_path = self.save_npy(data)
            dat_path = self.root / "vibration.dat"
            data.tofile(dat_path)
            for path in (npy_path, dat_path):
                with self.subTest(format=path.suffix, value=value), self.assertRaises(ValueError):
                    load_vibration(path, 3)

    def test_invalid_counts_scales_and_float32_overflow_are_rejected(self):
        path = self.save_npy(self.displacement)
        for count, scale in ((0, 1), (-1, 1), (3, np.inf), (3, np.nan), (3, 1e100)):
            with self.subTest(count=count, scale=scale), self.assertRaises(ValueError):
                load_vibration(path, count, scale)

    def test_non_real_and_object_npy_are_rejected(self):
        for values in (self.displacement.astype(complex), self.displacement.astype(str),
                       np.full((1, 3, 3), object(), dtype=object)):
            with self.subTest(dtype=values.dtype), self.assertRaises(ValueError):
                load_vibration(self.save_npy(values), 3)

    def test_unknown_suffix_is_rejected(self):
        path = self.root / "vibration.bin"
        self.displacement.tofile(path)
        with self.assertRaises(ValueError):
            load_vibration(path, 3)


if __name__ == "__main__":
    unittest.main()
