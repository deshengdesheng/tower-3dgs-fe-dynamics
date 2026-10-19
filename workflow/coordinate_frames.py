"""Coordinate helpers for presenting the registered tower in physical axes."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def physical_up_in_gaussian_frame(gs_to_fe_transform: str | Path) -> np.ndarray:
    """Map FE +Z back into Gaussian coordinates without modifying model data."""
    matrix = np.loadtxt(Path(gs_to_fe_transform), dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected a 4x4 Gaussian-to-FE transform, got {matrix.shape}")
    linear = matrix[:3, :3]
    if not np.isfinite(linear).all() or abs(float(np.linalg.det(linear))) < 1e-12:
        raise ValueError("Gaussian-to-FE transform has a singular or non-finite linear part")
    up = np.linalg.solve(linear, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    norm = float(np.linalg.norm(up))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("could not derive the physical up direction from the transform")
    return (up / norm).astype(np.float32)


def display_basis(up_vector: np.ndarray) -> np.ndarray:
    """Build a right-handed local basis whose third axis is physical up."""
    up = np.asarray(up_vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(up))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("up vector must be finite and non-zero")
    up /= norm
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, up))) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])
    horizontal_x = reference - np.dot(reference, up) * up
    horizontal_x /= np.linalg.norm(horizontal_x)
    horizontal_y = np.cross(up, horizontal_x)
    horizontal_y /= np.linalg.norm(horizontal_y)
    return np.column_stack((horizontal_x, horizontal_y, up)).astype(np.float32)
