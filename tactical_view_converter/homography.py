"""
Thin wrapper around :pyfunc:`cv2.findHomography` and :pyfunc:`cv2.perspectiveTransform`.

A :class:`Homography` object is initialised once with matched source/target
point pairs and can then be re-used to transform arbitrary 2-D points from
the source coordinate system to the target.
"""

import numpy as np
import cv2


class Homography:
    """Compute and apply a planar homography.

    Parameters
    ----------
    source : np.ndarray
        ``(N, 2)`` array of 2-D source points.
    target : np.ndarray
        ``(N, 2)`` array of corresponding 2-D target points.

    Raises
    ------
    ValueError
        If shapes are inconsistent or the homography cannot be computed.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Project *points* through the homography.

        Parameters
        ----------
        points : np.ndarray
            ``(M, 2)`` array of 2-D coordinates.

        Returns
        -------
        np.ndarray
            ``(M, 2)`` transformed coordinates (float32).
        """
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)
