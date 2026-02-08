"""
Bounding-box geometry helpers used throughout the pipeline.

Every function operates on bounding boxes represented as ``(x1, y1, x2, y2)``
where ``(x1, y1)`` is the top-left corner and ``(x2, y2)`` is the bottom-right.
"""

import math


def get_center_of_bbox(bbox):
    """Return the ``(cx, cy)`` centre point of a bounding box.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box as ``(x1, y1, x2, y2)``.

    Returns
    -------
    tuple[int, int]
        Integer centre coordinates.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    """Return the pixel width of a bounding box.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box as ``(x1, y1, x2, y2)``.

    Returns
    -------
    int
        Width in pixels.
    """
    return int(bbox[2] - bbox[0])


def get_foot_position(bbox):
    """Return the bottom-centre point of a bounding box (the "foot" position).

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box as ``(x1, y1, x2, y2)``.

    Returns
    -------
    tuple[int, int]
        ``(x_centre, y_bottom)`` coordinates.
    """
    x1, _, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def measure_distance(p1, p2):
    """Compute the Euclidean distance between two 2-D points.

    Parameters
    ----------
    p1, p2 : tuple[float, float]
        ``(x, y)`` coordinates.

    Returns
    -------
    float
        Euclidean distance.
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
