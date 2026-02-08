"""
Map video-frame coordinates to a top-down tactical court diagram.

The converter defines 18 reference keypoints on a canonical basketball-court
image, validates detected keypoints via proportional-distance checks, and
computes per-frame homographies to project player foot-positions into the
tactical coordinate system.
"""

from copy import deepcopy

import numpy as np
import cv2

from .homography import Homography
from utils import get_foot_position, measure_distance


class TacticalViewConverter:
    """Convert player positions from camera view to a bird's-eye tactical view.

    Parameters
    ----------
    court_image_path : str
        Path to the court diagram used as the tactical-view background.
    """

    def __init__(self, court_image_path):
        self.court_image_path = court_image_path

        # Tactical-view dimensions (pixels)
        self.width = 300
        self.height = 161

        # Real-world court dimensions
        self.actual_width_in_meters = 28
        self.actual_height_in_meters = 15

        # 18 canonical keypoints on the tactical-view image
        self.key_points = [
            # ---- left edge (6 points) ----
            (0, 0),
            (0, int((0.91 / self.actual_height_in_meters) * self.height)),
            (0, int((5.18 / self.actual_height_in_meters) * self.height)),
            (0, int((10 / self.actual_height_in_meters) * self.height)),
            (0, int((14.1 / self.actual_height_in_meters) * self.height)),
            (0, int(self.height)),
            # ---- centre line (2 points) ----
            (int(self.width / 2), self.height),
            (int(self.width / 2), 0),
            # ---- left free-throw line (2 points) ----
            (
                int((5.79 / self.actual_width_in_meters) * self.width),
                int((5.18 / self.actual_height_in_meters) * self.height),
            ),
            (
                int((5.79 / self.actual_width_in_meters) * self.width),
                int((10 / self.actual_height_in_meters) * self.height),
            ),
            # ---- right edge (6 points) ----
            (self.width, int(self.height)),
            (self.width, int((14.1 / self.actual_height_in_meters) * self.height)),
            (self.width, int((10 / self.actual_height_in_meters) * self.height)),
            (self.width, int((5.18 / self.actual_height_in_meters) * self.height)),
            (self.width, int((0.91 / self.actual_height_in_meters) * self.height)),
            (self.width, 0),
            # ---- right free-throw line (2 points) ----
            (
                int(((self.actual_width_in_meters - 5.79) / self.actual_width_in_meters) * self.width),
                int((5.18 / self.actual_height_in_meters) * self.height),
            ),
            (
                int(((self.actual_width_in_meters - 5.79) / self.actual_width_in_meters) * self.width),
                int((10 / self.actual_height_in_meters) * self.height),
            ),
        ]

    # ------------------------------------------------------------------
    # Keypoint validation
    # ------------------------------------------------------------------

    def validate_keypoints(self, keypoints_list):
        """
        Validates detected keypoints by comparing their proportional distances
        to the tactical view keypoints.

        Args:
            keypoints_list (List[List[Tuple[float, float]]]): A list containing keypoints for each frame.
                Each outer list represents a frame.
                Each inner list contains keypoints as (x, y) tuples.
                A keypoint of (0, 0) indicates that the keypoint is not detected for that frame.

        Returns:
            List[bool]: A list indicating whether each frame's keypoints are valid.
        """
        keypoints_list = deepcopy(keypoints_list)

        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            frame_keypoints = frame_keypoints.xy.tolist()[0]

            detected_indices = [
                i for i, kp in enumerate(frame_keypoints) if kp[0] > 0 and kp[1] > 0
            ]

            # Need at least 3 detected keypoints to validate proportions
            if len(detected_indices) < 3:
                continue

            invalid_keypoints = []

            for i in detected_indices:
                if frame_keypoints[i][0] == 0 and frame_keypoints[i][1] == 0:
                    continue

                other_indices = [
                    idx
                    for idx in detected_indices
                    if idx != i and idx not in invalid_keypoints
                ]
                if len(other_indices) < 2:
                    continue

                j, k = other_indices[0], other_indices[1]

                # Proportional-distance check between detected and tactical keypoints
                d_ij = measure_distance(frame_keypoints[i], frame_keypoints[j])
                d_ik = measure_distance(frame_keypoints[i], frame_keypoints[k])
                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float("inf")
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float("inf")

                    error = abs((prop_detected - prop_tactical) / prop_tactical)

                    if error > 0.8:  # 80 % error margin → invalidate
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xyn[0][i] *= 0
                        invalid_keypoints.append(i)

        return keypoints_list

    # ------------------------------------------------------------------
    # Homography-based projection
    # ------------------------------------------------------------------

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        """
        Transform player positions from video frame coordinates to tactical view coordinates.

        Args:
            keypoints_list (list): List of detected court keypoints for each frame.
            player_tracks (list): List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.

        Returns:
            list: List of dictionaries where each dictionary maps player IDs to their (x, y) positions
                in the tactical view coordinate system. The list index corresponds to the frame number.
        """
        tactical_player_positions = []

        for frame_idx, (frame_keypoints, frame_tracks) in enumerate(
            zip(keypoints_list, player_tracks)
        ):
            tactical_positions = {}
            frame_keypoints = frame_keypoints.xy.tolist()[0]

            if frame_keypoints is None or len(frame_keypoints) == 0:
                tactical_player_positions.append(tactical_positions)
                continue

            valid_indices = [
                i for i, kp in enumerate(frame_keypoints) if kp[0] > 0 and kp[1] > 0
            ]

            # A minimum of 4 point-pairs is required for a reliable homography
            if len(valid_indices) < 4:
                tactical_player_positions.append(tactical_positions)
                continue

            source_points = np.array(
                [frame_keypoints[i] for i in valid_indices], dtype=np.float32
            )
            target_points = np.array(
                [self.key_points[i] for i in valid_indices], dtype=np.float32
            )

            try:
                homography = Homography(source_points, target_points)

                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    player_position = np.array([get_foot_position(bbox)])
                    tactical_position = homography.transform_points(player_position)

                    tx, ty = tactical_position[0]
                    if 0 <= tx <= self.width and 0 <= ty <= self.height:
                        tactical_positions[player_id] = tactical_position[0].tolist()

            except (ValueError, cv2.error):
                pass  # Homography failed — leave this frame empty

            tactical_player_positions.append(tactical_positions)

        return tactical_player_positions
