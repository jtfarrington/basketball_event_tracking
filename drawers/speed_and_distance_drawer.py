"""Render per-player speed (km/h) and cumulative distance (m) labels."""

import cv2

'''
class SpeedAndDistanceDrawer:
    def __init__(self):
        pass

    def draw(self, video_frames, player_tracks, player_distances_per_frame, player_speed_per_frame):
        """Draw speed and distance labels beneath each player.

        Parameters
        ----------
        video_frames : list[numpy.ndarray]
        player_tracks : list[dict]
        player_distances_per_frame : list[dict]
        player_speed_per_frame : list[dict]

        Returns
        -------
        list[numpy.ndarray]
        """
        output_video_frames = []
        total_distances = {}

        for frame, tracks, distances, speeds in zip(
            video_frames, player_tracks, player_distances_per_frame, player_speed_per_frame
        ):
            output_frame = frame.copy()

            # Accumulate total distance per player
            for player_id, distance in distances.items():
                total_distances[player_id] = total_distances.get(player_id, 0) + distance

            # Draw labels
            for player_id, bbox_data in tracks.items():
                x1, y1, x2, y2 = bbox_data["bbox"]
                position = [int((x1 + x2) / 2), int(y2) + 40]

                speed = speeds.get(player_id)
                distance = total_distances.get(player_id)

                if speed is not None:
                    cv2.putText(
                        output_frame,
                        f"{speed:.2f} km/h",
                        tuple(position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )
                if distance is not None:
                    cv2.putText(
                        output_frame,
                        f"{distance:.2f} m",
                        (position[0], position[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )

            output_video_frames.append(output_frame)

        return output_video_frames
'''