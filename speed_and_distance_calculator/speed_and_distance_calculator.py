"""
Player kinematics: distance travelled and instantaneous speed.

Positions in the tactical view (pixel coordinates) are converted to metres
using the known court dimensions, and speed is computed over a configurable
rolling window of frames.
"""

from utils import measure_distance


class SpeedAndDistanceCalculator:
    """Convert tactical-view pixel displacements into real-world distances and speeds.

    Parameters
    ----------
    width_in_pixels : int
        Width of the tactical-view image in pixels.
    height_in_pixels : int
        Height of the tactical-view image in pixels.
    width_in_meters : float
        Real-world court width in metres.
    height_in_meters : float
        Real-world court height in metres.
    """

    def __init__(
        self,
        width_in_pixels,
        height_in_pixels,
        width_in_meters,
        height_in_meters,
    ):
        self.width_in_pixels = width_in_pixels
        self.height_in_pixels = height_in_pixels
        self.width_in_meters = width_in_meters
        self.height_in_meters = height_in_meters

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def calculate_distance(self, tactical_player_positions):
        """Compute per-frame distance for each player from consecutive positions.

        Parameters
        ----------
        tactical_player_positions : list[dict]
            Per-frame mapping of ``player_id → (x, y)`` in tactical-view pixels.

        Returns
        -------
        list[dict]
            Per-frame mapping of ``player_id → distance_in_metres``.
        """
        previous_players_position = {}
        output_distances = []

        for frame_number, frame_positions in enumerate(tactical_player_positions):
            output_distances.append({})

            for player_id, current_pos in frame_positions.items():
                if player_id in previous_players_position:
                    prev_pos = previous_players_position[player_id]
                    meter_distance = self.calculate_meter_distance(prev_pos, current_pos)
                    output_distances[frame_number][player_id] = meter_distance

                previous_players_position[player_id] = current_pos

        return output_distances

    def calculate_meter_distance(self, previous_pixel_position, current_pixel_position):
        """Convert a pixel-space displacement to metres.

        A 0.4× scaling factor is applied as an empirical correction for
        the projection distortion.
        """
        prev_x, prev_y = previous_pixel_position
        curr_x, curr_y = current_pixel_position

        prev_mx = prev_x * self.width_in_meters / self.width_in_pixels
        prev_my = prev_y * self.height_in_meters / self.height_in_pixels
        curr_mx = curr_x * self.width_in_meters / self.width_in_pixels
        curr_my = curr_y * self.height_in_meters / self.height_in_pixels

        meter_distance = measure_distance((curr_mx, curr_my), (prev_mx, prev_my))
        meter_distance = meter_distance * 0.4
        return meter_distance

    # ------------------------------------------------------------------
    # Speed
    # ------------------------------------------------------------------

    def calculate_speed(self, distances, fps=30):
        """
        Calculate player speeds based on distances covered over the last 5 frames.

        Args:
            distances (list): List of dictionaries containing distance per player per frame,
                            as output by calculate_distance method.
            fps (float): Frames per second of the video, used to calculate elapsed time.

        Returns:
            list: List of dictionaries where each dictionary maps player_id to their
                speed in km/h at that frame.
        """
        speeds = []
        window_size = 5

        for frame_idx in range(len(distances)):
            speeds.append({})

            for player_id in distances[frame_idx].keys():
                start_frame = max(0, frame_idx - (window_size * 3) + 1)

                total_distance = 0
                frames_present = 0
                last_frame_present = None

                for i in range(start_frame, frame_idx + 1):
                    if player_id in distances[i]:
                        if last_frame_present is not None:
                            total_distance += distances[i][player_id]
                            frames_present += 1
                        last_frame_present = i

                if frames_present >= window_size:
                    time_in_seconds = frames_present / fps
                    time_in_hours = time_in_seconds / 3600

                    if time_in_hours > 0:
                        speeds[frame_idx][player_id] = (total_distance / 1000) / time_in_hours
                    else:
                        speeds[frame_idx][player_id] = 0
                else:
                    speeds[frame_idx][player_id] = 0

        return speeds
