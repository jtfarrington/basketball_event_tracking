"""
Basketball shot-attempt detection from ball trajectory and possession data.

A shot is registered when the ball exhibits a significant upward movement
(negative y-delta in image coordinates) while a player recently had possession.
After the shot frame, the detector monitors whether the ball enters a
configurable "scoring zone" near the top of the frame to classify each
attempt as **made** or **missed**.

The detector outputs three parallel frame-level lists:

* ``shot_frames``  — ``-1`` (no event), ``1`` (Team 1 shot), or ``2`` (Team 2 shot)
* ``shot_results`` — ``None``, ``"made"``, or ``"missed"`` (set on the resolution frame)
* ``shot_team``    — team id of the shooter (carried from the possession data)
"""

from utils.bbox_utils import get_center_of_bbox


class ShotDetector:
    """Detect basketball shot attempts from ball trajectory and possession data.

    Parameters
    ----------
    upward_threshold : int
        Minimum upward pixel displacement (in image coords, where *up* means
        a *decrease* in y) over ``lookback_frames`` to trigger a shot.
    min_possession_frames : int
        A player must have held possession for at least this many of the last
        ``possession_lookback`` frames for a shot to be attributed.
    shot_cooldown_frames : int
        Minimum gap between two consecutive shot events.
    scoring_zone_y_ratio : float
        Fraction of frame height (measured from the top) considered the
        "rim / scoring zone".  The ball entering this zone after a shot
        attempt marks the shot as *made*.
    lookback_frames : int
        Number of previous frames over which the upward displacement is
        measured.
    resolution_window : int
        Number of frames after a shot event within which to look for a
        made / missed resolution.
    possession_lookback : int
        Number of recent frames to search for the last player with possession.
    """

    def __init__(
        self,
        upward_threshold=40,
        min_possession_frames=5,
        shot_cooldown_frames=30,
        scoring_zone_y_ratio=0.25,
        lookback_frames=8,
        resolution_window=30,
        possession_lookback=15,
    ):
        self.upward_threshold = upward_threshold
        self.min_possession_frames = min_possession_frames
        self.shot_cooldown_frames = shot_cooldown_frames
        self.scoring_zone_y_ratio = scoring_zone_y_ratio
        self.lookback_frames = lookback_frames
        self.resolution_window = resolution_window
        self.possession_lookback = possession_lookback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ball_center_y(self, ball_tracks, frame_num):
        """Return the y-coordinate of the ball centre, or ``None``."""
        ball_info = ball_tracks[frame_num].get(1, {})
        if not ball_info:
            return None
        bbox = ball_info.get("bbox", [])
        if not bbox:
            return None
        _, cy = get_center_of_bbox(bbox)
        return cy

    def _find_recent_possessor(self, ball_acquisition, player_assignment, frame_num):
        """Look back to find the most recent player with possession and their team.

        Returns
        -------
        tuple[int, int]
            ``(player_id, team_id)`` or ``(-1, -1)`` if nobody had the ball recently.
        """
        start = max(0, frame_num - self.possession_lookback)
        for f in range(frame_num, start - 1, -1):
            pid = ball_acquisition[f]
            if pid != -1:
                team_id = player_assignment[f].get(pid, -1)
                if team_id != -1:
                    return pid, team_id
        return -1, -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_shots(self, ball_tracks, ball_acquisition, player_assignment, frame_height):
        """Run shot detection across every frame.

        Parameters
        ----------
        ball_tracks : list[dict]
            Per-frame ball bounding-box dictionaries (same format used elsewhere).
        ball_acquisition : list[int]
            Per-frame possessing player id (``-1`` = nobody).
        player_assignment : list[dict]
            Per-frame ``{player_id: team_id}`` mappings.
        frame_height : int
            Height of the video frames in pixels (used to compute the scoring zone).

        Returns
        -------
        tuple[list[int], list[str | None], list[int]]
            ``(shot_frames, shot_results, shot_team)`` — see module docstring.
        """
        num_frames = len(ball_tracks)
        shot_frames = [-1] * num_frames
        shot_results = [None] * num_frames
        shot_team = [-1] * num_frames

        scoring_zone_y = int(frame_height * self.scoring_zone_y_ratio)
        last_shot_frame = -self.shot_cooldown_frames  # allow the first shot immediately

        for frame_num in range(self.lookback_frames, num_frames):
            # --- cooldown guard ---
            if (frame_num - last_shot_frame) < self.shot_cooldown_frames:
                continue

            # --- compute upward displacement ---
            current_y = self._get_ball_center_y(ball_tracks, frame_num)
            past_y = self._get_ball_center_y(
                ball_tracks, frame_num - self.lookback_frames
            )
            if current_y is None or past_y is None:
                continue

            upward_displacement = past_y - current_y  # positive = ball moving up

            if upward_displacement < self.upward_threshold:
                continue

            # --- attribute the shot to a player / team ---
            shooter_id, team_id = self._find_recent_possessor(
                ball_acquisition, player_assignment, frame_num
            )
            if team_id == -1:
                continue  # cannot attribute — skip

            # Register the shot event
            shot_frames[frame_num] = team_id
            shot_team[frame_num] = team_id
            last_shot_frame = frame_num

            # --- resolve made / missed within the resolution window ---
            made = False
            resolve_end = min(num_frames, frame_num + self.resolution_window)
            for rf in range(frame_num + 1, resolve_end):
                ry = self._get_ball_center_y(ball_tracks, rf)
                if ry is not None and ry < scoring_zone_y:
                    made = True
                    break

            shot_results[frame_num] = "made" if made else "missed"

        return shot_frames, shot_results, shot_team
