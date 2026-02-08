"""Draw a semi-transparent overlay showing cumulative shot-attempt statistics."""

import cv2


class ShotDrawer:
    """Render shot-attempt counts and made/missed breakdowns per team.

    The overlay appears in the top-right region of each frame and shows
    running totals computed up to the current frame.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, shot_frames, shot_results):
        """Compute per-team shot totals and make/miss counts.

        Parameters
        ----------
        shot_frames : list[int]
            ``-1`` (no shot), ``1`` (Team 1), or ``2`` (Team 2).
        shot_results : list[str | None]
            ``None``, ``"made"``, or ``"missed"``.

        Returns
        -------
        dict
            ``{1: {"attempts": int, "made": int, "missed": int},
              2: {"attempts": int, "made": int, "missed": int}}``
        """
        stats = {
            1: {"attempts": 0, "made": 0, "missed": 0},
            2: {"attempts": 0, "made": 0, "missed": 0},
        }
        for sf, sr in zip(shot_frames, shot_results):
            if sf in (1, 2):
                stats[sf]["attempts"] += 1
                if sr == "made":
                    stats[sf]["made"] += 1
                elif sr == "missed":
                    stats[sf]["missed"] += 1
        return stats

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, video_frames, shot_frames, shot_results):
        """Draw shot statistics on every frame (skipping frame 0).

        Parameters
        ----------
        video_frames : list[numpy.ndarray]
        shot_frames : list[int]
        shot_results : list[str | None]

        Returns
        -------
        list[numpy.ndarray]
            Annotated frames (one fewer than input, frame 0 is skipped).
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            if frame_num == 0:
                continue
            frame_drawn = self.draw_frame(frame, frame_num, shot_frames, shot_results)
            output_video_frames.append(frame_drawn)
        return output_video_frames

    def draw_frame(self, frame, frame_num, shot_frames, shot_results):
        """Render the shot-stats overlay on a single frame."""
        overlay = frame.copy()
        font_scale = 0.6
        font_thickness = 2

        frame_height, frame_width = overlay.shape[:2]

        # Position: top-right area
        rect_x1 = int(frame_width * 0.60)
        rect_y1 = int(frame_height * 0.02)
        rect_x2 = int(frame_width * 0.99)
        rect_y2 = int(frame_height * 0.14)
        text_x = int(frame_width * 0.62)
        text_y1 = int(frame_height * 0.06)
        text_y2 = int(frame_height * 0.12)

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        stats = self.get_stats(
            shot_frames[: frame_num + 1], shot_results[: frame_num + 1]
        )

        t1 = stats[1]
        t2 = stats[2]

        cv2.putText(
            frame,
            f"Team 1 Shots: {t1['attempts']}  Made: {t1['made']}  Missed: {t1['missed']}",
            (text_x, text_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )
        cv2.putText(
            frame,
            f"Team 2 Shots: {t2['attempts']}  Made: {t2['made']}  Missed: {t2['missed']}",
            (text_x, text_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )

        return frame
