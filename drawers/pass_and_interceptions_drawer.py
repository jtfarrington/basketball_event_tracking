"""Draw a semi-transparent overlay with cumulative pass and interception counts."""

import cv2
import numpy as np


class PassInterceptionDrawer:
    """
    A class responsible for calculating and drawing pass and interception statistics
    on a sequence of video frames.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, passes, interceptions):
        """
        Calculate the number of passes and interceptions for Team 1 and Team 2.

        Args:
            passes (list): A list of integers representing pass events at each frame.
            interceptions (list): A list of integers representing interception events at each frame.

        Returns:
            tuple: A tuple of four integers (team1_pass_total, team2_pass_total,
                team1_interception_total, team2_interception_total).
        """
        team1_passes = sum(1 for p in passes if p == 1)
        team2_passes = sum(1 for p in passes if p == 2)
        team1_interceptions = sum(1 for i in interceptions if i == 1)
        team2_interceptions = sum(1 for i in interceptions if i == 2)
        return team1_passes, team2_passes, team1_interceptions, team2_interceptions

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, video_frames, passes, interceptions):
        """
        Draw pass and interception statistics on a list of video frames.

        Args:
            video_frames (list): A list of frames on which to draw.
            passes (list): A list of integers representing pass events at each frame.
            interceptions (list): A list of integers representing interception events at each frame.

        Returns:
            list: A list of frames with pass and interception statistics drawn on them.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            if frame_num == 0:
                continue
            frame_drawn = self.draw_frame(frame, frame_num, passes, interceptions)
            output_video_frames.append(frame_drawn)
        return output_video_frames

    def draw_frame(self, frame, frame_num, passes, interceptions):
        """
        Draw a semi-transparent overlay of pass and interception counts on a single frame.

        Args:
            frame (numpy.ndarray): The current video frame.
            frame_num (int): The index of the current frame.
            passes (list): A list of pass events up to this frame.
            interceptions (list): A list of interception events up to this frame.

        Returns:
            numpy.ndarray: The frame with the overlay and statistics.
        """
        overlay = frame.copy()
        font_scale = 0.7
        font_thickness = 2

        frame_height, frame_width = overlay.shape[:2]
        rect_x1 = int(frame_width * 0.16)
        rect_y1 = int(frame_height * 0.75)
        rect_x2 = int(frame_width * 0.55)
        rect_y2 = int(frame_height * 0.90)
        text_x = int(frame_width * 0.19)
        text_y1 = int(frame_height * 0.80)
        text_y2 = int(frame_height * 0.88)

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        passes_till_frame = passes[: frame_num + 1]
        interceptions_till_frame = interceptions[: frame_num + 1]

        t1p, t2p, t1i, t2i = self.get_stats(passes_till_frame, interceptions_till_frame)

        cv2.putText(
            frame,
            f"Team 1 - Passes: {t1p} Interceptions: {t1i}",
            (text_x, text_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )
        cv2.putText(
            frame,
            f"Team 2 - Passes: {t2p} Interceptions: {t2i}",
            (text_x, text_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )

        return frame
