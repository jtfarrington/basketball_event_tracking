"""Draw a semi-transparent overlay showing cumulative ball-control percentages."""

import cv2
import numpy as np


class TeamBallControlDrawer:
    """
    A class responsible for calculating and drawing team ball control statistics on video frames.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_team_ball_control(self, player_assignment, ball_aquisition):
        """
        Calculate which team has ball control for each frame.

        Args:
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.
            ball_aquisition (list): A list indicating which player has possession of the ball in each frame.

        Returns:
            numpy.ndarray: An array indicating which team has ball control for each frame
                (1 for Team 1, 2 for Team 2, -1 for no control).
        """
        team_ball_control = []
        for assignment_frame, acquisition_frame in zip(player_assignment, ball_aquisition):
            if acquisition_frame == -1:
                team_ball_control.append(-1)
                continue
            if acquisition_frame not in assignment_frame:
                team_ball_control.append(-1)
                continue
            if assignment_frame[acquisition_frame] == 1:
                team_ball_control.append(1)
            else:
                team_ball_control.append(2)

        return np.array(team_ball_control)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, video_frames, player_assignment, ball_aquisition):
        """
        Draw team ball control statistics on a list of video frames.

        Args:
            video_frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.
            ball_aquisition (list): A list indicating which player has possession of the ball in each frame.

        Returns:
            list: A list of frames with team ball control statistics drawn on them.
        """
        team_ball_control = self.get_team_ball_control(player_assignment, ball_aquisition)

        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            if frame_num == 0:
                continue
            frame_drawn = self.draw_frame(frame, frame_num, team_ball_control)
            output_video_frames.append(frame_drawn)
        return output_video_frames

    def draw_frame(self, frame, frame_num, team_ball_control):
        """
        Draw a semi-transparent overlay of team ball control percentages on a single frame.

        Args:
            frame (numpy.ndarray): The current video frame on which the overlay will be drawn.
            frame_num (int): The index of the current frame.
            team_ball_control (numpy.ndarray): An array indicating which team has ball control for each frame.

        Returns:
            numpy.ndarray: The frame with the semi-transparent overlay and statistics.
        """
        overlay = frame.copy()
        font_scale = 0.7
        font_thickness = 2

        frame_height, frame_width = overlay.shape[:2]
        rect_x1 = int(frame_width * 0.60)
        rect_y1 = int(frame_height * 0.75)
        rect_x2 = int(frame_width * 0.99)
        rect_y2 = int(frame_height * 0.90)
        text_x = int(frame_width * 0.63)
        text_y1 = int(frame_height * 0.80)
        text_y2 = int(frame_height * 0.88)

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        control_slice = team_ball_control[: frame_num + 1]
        total = control_slice.shape[0]
        team_1_pct = control_slice[control_slice == 1].shape[0] / total if total else 0
        team_2_pct = control_slice[control_slice == 2].shape[0] / total if total else 0

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1_pct * 100:.2f}%",
            (text_x, text_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2_pct * 100:.2f}%",
            (text_x, text_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )

        return frame
