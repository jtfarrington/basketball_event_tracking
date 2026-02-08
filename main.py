"""
Basketball Video Analysis — main pipeline entry-point.

Usage::

    python main.py input_video.mp4 \\
        --output_video output_videos/output.avi \\
        --stub_path stubs/

The pipeline runs detection, tracking, team assignment, possession analysis,
event detection (passes, interceptions, shots), tactical-view projection,
speed/distance computation, and finally renders all annotation layers onto
the video frames before writing the output file.
"""

import os
import argparse

from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from shot_detector import ShotDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    CourtKeypointDrawer,
    TeamBallControlDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    ShotDrawer,
)
from configs import (
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH,
)


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Basketball Video Analysis")
    parser.add_argument("input_video", type=str, help="/input_video")
    parser.add_argument(
        "--output_video",
        type=str,
        default=OUTPUT_VIDEO_PATH,
        help="/output_video",
    )
    parser.add_argument(
        "--stub_path",
        type=str,
        default=STUBS_DEFAULT_PATH,
        help="/stubs",
    )
    return parser.parse_args()


# ======================================================================
# Pipeline
# ======================================================================

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Read video
    # ------------------------------------------------------------------
    video_frames = read_video(args.input_video)

    # ------------------------------------------------------------------
    # 2. Detection & tracking
    # ------------------------------------------------------------------
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)

    player_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=os.path.join(args.stub_path, "player_track_stubs.pkl"),
    )

    ball_tracks = ball_tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=os.path.join(args.stub_path, "ball_track_stubs.pkl"),
    )

    # ------------------------------------------------------------------
    # 3. Court keypoints
    # ------------------------------------------------------------------
    court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)

    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(
        video_frames,
        read_from_stub=True,
        stub_path=os.path.join(args.stub_path, "court_key_points_stub.pkl"),
    )

    # ------------------------------------------------------------------
    # 4. Ball post-processing
    # ------------------------------------------------------------------
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # ------------------------------------------------------------------
    # 5. Team assignment
    # ------------------------------------------------------------------
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(
        video_frames,
        player_tracks,
        read_from_stub=True,
        stub_path=os.path.join(args.stub_path, "player_assignment_stub.pkl"),
    )

    # ------------------------------------------------------------------
    # 6. Ball possession
    # ------------------------------------------------------------------
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(
        player_tracks, ball_tracks
    )

    # ------------------------------------------------------------------
    # 7. Passes & interceptions
    # ------------------------------------------------------------------
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(
        ball_aquisition, player_assignment
    )
    interceptions = pass_and_interception_detector.detect_interceptions(
        ball_aquisition, player_assignment
    )

    # ------------------------------------------------------------------
    # 8. Shot detection  ★ NEW
    # ------------------------------------------------------------------
    shot_detector = ShotDetector()
    frame_height = video_frames[0].shape[0]
    shot_frames, shot_results, shot_team = shot_detector.detect_shots(
        ball_tracks, ball_aquisition, player_assignment, frame_height
    )

    # ------------------------------------------------------------------
    # 9. Tactical view
    # ------------------------------------------------------------------
    tactical_view_converter = TacticalViewConverter(
        court_image_path="./images/basketball_court.png"
    )

    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(
        court_keypoints_per_frame
    )
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(
        court_keypoints_per_frame, player_tracks
    )

    # ==================================================================
    # 11. Drawing
    # ==================================================================

    # --- initialise drawers ---
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    frame_number_drawer = FrameNumberDrawer()
    pass_and_interceptions_drawer = PassInterceptionDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    shot_drawer = ShotDrawer()

    # --- layer 1: player + ball tracks ---
    output_video_frames = player_tracks_drawer.draw(
        video_frames, player_tracks, player_assignment, ball_aquisition
    )
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    # --- layer 2: court keypoints ---
    output_video_frames = court_keypoint_drawer.draw(
        output_video_frames, court_keypoints_per_frame
    )

    # --- layer 3: frame number ---
    output_video_frames = frame_number_drawer.draw(output_video_frames)

    # --- layer 4: ball control ---
    output_video_frames = team_ball_control_drawer.draw(
        output_video_frames, player_assignment, ball_aquisition
    )

    # --- layer 5: passes & interceptions ---
    output_video_frames = pass_and_interceptions_drawer.draw(
        output_video_frames, passes, interceptions
    )

    # --- layer 6: shot stats  ★ NEW ---
    output_video_frames = shot_drawer.draw(
        output_video_frames, shot_frames, shot_results
    )

    # --- layer 8: tactical mini-map ---
    output_video_frames = tactical_view_drawer.draw(
        output_video_frames,
        tactical_view_converter.court_image_path,
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.key_points,
        tactical_player_positions,
        player_assignment,
        ball_aquisition,
    )

    # ------------------------------------------------------------------
    # 12. Save output
    # ------------------------------------------------------------------
    save_video(output_video_frames, args.output_video)


if __name__ == "__main__":
    main()

    