"""
Centralised configuration constants for the basketball analysis pipeline.

All default file paths for models, stubs, and outputs are defined here so that
every module can import them from a single location.
"""

STUBS_DEFAULT_PATH = "stubs"
PLAYER_DETECTOR_PATH = "models/player_detector.pt"
BALL_DETECTOR_PATH = "models/ball_detector_model.pt"
COURT_KEYPOINT_DETECTOR_PATH = "models/court_keypoint_detector.pt"
OUTPUT_VIDEO_PATH = "output_videos/output_video.avi"