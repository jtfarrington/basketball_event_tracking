"""
Player detection and multi-object tracking using YOLO + ByteTrack.

The :class:`PlayerTracker` combines YOLO object-detection with Supervision's
ByteTrack algorithm so that every detected player receives a stable track ID
that persists across frames.
"""

from ultralytics import YOLO
import supervision as sv

from utils import read_stub, save_stub


class PlayerTracker:
    """Detect and track basketball players across video frames.

    Attributes
    ----------
    model : YOLO
        The YOLO detection model.
    tracker : sv.ByteTrack
        The ByteTrack multi-object tracker.
    """

    def __init__(self, model_path):
        """
        Initialize the PlayerTracker with YOLO model and ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_frames(self, frames):
        """
        Detect players in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i : i + batch_size], conf=0.5)
            detections += detections_batch
        return detections

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get player tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
        """
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None and len(tracks) == len(frames):
            return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to Supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Update tracker with current-frame detections
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks.append({})
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["Player"]:
                    tracks[frame_num][track_id] = {"bbox": bbox}

        save_stub(stub_path, tracks)
        return tracks
