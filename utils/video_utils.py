"""
Video I/O helpers wrapping OpenCV's :pyclass:`cv2.VideoCapture` and
:pyclass:`cv2.VideoWriter`.
"""

import cv2


def read_video(video_path):
    """Read every frame from a video file into a list.

    Parameters
    ----------
    video_path : str
        Path to the input video file.

    Returns
    -------
    list[numpy.ndarray]
        BGR frames in the order they appear in the video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path):
    """Write a list of frames to an AVI video file.

    The codec is set to XVID and the frame-rate defaults to 24 fps.

    Parameters
    ----------
    output_video_frames : list[numpy.ndarray]
        BGR frames to write.
    output_video_path : str
        Destination file path.
    """
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    height, width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
