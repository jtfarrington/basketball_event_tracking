"""Overlay the frame index number on the top-left corner of each frame."""

import cv2


class FrameNumberDrawer:
    def __init__(self):
        pass

    def draw(self, frames):
        """Write the frame number on the top-left corner of each frame.

        Parameters
        ----------
        frames : list[numpy.ndarray]
            Input video frames.

        Returns
        -------
        list[numpy.ndarray]
            Frames with the index rendered at ``(10, 30)``.
        """
        output_frames = []
        for i in range(len(frames)):
            frame = frames[i].copy()
            cv2.putText(
                frame, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            output_frames.append(frame)
        return output_frames
