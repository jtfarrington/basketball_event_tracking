# Basketball Event Tracking

A modular computer-vision pipeline for analyzing basketball game footage. The system
detects players and the ball, assigns teams by jersey color, tracks possession, computes
tactical views via homography, measures player speed/distance, detects passes,
interceptions, and shot attempts, then renders all annotations back onto the video.

---

## Pipeline Overview (main.py)

1. **Read video** → list of BGR frames.
2. **Detect & track players** (YOLO + ByteTrack) → per-frame bounding boxes with stable IDs.
3. **Detect & track ball** (YOLO) → single best detection per frame, filtered and interpolated.
4. **Detect court keypoints** (YOLO-pose) → 18 reference points on the court per frame.
5. **Assign teams** (CLIP fashion model) → each player ID mapped to Team 1 or Team 2.
6. **Detect ball possession** → frame-level player ID with the ball (or −1).
7. **Detect passes & interceptions** → frame-level event tags.
8. **Detect shot attempts** → frame-level shot events (attempt / make / miss).
9. **Build tactical view** → validate keypoints, compute homography, project players onto
   a top-down court diagram.
10. **Compute speed & distance** → per-player metres travelled and km/h.
11. **Draw all overlays** → annotations rendered onto each frame in layer order.
12. **Save output video**.

## Dependencies

- `ultralytics` — YOLO inference
- `supervision` — detection format conversion + ByteTrack
- `transformers` — CLIP model for team assignment
- `opencv-python` — video I/O and drawing
- `numpy`, `pandas`, `Pillow`

---

## Usage

```bash
python main.py input_video.mp4 \
    --output_video output_videos/output.avi \
    --stub_path stubs/
```

## Acknowledgments

This project was built following the excellent tutorial by codeinajiffy Abdullah Tarek.(https://www.youtube.com/@codeinajiffy) The core tracking and team assignment functionality is based on their work.