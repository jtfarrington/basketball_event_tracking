# Basketball Event Tracking

A modular computer-vision pipeline for analyzing basketball game footage. The system detects players and the ball, assigns teams by jersey color, tracks possession, computes tactical views via homography, measures player speed/distance, detects passes, interceptions, and shot attempts, then renders all annotations back onto the video.

![](https://github.com/jtfarrington/basketball_event_tracking/blob/main/my_gif.gif)

## Project Structure
```
basketball_analysis/
├── main.py                             # CLI entry-point; orchestrates the full pipeline
├── configs.py                          # Centralised path constants and default settings
├── utils/                              # Shared helpers (bbox, video I/O, caching)
├── trackers/                           # YOLO + ByteTrack player/ball detection & tracking
├── team_assigner/                      # CLIP-based jersey-colour team classification
├── court_keypoint_detector/            # YOLO-pose court keypoint detection
├── ball_aquisition/                    # Ball possession heuristics
├── pass_and_interception_detector/     # Pass & interception event detection
├── shot_detector/                      # Shot attempt detection (made/missed)
├── tactical_view_converter/            # Homography-based court mapping
├── speed_and_distance_calculator/      # Player kinematics (distance + speed)
└── drawers/                            # All visualisation / annotation layers
```

## Pipeline Overview

1. **Read video** → list of BGR frames
2. **Detect & track players** (YOLO + ByteTrack) → per-frame bounding boxes with stable IDs
3. **Detect & track ball** (YOLO) → single best detection per frame, filtered and interpolated
4. **Detect court keypoints** (YOLO-pose) → 18 reference points on the court per frame
5. **Assign teams** (CLIP fashion model) → each player ID mapped to Team 1 or Team 2
6. **Detect ball possession** → frame-level player ID with the ball (or −1)
7. **Detect passes & interceptions** → frame-level event tags
8. **Detect shot attempts** → frame-level shot events (attempt / make / miss)
9. **Build tactical view** → validate keypoints, compute homography, project players onto a top-down court diagram
10. **Compute speed & distance** → per-player metres travelled and km/h
11. **Draw all overlays** → annotations rendered onto each frame in layer order
12. **Save output video**

## Setup
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Model Weights

Place your trained YOLO model weights in the `models/` directory:
- `models/player_detector.pt`
- `models/ball_detector_model.pt`
- `models/court_keypoint_detector.pt`

## Usage
```bash
python main.py input_video.mp4 \
    --output_video output_videos/output.avi \
    --stub_path stubs/
```

## Dependencies

- `ultralytics` — YOLO inference
- `supervision` — detection format conversion + ByteTrack
- `transformers` — CLIP model for team assignment
- `opencv-python` — video I/O and drawing
- `numpy`, `pandas`, `Pillow`

## Acknowledgments

This project was built following the excellent tutorial by [Abdullah Tarek (codeinajiffy)](https://www.youtube.com/@codeinajiffy). The core tracking and team assignment functionality is based on their work.