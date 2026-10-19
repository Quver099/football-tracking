# ⚽ Football Player & Ball Tracking

> Work in progress (milestones 1–5 of 8 done). Demo GIF and full write-up come with Milestone 8.

From a broadcast match clip, produce:
1. an annotated video (players / goalkeepers / referees / ball, persistent IDs, team colours),
2. a bird's-eye 2D pitch view,
3. per-player stats (distance, speed, heatmap) and team possession.

## Stack
YOLO11 (ultralytics) · ByteTrack (supervision) · OpenCV · scikit-learn (KMeans) · pandas · homography · Streamlit · Docker

## Pipeline so far
| Step | What it does | Code |
|---|---|---|
| Detect | YOLO11s fine-tuned on football frames (ball, goalkeeper, player, referee), imgsz 1280 | `detection.py`, notebooks 01–02 |
| Track | ByteTrack IDs, confidence-weighted class vote, ghost-track filter, ID-switch split | `tracking.py`, notebook 03 |
| Teams | shirt colour (per-player background removal, Lab median) + KMeans, vote per track, goalkeepers by position | `teams.py`, notebook 04 |
| Ball | rolling-median outlier removal, gap interpolation, possession, passes and turnovers | `ball.py`, notebook 05 |

**Sample-clip results (30 s):** ball known in 89% of frames after cleaning (71% detected); possession 61% / 39%, stable within about 4 points across reasonable settings.

## Repo layout
```
data/raw/        input clips (git-ignored)
models/          trained weights (git-ignored)
outputs/         generated videos / tables (git-ignored)
notebooks/       one notebook per milestone (01–05)
scripts/         command-line entry points
src/football_tracking/   reusable modules
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
gdown 1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK- -O data/raw/08fd33_0.mp4                    # sample clip
python scripts/run_tracking.py --video data/raw/08fd33_0.mp4 --weights models/football_yolo11s_1280.pt --pitch-filter
python scripts/run_teams.py --video data/raw/08fd33_0.mp4 --tracks outputs/m3_tracks.csv --names "Green,White"
```
The fine-tuned weights are not in the repo; train them with notebook 02 or `scripts/train.py`.

## Progress
- [x] M1 Setup + pretrained YOLO baseline
- [x] M2 Fine-tuned detector (player / goalkeeper / referee / ball)
- [x] M3 Tracking (ByteTrack) + class vote + ghost filter
- [x] M4 Team assignment (shirt colour KMeans) + ID-switch repair
- [x] M5 Ball cleaning, interpolation + possession
- [ ] M6 Pitch keypoints + homography (metres, 2D view)
- [ ] M7 Speed, distance, heatmaps
- [ ] M8 Streamlit demo + Docker

## Data and credits
- Sample clip: from the DFL Bundesliga Data Shootout (Kaggle), via the download used in Roboflow's `sports` examples. Video is not redistributed here.
- Training data: Roboflow Universe "football-players-detection" dataset (CC BY 4.0).
