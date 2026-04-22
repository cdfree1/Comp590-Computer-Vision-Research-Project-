# Video Compression vs Object Detection Accuracy

Minimal class-paper experiment: how does H.264 compression level affect YOLOv8n
object detection on short videos?

- **Detector:** YOLOv8n (`yolov8n.pt`)
- **Codec:** H.264 (libx264)
- **Compression levels:** CRF 18 (high), 28 (medium), 38 (low), plus the uncompressed original as a baseline
- **Metrics per video version:** file size, total detections, average confidence

## Repo layout

```
data/
  input_videos/         # put your 3 short source videos here (.mp4)
  compressed/<stem>/    # compressed_{high,medium,low}.mp4 (auto-generated)
outputs/
  annotated/<stem>/     # one annotated sample frame per version
  results/
    results.csv         # all metrics, one row per (video, version)
    plots/*.png         # 3 summary charts
scripts/
  compress.py           # FFmpeg libx264 compression at CRF 18/28/38
  detect.py             # YOLOv8n inference + metric aggregation
  run_experiment.py     # orchestrator (compress -> detect -> CSV)
  plot_results.py       # CSV -> 3 plots
main.py                 # smoke test: verify YOLOv8n loads
```

## Setup

Requires Python 3.14+ (see `.python-version`) and an FFmpeg binary on `PATH`.

```bash
# install ffmpeg (macOS)
brew install ffmpeg

# install python deps
uv sync            # or: pip install -e .
```

## Running the experiment

1. **Drop 3 short videos** into `data/input_videos/` (any `.mp4`/`.mov`/etc.).
   A sample `d.mp4` is already included.
2. **Compress + detect + write CSV:**
   ```bash
   python -m scripts.run_experiment
   ```
   This produces `data/compressed/<stem>/compressed_{high,medium,low}.mp4`,
   one annotated sample frame per version under `outputs/annotated/<stem>/`,
   and `outputs/results/results.csv`.
3. **Generate plots:**
   ```bash
   python -m scripts.plot_results
   ```
   Plots land in `outputs/results/plots/`.

### Running steps individually

```bash
python -m scripts.compress                        # compress everything in data/input_videos/
python -m scripts.compress data/input_videos/d.mp4  # or a single file
```

## CSV schema

| column            | meaning                                    |
|-------------------|--------------------------------------------|
| `source_video`    | original filename                          |
| `version`         | `original` / `high` / `medium` / `low`     |
| `codec`           | `original` or `h264`                       |
| `crf`             | CRF value (`18`, `28`, `38`) or blank       |
| `file_path`       | path to the video used for detection       |
| `file_size_bytes` | raw bytes                                  |
| `file_size_mb`    | MB, rounded to 3 decimals                  |
| `total_detections`| sum of YOLO boxes across all frames        |
| `avg_confidence`  | mean confidence across all detections       |
| `sample_frame`    | path to an annotated sample frame, if any  |

## Notes for the paper

- CRF is a quality knob, not a bitrate; lower = better quality / larger file.
- CRF 18 ≈ visually lossless, 28 is ffmpeg's default, 38 is visibly degraded.
- `preset=medium` keeps encode time reasonable; audio is stripped (`-an`) so
  file sizes reflect the video stream only.
- No ground-truth labels are used — we do not compute mAP. We report the
  detector's own output counts and confidences as a proxy for quality.
