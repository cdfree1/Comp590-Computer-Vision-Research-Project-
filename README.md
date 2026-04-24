# Video Compression vs Object Detection Accuracy

Minimal class-paper experiment: how does H.264 compression level affect YOLO26l
object detection on short videos?

- **Detector:** YOLO26l (`yolo26l.pt`)
- **Codec:** H.264 (libx264)
- **Compression levels:** CRF 18 (high), 28 (medium), 38 (low), plus the uncompressed original as a baseline
- **Metrics per video version:** file size, total detections, average confidence,
  and box agreement against YOLO detections on the original video

## Repo layout

```
data/
  input_videos/         # put your 3 short source videos here (.mp4)
  compressed/<stem>/    # compressed_{high,medium,low}.mp4 (auto-generated)
outputs/
  annotated/<stem>/     # one annotated sample frame per version
  results/
    results.csv         # all metrics, one row per (video, version)
    box_iou_samples.csv # one original-box IoU sample per row for violin plots
    plots/*.png         # summary charts, including aggregate and by-video violins
scripts/
  compress.py           # FFmpeg libx264 compression at CRF 18/28/38
  detect.py             # YOLO26l inference + metric aggregation
  run_experiment.py     # orchestrator (compress -> detect -> CSV)
  plot_results.py       # CSV -> summary plots
experiment.ipynb        # notebook walkthrough of the whole pipeline
main.py                 # smoke test: verify YOLO26l loads
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

Two equivalent ways to run the full pipeline:

### Option A — Notebook (recommended for the paper/demo)

Open `experiment.ipynb` and run the cells top-to-bottom. It walks through
setup → compression → detection → CSV → plots → annotated samples, reusing
the same functions as the CLI scripts.

### Option B — CLI scripts

1. **Drop 3 short videos** into `data/input_videos/` (any `.mp4`/`.mov`/etc.).
   A sample `d.mp4` is already included.
2. **Compress + detect + write CSV:**
   ```bash
   python -m scripts.run_experiment
   ```
   This produces `data/compressed/<stem>/compressed_{high,medium,low}.mp4`,
   one annotated sample frame per version under `outputs/annotated/<stem>/`,
   `outputs/results/results.csv`, and `outputs/results/box_iou_samples.csv`.
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
| `baseline_detections` | YOLO boxes found in the original video baseline |
| `matched_box_count` | same-class boxes matched at IoU >= 0.50 against the original |
| `avg_box_iou` | mean IoU of same-class overlapping boxes against the original |
| `avg_box_accuracy_iou` | mean best same-class IoU per original box; missed boxes count as 0 |
| `box_recall_iou50` | fraction of original boxes recovered at IoU >= 0.50 |
| `box_precision_iou50` | fraction of this version's boxes matching original boxes at IoU >= 0.50 |
| `sample_frame`    | path to an annotated sample frame, if any  |

`outputs/results/box_iou_samples.csv` stores the individual original-box IoU
values used for the violin plot. Each original-video YOLO box contributes one
row per version; if the compressed video misses that object, `iou` is `0`.

## Notes for the paper

- CRF is a quality knob, not a bitrate; lower = better quality / larger file.
- CRF 18 ≈ visually lossless, 28 is ffmpeg's default, 38 is visibly degraded.
- `preset=medium` keeps encode time reasonable; audio is stripped (`-an`) so
  file sizes reflect the video stream only.
- No human ground-truth labels are used, so we do not compute mAP. For box
  accuracy, YOLO detections on the original video are treated as the baseline;
  compressed-video boxes are matched frame-by-frame by class and IoU.
