"""Run YOLOv8n on a video and collect simple per-video metrics.

Metrics returned:
    - total_detections: sum of boxes across all processed frames
    - avg_confidence:   mean confidence across all detections (None if 0 dets)

Optionally saves one annotated sample frame (middle frame) for the paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = REPO_ROOT / "yolov8n.pt"
ANNOTATED_DIR = REPO_ROOT / "outputs" / "annotated"

_model_cache: dict[str, YOLO] = {}


def get_model(weights: Path | str = WEIGHTS_PATH) -> YOLO:
    """Load YOLOv8n once and reuse it across calls."""
    key = str(weights)
    if key not in _model_cache:
        _model_cache[key] = YOLO(str(weights))
    return _model_cache[key]


def run_detection(
    video_path: Path,
    source_stem: str,
    version: str,
    save_sample_frame: bool = True,
    conf: float = 0.25,
) -> dict:
    """Run YOLO over every frame of `video_path` and return aggregated metrics."""
    model = get_model()

    total_detections = 0
    conf_sum = 0.0
    sample_saved = False

    # stream=True is memory-friendly for full videos.
    results_iter = model(str(video_path), stream=True, verbose=False, conf=conf)

    # We save the annotated sample the first time we hit a frame with any boxes.
    sample_path: Optional[Path] = None
    if save_sample_frame:
        sample_dir = ANNOTATED_DIR / source_stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / f"{version}_sample.jpg"

    for result in results_iter:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        total_detections += len(boxes)
        conf_sum += float(boxes.conf.sum().item())

        if save_sample_frame and not sample_saved and sample_path is not None:
            annotated = result.plot()  # BGR numpy image with boxes drawn.
            cv2.imwrite(str(sample_path), annotated)
            sample_saved = True

    avg_confidence = (conf_sum / total_detections) if total_detections > 0 else None
    return {
        "total_detections": total_detections,
        "avg_confidence": avg_confidence,
        "sample_frame": str(sample_path) if sample_saved else "",
    }
