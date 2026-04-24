"""Run YOLO26l on a video and collect per-video detection metrics.

Metrics returned:
    - total_detections: sum of boxes across all processed frames
    - avg_confidence:   mean confidence across all detections (None if 0 dets)
    - frame_detections: per-frame boxes/classes/confidences for IoU comparison

Optionally saves one annotated sample frame for the paper.
"""

from __future__ import annotations

from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Optional, TypedDict

import cv2
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_WEIGHTS = "yolo26l.pt"
ANNOTATED_DIR = REPO_ROOT / "outputs" / "annotated"

_model_cache: dict[str, YOLO] = {}


class Detection(TypedDict):
    xyxy: tuple[float, float, float, float]
    cls: int
    conf: float


FrameDetections = list[list[Detection]]


def get_model(weights: Path | str = MODEL_WEIGHTS) -> YOLO:
    """Load YOLO26l once and reuse it across calls."""
    key = str(weights)
    if key not in _model_cache:
        _model_cache[key] = YOLO(str(weights))
    return _model_cache[key]


def _extract_detections(boxes) -> list[Detection]:
    """Convert Ultralytics boxes into lightweight Python data."""
    if boxes is None or len(boxes) == 0:
        return []

    xyxys = boxes.xyxy.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    confs = boxes.conf.cpu().tolist()

    detections: list[Detection] = []
    for xyxy, cls, conf in zip(xyxys, classes, confs):
        x1, y1, x2, y2 = xyxy
        det: Detection = {
            "xyxy": (float(x1), float(y1), float(x2), float(y2)),
            "cls": int(cls),
            "conf": float(conf),
        }
        detections.append(det)

    return detections


def _class_name(model: YOLO, class_id: int) -> str:
    """Return a stable display name for a model class id."""
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def summarize_detected_objects(
    frame_detections: FrameDetections,
    model: YOLO,
) -> tuple[list[str], dict[str, int]]:
    """Return unique object names and detection counts across all frames."""
    class_counts = Counter(
        det["cls"]
        for frame in frame_detections
        for det in frame
    )
    object_counts = {
        _class_name(model, class_id): count
        for class_id, count in sorted(
            class_counts.items(),
            key=lambda item: _class_name(model, item[0]),
        )
    }
    return list(object_counts), object_counts


def box_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Return intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter_area
    return inter_area / union if union > 0 else 0.0


def compare_to_baseline(
    baseline_frames: FrameDetections,
    candidate_frames: FrameDetections,
    iou_threshold: float = 0.5,
) -> dict:
    """Compare candidate boxes against original-video boxes frame by frame.

    Boxes are matched one-to-one within the same frame and class using greedy IoU.
    The original video is treated as the baseline because this project does not
    include human-labeled ground truth.
    """
    baseline_total = sum(len(frame) for frame in baseline_frames)
    candidate_total = sum(len(frame) for frame in candidate_frames)
    pair_ious: list[float] = []
    baseline_box_samples: list[dict] = []
    threshold_matches = 0

    for frame_index, (baseline, candidate) in enumerate(
        zip_longest(baseline_frames, candidate_frames, fillvalue=[])
    ):
        pairs: list[tuple[float, int, int]] = []
        for b_idx, b_det in enumerate(baseline):
            for c_idx, c_det in enumerate(candidate):
                if b_det["cls"] != c_det["cls"]:
                    continue
                iou = box_iou(b_det["xyxy"], c_det["xyxy"])
                if iou > 0:
                    pairs.append((iou, b_idx, c_idx))

        used_baseline: set[int] = set()
        used_candidate: set[int] = set()
        best_iou_by_baseline = [0.0 for _ in baseline]
        for iou, b_idx, c_idx in sorted(pairs, reverse=True):
            if b_idx in used_baseline or c_idx in used_candidate:
                continue
            used_baseline.add(b_idx)
            used_candidate.add(c_idx)
            pair_ious.append(iou)
            best_iou_by_baseline[b_idx] = iou
            if iou >= iou_threshold:
                threshold_matches += 1

        for b_idx, b_det in enumerate(baseline):
            best_iou = best_iou_by_baseline[b_idx]
            baseline_box_samples.append(
                {
                    "frame_index": frame_index,
                    "baseline_box_index": b_idx,
                    "class_id": b_det["cls"],
                    "iou": best_iou,
                    "matched": best_iou >= iou_threshold,
                }
            )

    avg_box_iou = (sum(pair_ious) / len(pair_ious)) if pair_ious else None
    avg_box_accuracy_iou = (
        sum(sample["iou"] for sample in baseline_box_samples) / baseline_total
        if baseline_total
        else None
    )
    recall = (threshold_matches / baseline_total) if baseline_total else None
    precision = (threshold_matches / candidate_total) if candidate_total else None
    return {
        "baseline_detections": baseline_total,
        "matched_box_count": threshold_matches,
        "avg_box_iou": avg_box_iou,
        "avg_box_accuracy_iou": avg_box_accuracy_iou,
        "box_recall_iou50": recall,
        "box_precision_iou50": precision,
        "matched_box_ious": pair_ious,
        "baseline_box_samples": baseline_box_samples,
    }


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
    frame_detections: FrameDetections = []

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
        detections = _extract_detections(boxes)
        frame_detections.append(detections)
        if not detections:
            continue

        total_detections += len(detections)
        conf_sum += sum(d["conf"] for d in detections)

        if save_sample_frame and not sample_saved and sample_path is not None:
            annotated = result.plot()  # BGR numpy image with boxes drawn.
            cv2.imwrite(str(sample_path), annotated)
            sample_saved = True

    avg_confidence = (conf_sum / total_detections) if total_detections > 0 else None
    unique_objects, object_counts = summarize_detected_objects(frame_detections, model)
    return {
        "total_detections": total_detections,
        "avg_confidence": avg_confidence,
        "unique_objects": unique_objects,
        "object_counts": object_counts,
        "sample_frame": str(sample_path) if sample_saved else "",
        "frame_detections": frame_detections,
    }
