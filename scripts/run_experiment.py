"""End-to-end experiment: compress every input video, run YOLO26l on each
version (original + high/medium/low), and write results to a single CSV.

Usage:
    python -m scripts.run_experiment
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from scripts.compress import (
    COMPRESSED_DIR,
    CRF_LEVELS,
    compress_all,
    discover_input_videos,
)
from scripts.detect import FrameDetections, compare_to_baseline, run_detection

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "outputs" / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
IOU_SAMPLES_CSV = RESULTS_DIR / "box_iou_samples.csv"
DETECTED_OBJECTS_CSV = RESULTS_DIR / "detected_objects.csv"

CSV_COLUMNS = [
    "source_video",
    "version",
    "codec",
    "crf",
    "file_path",
    "file_size_bytes",
    "file_size_mb",
    "total_detections",
    "avg_confidence",
    "unique_objects",
    "object_counts",
    "baseline_detections",
    "matched_box_count",
    "avg_box_iou",
    "avg_box_accuracy_iou",
    "box_recall_iou50",
    "box_precision_iou50",
    "sample_frame",
]


def file_sizes(path: Path) -> tuple[int, float]:
    size_b = path.stat().st_size
    size_mb = round(size_b / (1024 * 1024), 3)
    return size_b, size_mb


def format_unique_objects(objects: list[str]) -> str:
    return "; ".join(objects)


def format_object_counts(counts: dict[str, int]) -> str:
    return "; ".join(
        f"{name}: {count}"
        for name, count in counts.items()
    )


def evaluate_version(
    source_video: Path,
    version: str,
    codec: str,
    crf: str | int,
    video_path: Path,
    baseline_frames: FrameDetections | None = None,
) -> dict:
    """Run detection on one (video, version) pair and return a CSV row dict."""
    size_b, size_mb = file_sizes(video_path)
    metrics = run_detection(
        video_path=video_path,
        source_stem=source_video.stem,
        version=version,
    )
    if baseline_frames is None:
        box_metrics = compare_to_baseline(
            metrics["frame_detections"],
            metrics["frame_detections"],
        )
    else:
        box_metrics = compare_to_baseline(baseline_frames, metrics["frame_detections"])

    return {
        "source_video": source_video.name,
        "version": version,
        "codec": codec,
        "crf": crf,
        "file_path": str(video_path.relative_to(REPO_ROOT)),
        "file_size_bytes": size_b,
        "file_size_mb": size_mb,
        "total_detections": metrics["total_detections"],
        "avg_confidence": (
            round(metrics["avg_confidence"], 4)
            if metrics["avg_confidence"] is not None
            else ""
        ),
        "unique_objects": format_unique_objects(metrics["unique_objects"]),
        "object_counts": format_object_counts(metrics["object_counts"]),
        "baseline_detections": box_metrics["baseline_detections"],
        "matched_box_count": box_metrics["matched_box_count"],
        "avg_box_iou": (
            round(box_metrics["avg_box_iou"], 4)
            if box_metrics["avg_box_iou"] is not None
            else ""
        ),
        "avg_box_accuracy_iou": (
            round(box_metrics["avg_box_accuracy_iou"], 4)
            if box_metrics["avg_box_accuracy_iou"] is not None
            else ""
        ),
        "box_recall_iou50": (
            round(box_metrics["box_recall_iou50"], 4)
            if box_metrics["box_recall_iou50"] is not None
            else ""
        ),
        "box_precision_iou50": (
            round(box_metrics["box_precision_iou50"], 4)
            if box_metrics["box_precision_iou50"] is not None
            else ""
        ),
        "sample_frame": metrics["sample_frame"],
        "_frame_detections": metrics["frame_detections"],
        "_matched_box_ious": box_metrics["matched_box_ious"],
        "_baseline_box_samples": box_metrics["baseline_box_samples"],
    }


def process_video(source_video: Path) -> list[dict]:
    """Compress a single source video and evaluate all 4 versions."""
    print(f"\n=== {source_video.name} ===")

    # Reuse already-compressed files if present, otherwise (re)generate them.
    out_dir = COMPRESSED_DIR / source_video.stem
    expected = {lvl: out_dir / f"compressed_{lvl}.mp4" for lvl in CRF_LEVELS}
    if not all(p.exists() for p in expected.values()):
        compress_all(source_video)

    original_row = evaluate_version(
        source_video,
        version="original",
        codec="original",
        crf="",
        video_path=source_video,
    )
    baseline_frames = original_row["_frame_detections"]
    rows = [original_row]
    for level, crf in CRF_LEVELS.items():
        rows.append(
            evaluate_version(
                source_video,
                version=level,
                codec="h264",
                crf=crf,
                video_path=expected[level],
                baseline_frames=baseline_frames,
            )
        )
    return rows


def strip_internal_columns(rows: list[dict]) -> list[dict]:
    """Remove in-memory data that should not be written to the summary CSV."""
    return [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in rows
    ]


def write_csv(rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return RESULTS_CSV


def write_iou_samples_csv(rows: list[dict]) -> Path:
    """Write one original-video box accuracy sample per row for violin plots."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "source_video",
        "version",
        "frame_index",
        "baseline_box_index",
        "class_id",
        "iou",
        "matched_iou50",
    ]
    with IOU_SAMPLES_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            for sample in row.get("_baseline_box_samples", []):
                writer.writerow(
                    {
                        "source_video": row["source_video"],
                        "version": row["version"],
                        "frame_index": sample["frame_index"],
                        "baseline_box_index": sample["baseline_box_index"],
                        "class_id": sample["class_id"],
                        "iou": round(sample["iou"], 6),
                        "matched_iou50": int(sample["matched"]),
                    }
                )
    return IOU_SAMPLES_CSV


def write_detected_objects_csv(rows: list[dict]) -> Path:
    """Write a compact object inventory for every evaluated video version."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "source_video",
        "version",
        "unique_objects",
        "object_counts",
        "total_detections",
    ]
    with DETECTED_OBJECTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})
    return DETECTED_OBJECTS_CSV


def main() -> int:
    videos = discover_input_videos()
    if not videos:
        print("No input videos found in data/input_videos/. Add .mp4 files first.")
        return 1

    all_rows: list[dict] = []
    for video in videos:
        all_rows.extend(process_video(video))

    csv_rows = strip_internal_columns(all_rows)
    iou_samples_path = write_iou_samples_csv(all_rows)
    objects_path = write_detected_objects_csv(csv_rows)
    csv_path = write_csv(csv_rows)
    print(f"\n[run_experiment] wrote {len(all_rows)} rows -> {csv_path.relative_to(REPO_ROOT)}")
    print(f"[run_experiment] wrote IoU samples -> {iou_samples_path.relative_to(REPO_ROOT)}")
    print(f"[run_experiment] wrote detected objects -> {objects_path.relative_to(REPO_ROOT)}")

    # Tiny human-readable summary so you can sanity-check without opening the CSV.
    print("\nsummary:")
    for r in csv_rows:
        print(
            f"  {r['source_video']:<20s} {r['version']:<8s} "
            f"size={r['file_size_mb']:>7.3f} MB  "
            f"dets={r['total_detections']:<6d} "
            f"objects={r['unique_objects']}  "
            f"avg_conf={r['avg_confidence']}  "
            f"box_acc_iou={r['avg_box_accuracy_iou']}  "
            f"recall@.50={r['box_recall_iou50']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
