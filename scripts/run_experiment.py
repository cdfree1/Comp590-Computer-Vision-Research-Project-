"""End-to-end experiment: compress every input video, run YOLOv8n on each
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
from scripts.detect import run_detection

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "outputs" / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"

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
    "sample_frame",
]


def file_sizes(path: Path) -> tuple[int, float]:
    size_b = path.stat().st_size
    size_mb = round(size_b / (1024 * 1024), 3)
    return size_b, size_mb


def evaluate_version(
    source_video: Path,
    version: str,
    codec: str,
    crf: str | int,
    video_path: Path,
) -> dict:
    """Run detection on one (video, version) pair and return a CSV row dict."""
    size_b, size_mb = file_sizes(video_path)
    metrics = run_detection(
        video_path=video_path,
        source_stem=source_video.stem,
        version=version,
    )
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
        "sample_frame": metrics["sample_frame"],
    }


def process_video(source_video: Path) -> list[dict]:
    """Compress a single source video and evaluate all 4 versions."""
    print(f"\n=== {source_video.name} ===")

    # Reuse already-compressed files if present, otherwise (re)generate them.
    out_dir = COMPRESSED_DIR / source_video.stem
    expected = {lvl: out_dir / f"compressed_{lvl}.mp4" for lvl in CRF_LEVELS}
    if not all(p.exists() for p in expected.values()):
        compress_all(source_video)

    rows = [
        evaluate_version(
            source_video,
            version="original",
            codec="original",
            crf="",
            video_path=source_video,
        )
    ]
    for level, crf in CRF_LEVELS.items():
        rows.append(
            evaluate_version(
                source_video,
                version=level,
                codec="h264",
                crf=crf,
                video_path=expected[level],
            )
        )
    return rows


def write_csv(rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return RESULTS_CSV


def main() -> int:
    videos = discover_input_videos()
    if not videos:
        print("No input videos found in data/input_videos/. Add .mp4 files first.")
        return 1

    all_rows: list[dict] = []
    for video in videos:
        all_rows.extend(process_video(video))

    csv_path = write_csv(all_rows)
    print(f"\n[run_experiment] wrote {len(all_rows)} rows -> {csv_path.relative_to(REPO_ROOT)}")

    # Tiny human-readable summary so you can sanity-check without opening the CSV.
    print("\nsummary:")
    for r in all_rows:
        print(
            f"  {r['source_video']:<20s} {r['version']:<8s} "
            f"size={r['file_size_mb']:>7.3f} MB  "
            f"dets={r['total_detections']:<6d} "
            f"avg_conf={r['avg_confidence']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
