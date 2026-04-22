"""Read outputs/results/results.csv and generate three simple charts:
    1. file size vs compression level
    2. total detections vs compression level
    3. average confidence vs compression level

Usage:
    python -m scripts.plot_results
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "outputs" / "results" / "results.csv"
PLOTS_DIR = REPO_ROOT / "outputs" / "results" / "plots"

# X-axis order (low quality -> high quality). "original" is the rightmost baseline.
VERSION_ORDER = ["low", "medium", "high", "original"]


def _to_float(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def group_by_source(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """source_video -> version -> row"""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        out[r["source_video"]][r["version"]] = r
    return out


def _plot_metric(
    grouped: dict[str, dict[str, dict]],
    field: str,
    ylabel: str,
    title: str,
    out_file: Path,
    cast=float,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for source, by_version in grouped.items():
        xs, ys = [], []
        for v in VERSION_ORDER:
            if v not in by_version:
                continue
            val = _to_float(by_version[v][field]) if cast is float else by_version[v][field]
            if val is None:
                continue
            xs.append(v)
            ys.append(val)
        ax.plot(xs, ys, marker="o", label=source)
    ax.set_xlabel("Compression level")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    if len(grouped) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_file.relative_to(REPO_ROOT)}")


def main() -> int:
    if not RESULTS_CSV.exists():
        print(f"CSV not found: {RESULTS_CSV}. Run `python -m scripts.run_experiment` first.")
        return 1

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(RESULTS_CSV)
    grouped = group_by_source(rows)

    _plot_metric(
        grouped, "file_size_mb",
        ylabel="File size (MB)",
        title="File size vs compression level",
        out_file=PLOTS_DIR / "file_size.png",
    )
    _plot_metric(
        grouped, "total_detections",
        ylabel="Total detections",
        title="Total YOLOv8n detections vs compression level",
        out_file=PLOTS_DIR / "total_detections.png",
    )
    _plot_metric(
        grouped, "avg_confidence",
        ylabel="Average confidence",
        title="Average detection confidence vs compression level",
        out_file=PLOTS_DIR / "avg_confidence.png",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
