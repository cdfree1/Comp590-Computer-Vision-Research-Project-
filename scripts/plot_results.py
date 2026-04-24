"""Read outputs/results/results.csv and generate simple charts:
    1. file size vs compression level
    2. total detections vs compression level
    3. average confidence vs compression level
    4. average box IoU vs compression level
    5. box recall at IoU >= 0.50 vs compression level
    6. violin plots of original-box IoU accuracy distributions

Usage:
    python -m scripts.plot_results
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "outputs" / "results" / "results.csv"
IOU_SAMPLES_CSV = REPO_ROOT / "outputs" / "results" / "box_iou_samples.csv"
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
            raw_val = by_version[v].get(field, "")
            val = _to_float(raw_val) if cast is float else raw_val
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


def _plot_iou_violin(rows: list[dict], out_file: Path) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        iou = _to_float(row.get("iou", ""))
        if iou is not None:
            grouped[row["version"]].append(iou)

    versions = [v for v in VERSION_ORDER if grouped[v]]
    if not versions:
        print("[plot] skipped IoU violin plot; no original-box IoU samples found")
        return

    data = [grouped[v] for v in versions]
    fig, ax = plt.subplots(figsize=(7, 4))
    parts = ax.violinplot(
        data,
        positions=range(1, len(versions) + 1),
        showmeans=True,
        showmedians=True,
        widths=0.8,
    )
    for body in cast(list, parts["bodies"]):
        body.set_facecolor("#7aa6c2")
        body.set_edgecolor("#31465a")
        body.set_alpha(0.75)

    ax.set_xticks(range(1, len(versions) + 1), versions)
    ax.set_xlabel("Compression level")
    ax.set_ylabel("Best same-class IoU per original box")
    ax.set_title("Distribution of box accuracy vs original video")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_file.relative_to(REPO_ROOT)}")


def _plot_iou_violin_by_source(rows: list[dict], out_file: Path) -> None:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        iou = _to_float(row.get("iou", ""))
        if iou is not None:
            grouped[row["source_video"]][row["version"]].append(iou)

    sources = sorted(grouped)
    if not sources:
        print("[plot] skipped by-video IoU violin plot; no original-box IoU samples found")
        return

    cols = min(2, len(sources))
    rows_count = (len(sources) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows_count,
        cols,
        figsize=(7 * cols, 3.8 * rows_count),
        sharey=True,
        squeeze=False,
    )

    for ax, source in zip(axes.flat, sources):
        by_version = grouped[source]
        versions = [v for v in VERSION_ORDER if by_version[v]]
        data = [by_version[v] for v in versions]
        parts = ax.violinplot(
            data,
            positions=range(1, len(versions) + 1),
            showmeans=True,
            showmedians=True,
            widths=0.8,
        )
        for body in cast(list, parts["bodies"]):
            body.set_facecolor("#92b86f")
            body.set_edgecolor("#3f5d35")
            body.set_alpha(0.75)
        ax.set_xticks(range(1, len(versions) + 1), versions)
        ax.set_title(source)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    for ax in axes.flat[len(sources):]:
        ax.axis("off")

    fig.supylabel("Best same-class IoU per original box")
    fig.supxlabel("Compression level")
    fig.suptitle("Box accuracy distributions by source video", y=0.995)
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
        title="Total YOLO26l detections vs compression level",
        out_file=PLOTS_DIR / "total_detections.png",
    )
    _plot_metric(
        grouped, "avg_confidence",
        ylabel="Average confidence",
        title="Average detection confidence vs compression level",
        out_file=PLOTS_DIR / "avg_confidence.png",
    )
    _plot_metric(
        grouped, "avg_box_iou",
        ylabel="Average matched-box IoU",
        title="Box agreement with original video vs compression level",
        out_file=PLOTS_DIR / "avg_box_iou.png",
    )
    _plot_metric(
        grouped, "avg_box_accuracy_iou",
        ylabel="Mean best IoU per original box",
        title="Box accuracy vs original video",
        out_file=PLOTS_DIR / "avg_box_accuracy_iou.png",
    )
    _plot_metric(
        grouped, "box_recall_iou50",
        ylabel="Recall at IoU >= 0.50",
        title="Original-video box recovery vs compression level",
        out_file=PLOTS_DIR / "box_recall_iou50.png",
    )
    if IOU_SAMPLES_CSV.exists():
        iou_rows = load_rows(IOU_SAMPLES_CSV)
        _plot_iou_violin(iou_rows, out_file=PLOTS_DIR / "box_iou_violin.png")
        _plot_iou_violin_by_source(
            iou_rows,
            out_file=PLOTS_DIR / "box_iou_violin_by_video.png",
        )
    else:
        print(
            f"IoU samples CSV not found: {IOU_SAMPLES_CSV}. "
            "Run `python -m scripts.run_experiment` first."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
