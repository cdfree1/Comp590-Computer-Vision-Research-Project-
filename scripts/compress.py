"""Compress videos with FFmpeg / libx264 at three CRF levels.

Usage (as a module):
    from scripts.compress import compress_all, CRF_LEVELS

Usage (as a script):
    python -m scripts.compress            # compress every video in data/input_videos
    python -m scripts.compress path/to/video.mp4
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "data" / "input_videos"
COMPRESSED_DIR = REPO_ROOT / "data" / "compressed"

# CRF: lower = higher quality / bigger file.
# 18 is visually (near-)lossless, 28 is ffmpeg's default, 38 is visibly degraded.
CRF_LEVELS: dict[str, int] = {
    "high": 18,
    "medium": 28,
    "low": 38,
}

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg binary not found on PATH. Install it (e.g. `brew install ffmpeg`)."
        )


def compress_video(input_path: Path, level: str, crf: int, out_dir: Path) -> Path:
    """Compress one input video at the given CRF using libx264.

    Returns the path to the compressed file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compressed_{level}.mp4"

    # -preset medium keeps encode time reasonable while honoring CRF.
    # -an drops audio so file-size numbers reflect video only.
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(crf),
        "-an",
        str(out_path),
    ]
    print(f"[compress] {input_path.name} -> {out_path.relative_to(REPO_ROOT)} (crf={crf})")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def compress_all(input_path: Path) -> dict[str, Path]:
    """Produce high/medium/low H.264 versions of one input video.

    Outputs land in `data/compressed/<video_stem>/compressed_<level>.mp4`.
    """
    _ensure_ffmpeg()
    out_dir = COMPRESSED_DIR / input_path.stem
    return {
        level: compress_video(input_path, level, crf, out_dir)
        for level, crf in CRF_LEVELS.items()
    }


def discover_input_videos() -> list[Path]:
    if not INPUT_DIR.exists():
        return []
    return sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def main(argv: list[str]) -> int:
    _ensure_ffmpeg()
    if len(argv) > 1:
        targets = [Path(a) for a in argv[1:]]
    else:
        targets = discover_input_videos()

    if not targets:
        print(f"No videos found. Put .mp4 files in {INPUT_DIR}/.")
        return 1

    for video in targets:
        compress_all(video)
    print("[compress] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
