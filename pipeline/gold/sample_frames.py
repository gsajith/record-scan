"""Turn record videos dropped into a set's inbox into still frames ready to label.

    python -m pipeline.gold.sample_frames gold|wild|PATH [--fps 1.0] [--max-frames 8]
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from pipeline import config

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}


def sample_frames(video: Path, out_dir: Path, fps: float, max_frames: int) -> list[Path]:
    """Write up to max_frames JPEGs at fps frames per second as <video-stem>-NN.jpg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{video.stem}-%02d.jpg"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf", f"fps={fps}",
         "-frames:v", str(max_frames), "-q:v", "2", str(pattern)],
        check=True,
    )
    return sorted(out_dir.glob(f"{video.stem}-[0-9][0-9].jpg"))


def process_inbox(set_dir: Path, fps: float, max_frames: int) -> int:
    """Sample every video in set_dir/inbox into frames beside it, then move it to inbox/done/."""
    inbox = set_dir / "inbox"
    done = inbox / "done"
    count = 0
    for video in sorted(p for p in inbox.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES):
        count += len(sample_frames(video, inbox, fps, max_frames))
        done.mkdir(exist_ok=True)
        shutil.move(video, done / video.name)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("set", help='"gold", "wild", or a set folder path')
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=8)
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found. Install it with: brew install ffmpeg")
    set_dir = {"gold": config.GOLD_DIR, "wild": config.WILD_DIR}.get(args.set) or Path(args.set)
    if not (set_dir / "inbox").is_dir():
        raise SystemExit(f"No inbox at {set_dir / 'inbox'}. Drop videos there first.")
    print(f"{process_inbox(set_dir, args.fps, args.max_frames)} frames written to {set_dir / 'inbox'}")


if __name__ == "__main__":
    main()
