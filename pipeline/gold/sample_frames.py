"""Turn record videos dropped into a set's inbox into still frames ready to label.

    python -m pipeline.gold.sample_frames gold|wild|PATH [--max-frames 8] [--fps 2]

Without --fps, each video gives --max-frames frames spread evenly across the whole clip, so a
held-up clip's frames cover both the held and the laid-flat halves. With --fps (flipping videos),
frames come at that rate from the start, up to --max-frames.
"""

import argparse
import glob
import shutil
import subprocess
from pathlib import Path

from pipeline import config

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}


def sample_frames(video: Path, out_dir: Path, fps: float | None, max_frames: int) -> list[Path]:
    """Write up to max_frames JPEGs as <video-stem>-NN.jpg: at fps frames per second from the
    start, or with fps None, one from the middle of each of max_frames equal stretches of the clip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if fps is not None:
        # ffmpeg reads "%" in the output name as part of the sequence pattern, so double it
        pattern = out_dir / f"{video.stem.replace('%', '%%')}-%02d.jpg"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf", f"fps={fps}",
             "-frames:v", str(max_frames), "-q:v", "2", str(pattern)],
            check=True,
        )
    else:
        length = _duration(video)
        for i in range(max_frames):
            at = (i + 0.5) * length / max_frames
            subprocess.run(
                # -update 1 writes one image under its name as given, so "%" needs no escaping
                ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1",
                 "-update", "1", "-q:v", "2", str(out_dir / f"{video.stem}-{i + 1:02d}.jpg")],
                check=True,
            )
    return sorted(out_dir.glob(_frame_glob(video)))


def _duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def _frame_glob(video: Path) -> str:
    return f"{glob.escape(video.stem)}-[0-9][0-9].jpg"


def process_inbox(set_dir: Path, fps: float | None, max_frames: int) -> int:
    """Sample every video in set_dir/inbox into frames beside it, then move it to inbox/done/.

    Phone file names repeat, so a video whose name is already in done/, or whose frames are
    already in the inbox, is skipped rather than overwriting them.
    """
    inbox = set_dir / "inbox"
    done = inbox / "done"
    count = 0
    for video in sorted(p for p in inbox.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES):
        target = done / video.name
        if target.exists() or any(inbox.glob(_frame_glob(video))):
            print(f"Skipped {video.name}: the name is already used. Rename it and run again.")
            continue
        frames = sample_frames(video, inbox, fps, max_frames)
        if not frames:
            print(f"Skipped {video.name}: ffmpeg wrote no frames.")
            continue
        count += len(frames)
        done.mkdir(exist_ok=True)
        shutil.move(video, target)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("set", help='"gold", "wild", or a set folder path')
    parser.add_argument("--fps", type=float, default=None, help="a fixed rate from the start; omit to spread frames")
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
