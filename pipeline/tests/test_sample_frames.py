import shutil
import subprocess

import pytest

from pipeline.gold.sample_frames import process_inbox, sample_frames

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH (brew install ffmpeg)")


def make_video(path, seconds=3):
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x240:rate=10", str(path)],
        check=True,
    )


@needs_ffmpeg
def test_sample_frames_writes_n_frames(tmp_path):
    video = tmp_path / "abbey.mp4"
    make_video(video)
    frames = sample_frames(video, tmp_path / "out", fps=1.0, max_frames=2)
    assert [f.name for f in frames] == ["abbey-01.jpg", "abbey-02.jpg"]
    assert all(f.stat().st_size > 0 for f in frames)


def make_color_video(path, colors, seconds_each=3):
    """One solid color per stretch, so a frame's color says when it was taken."""
    inputs = []
    for c in colors:
        inputs += ["-f", "lavfi", "-i", f"color={c}:size=64x64:duration={seconds_each}:rate=10"]
    concat = "".join(f"[{i}:v]" for i in range(len(colors))) + f"concat=n={len(colors)}:v=1:a=0"
    subprocess.run(["ffmpeg", "-loglevel", "error", *inputs, "-filter_complex", concat, str(path)], check=True)


def color_of(frame):
    from PIL import Image

    r, g, b = Image.open(frame).convert("RGB").resize((1, 1)).getpixel((0, 0))
    names = {(1, 0, 0): "red", (0, 1, 0): "green", (0, 0, 1): "blue", (1, 1, 1): "white"}
    return names.get((r > 128, g > 128, b > 128))


@needs_ffmpeg
def test_sample_frames_spreads_frames_across_the_whole_clip(tmp_path):
    # A held-up clip is held for a while, then laid flat; frames must reach the end
    video = tmp_path / "held.mp4"
    make_color_video(video, ["red", "lime", "blue", "white"])
    frames = sample_frames(video, tmp_path / "out", fps=None, max_frames=4)
    assert [f.name for f in frames] == ["held-01.jpg", "held-02.jpg", "held-03.jpg", "held-04.jpg"]
    assert [color_of(f) for f in frames] == ["red", "green", "blue", "white"]


@needs_ffmpeg
def test_process_inbox_moves_videos_to_done(tmp_path):
    inbox = tmp_path / "gold" / "inbox"
    inbox.mkdir(parents=True)
    make_video(inbox / "one.MOV")
    (inbox / "notes.txt").write_text("not a video")
    assert process_inbox(tmp_path / "gold", fps=1.0, max_frames=8) == 3
    assert (inbox / "done" / "one.MOV").exists()
    assert sorted(p.name for p in inbox.glob("*.jpg")) == ["one-01.jpg", "one-02.jpg", "one-03.jpg"]


@needs_ffmpeg
def test_process_inbox_skips_a_reused_name(tmp_path):
    inbox = tmp_path / "gold" / "inbox"
    (inbox / "done").mkdir(parents=True)
    (inbox / "done" / "IMG_0001.MOV").write_text("earlier video")
    make_video(inbox / "IMG_0001.MOV")
    make_video(inbox / "IMG_0002.mov")
    make_video(inbox / "IMG_0002.mp4")
    assert process_inbox(tmp_path / "gold", fps=1.0, max_frames=8) == 3
    assert (inbox / "done" / "IMG_0001.MOV").read_text() == "earlier video"
    assert (inbox / "IMG_0001.MOV").exists() and (inbox / "IMG_0002.mp4").exists()


@needs_ffmpeg
def test_spread_frames_keep_a_percent_in_the_name(tmp_path):
    video = tmp_path / "50% off.mp4"
    make_video(video)
    frames = sample_frames(video, tmp_path / "out", fps=None, max_frames=2)
    assert [f.name for f in frames] == ["50% off-01.jpg", "50% off-02.jpg"]
