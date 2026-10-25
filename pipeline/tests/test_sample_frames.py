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
