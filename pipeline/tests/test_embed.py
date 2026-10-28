import shutil

import numpy as np
import pytest
from PIL import Image

from pipeline.retrieval.build_index import stale_shards
from pipeline.retrieval.embed import embed_prefix, is_done

RED = "00000000-0000-4000-8000-000000000001"
GREEN = "00000000-0000-4000-8000-000000000002"
CORRUPT = "00000000-0000-4000-8000-000000000003"
OTHER = "01000000-0000-4000-8000-000000000004"
BLUE = "00000000-0000-4000-8000-000000000005"


class MeanColorEmbedder:
    """3-d vectors from an image's mean color, so each cover's row is predictable."""

    name = "fake"
    dim = 3
    input_px = 8
    signature = "fake-8px-test"

    def __init__(self):
        self.calls = 0

    def preprocess(self, img):
        raise NotImplementedError

    def embed(self, images):
        self.calls += 1
        means = np.array([np.asarray(img, dtype=np.float32).mean(axis=(0, 1)) for img in images])
        return means / np.linalg.norm(means, axis=1, keepdims=True)


def test_embed_prefix_skips_corrupt_and_resumes(tmp_path, reload_config):
    root = tmp_path / "images"
    (root / "00").mkdir(parents=True)
    (root / "01").mkdir()
    Image.new("RGB", (40, 40), (255, 0, 0)).save(root / "00" / f"{RED}.jpg")
    Image.new("RGB", (40, 40), (0, 255, 0)).save(root / "00" / f"{GREEN}.jpg")
    (root / "00" / f"{CORRUPT}.jpg").write_bytes(b"not a jpeg")
    Image.new("RGB", (40, 40), (0, 0, 255)).save(root / "01" / f"{OTHER}.jpg")
    reload_config(RS_IMAGE_ROOT=root)
    out = tmp_path / "emb"
    out.mkdir()
    (out / "00.tmp.npz").write_bytes(b"left by a run that crashed mid-prefix")
    embedder = MeanColorEmbedder()
    files = [f"00/{k}.jpg" for k in (RED, GREEN, CORRUPT)]

    assert not is_done("00", files, out, embedder.signature)
    assert embed_prefix("00", files, embedder, out, batch=1, workers=2) == 2

    shard = np.load(out / "00.npz")
    assert shard["uuids"].dtype == np.dtype("U36")
    assert shard["embs"].dtype == np.float16
    assert str(shard["signature"]) == "fake-8px-test"
    rows = dict(zip(shard["uuids"], shard["embs"]))
    assert set(rows) == {RED, GREEN}
    np.testing.assert_allclose(rows[RED], [1, 0, 0], atol=0.02)
    np.testing.assert_allclose(rows[GREEN], [0, 1, 0], atol=0.02)
    assert CORRUPT in (out / "00.errors.log").read_text()
    assert not (out / "00.tmp.npz").exists()

    # The finished prefix is skipped on rerun, whatever order the list comes in; 01 is still pending
    calls = embedder.calls
    assert is_done("00", files[::-1], out, embedder.signature)
    assert not is_done("01", [f"01/{OTHER}.jpg"], out, embedder.signature)
    assert embed_prefix("00", files, embedder, out, batch=1, workers=2) == 0
    assert embedder.calls == calls


def test_embed_prefix_redoes_a_prefix_whose_cover_list_grew(tmp_path, reload_config):
    root = tmp_path / "images"
    (root / "00").mkdir(parents=True)
    Image.new("RGB", (40, 40), (255, 0, 0)).save(root / "00" / f"{RED}.jpg")
    reload_config(RS_IMAGE_ROOT=root)
    out = tmp_path / "emb"
    embedder = MeanColorEmbedder()
    assert embed_prefix("00", [f"00/{RED}.jpg"], embedder, out, batch=1, workers=1) == 1

    # A catalog update adds a cover to the done prefix
    Image.new("RGB", (40, 40), (0, 0, 255)).save(root / "00" / f"{BLUE}.jpg")
    files = [f"00/{RED}.jpg", f"00/{BLUE}.jpg"]
    assert not is_done("00", files, out, embedder.signature)
    assert embed_prefix("00", files, embedder, out, batch=1, workers=1) == 2
    assert set(np.load(out / "00.npz")["uuids"]) == {RED, BLUE}
    assert is_done("00", files, out, embedder.signature)


def test_stale_shards_names_prefixes_behind_the_catalog(tmp_path, reload_config):
    root = tmp_path / "images"
    (root / "00").mkdir(parents=True)
    Image.new("RGB", (40, 40), (255, 0, 0)).save(root / "00" / f"{RED}.jpg")
    reload_config(RS_IMAGE_ROOT=root)
    out = tmp_path / "emb"
    covers = {"00": [f"00/{RED}.jpg"]}
    embed_prefix("00", covers["00"], MeanColorEmbedder(), out, batch=1, workers=1)
    assert stale_shards([out / "00.npz"], covers.__getitem__) == []

    covers["00"].append(f"00/{BLUE}.jpg")  # a catalog update the embed run hasn't caught up with
    assert stale_shards([out / "00.npz"], covers.__getitem__) == ["00"]


def test_embed_prefix_refuses_to_finish_when_the_drive_drops(tmp_path, reload_config):
    root = tmp_path / "images"
    (root / "00").mkdir(parents=True)
    for uuid in (RED, GREEN):
        Image.new("RGB", (40, 40), (255, 0, 0)).save(root / "00" / f"{uuid}.jpg")
    reload_config(RS_IMAGE_ROOT=root)
    out = tmp_path / "emb"

    class UnmountingEmbedder(MeanColorEmbedder):
        def embed(self, images):
            shutil.rmtree(root, ignore_errors=True)  # the drive goes away after the first batch
            return super().embed(images)

    with pytest.raises(SystemExit):
        embed_prefix("00", [f"00/{RED}.jpg", f"00/{GREEN}.jpg"], UnmountingEmbedder(), out, batch=1, workers=1)
    assert not (out / "00.done").exists()
    assert not (out / "00.npz").exists()


def test_embed_prefix_redoes_a_prefix_done_by_another_embedder(tmp_path, reload_config):
    root = tmp_path / "images"
    (root / "00").mkdir(parents=True)
    Image.new("RGB", (40, 40), (255, 0, 0)).save(root / "00" / f"{RED}.jpg")
    reload_config(RS_IMAGE_ROOT=root)
    out = tmp_path / "emb"
    files = [f"00/{RED}.jpg"]
    embedder = MeanColorEmbedder()
    assert embed_prefix("00", files, embedder, out, batch=1, workers=1) == 1
    done = out / "00.done"
    done.write_text(done.read_text().replace(embedder.signature, "fake-8px-old"))  # done before a preprocessing change

    assert not is_done("00", files, out, embedder.signature)
    assert embed_prefix("00", files, embedder, out, batch=1, workers=1) == 1
    assert is_done("00", files, out, embedder.signature)
