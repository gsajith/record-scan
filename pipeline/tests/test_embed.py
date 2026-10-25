import numpy as np
from PIL import Image

from pipeline.retrieval.embed import embed_prefix, pending_prefixes

RED = "00000000-0000-4000-8000-000000000001"
GREEN = "00000000-0000-4000-8000-000000000002"
CORRUPT = "00000000-0000-4000-8000-000000000003"
OTHER = "01000000-0000-4000-8000-000000000004"


class MeanColorEmbedder:
    """3-d vectors from an image's mean color, so each cover's row is predictable."""

    name = "fake"
    dim = 3
    input_px = 8

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

    assert pending_prefixes(out) == ["00", "01"]
    assert embed_prefix("00", embedder, out, batch=1, workers=2) == 2

    shard = np.load(out / "00.npz")
    assert shard["uuids"].dtype == np.dtype("U36")
    assert shard["embs"].dtype == np.float16
    rows = dict(zip(shard["uuids"], shard["embs"]))
    assert set(rows) == {RED, GREEN}
    np.testing.assert_allclose(rows[RED], [1, 0, 0], atol=0.02)
    np.testing.assert_allclose(rows[GREEN], [0, 1, 0], atol=0.02)
    assert CORRUPT in (out / "errors.log").read_text()
    assert not (out / "00.tmp.npz").exists()

    # The finished prefix is skipped on rerun; only 01 is left
    calls = embedder.calls
    assert pending_prefixes(out) == ["01"]
    assert embed_prefix("00", embedder, out, batch=1, workers=2) == 0
    assert embedder.calls == calls
