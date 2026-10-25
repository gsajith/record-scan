import json

import numpy as np
import pytest
from PIL import Image

from pipeline.metadata.build_catalog_db import build
from pipeline.metadata.catalog import Catalog
from pipeline.retrieval.build_index import build_index
from pipeline.retrieval.identify import Identifier

BEATLES = "b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d"
BJORK = "87c5dedd-371d-4a53-9f7f-80522fb7f3cb"
ABBEY = "11111111-1111-4111-8111-111111111111"
ABBEY_REISSUE = "11111111-1111-4111-8111-222222222222"
DEBUT = "22222222-2222-4222-8222-222222222222"
HOMOGENIC = "33333333-3333-4333-8333-333333333333"

E0, E2 = [1, 0, 0, 0], [0, 0, 1, 0]


class FixedEmbedder:
    """Returns the given vectors in order, one per image, so each query view's vector is chosen."""

    name = "fixed"
    dim = 4
    input_px = 8

    def __init__(self, *vectors):
        self.vectors = np.array(vectors, dtype=np.float32)

    def preprocess(self, img):
        raise NotImplementedError

    def embed(self, images):
        return self.vectors[: len(images)]


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def catalog(tmp_path):
    releases = [
        {"title": "Abbey Road", "id": ABBEY, "artist_ids": [BEATLES]},
        {"title": "Abbey Road", "id": ABBEY_REISSUE, "artist_ids": [BEATLES]},
        {"title": "Debut", "id": DEBUT, "artist_ids": [BJORK]},
        {"title": "Homogenic", "id": HOMOGENIC, "artist_ids": [BJORK]},
    ]
    artists = {BEATLES: {"name": "Beatles, The"}, BJORK: {"name": "Björk"}}
    (tmp_path / "releases.json").write_text(json.dumps(releases))
    (tmp_path / "artists.json").write_text(json.dumps(artists))
    build(tmp_path / "releases.json", tmp_path / "artists.json", tmp_path / "catalog.sqlite", delta_dir=None)
    return Catalog(tmp_path / "catalog.sqlite")


@pytest.fixture
def index_dir(tmp_path):
    """Two shards: the reissue sits just off Abbey Road, Debut at cosine 0.6 from it."""
    shard_dir = tmp_path / "emb"
    shard_dir.mkdir()
    shards = {
        "00": {ABBEY: E0, DEBUT: unit([0.6, 0.8, 0, 0])},
        "01": {ABBEY_REISSUE: unit([0.98, 0.2, 0, 0]), HOMOGENIC: E2},
    }
    for prefix, rows in shards.items():
        np.savez(
            shard_dir / f"{prefix}.npz",
            uuids=np.array(list(rows), dtype="U36"),
            embs=np.array([unit(v) for v in rows.values()], dtype=np.float16),
        )
    assert build_index(sorted(shard_dir.glob("*.npz")), 4, shard_dir, M=16, ef_construction=100) == 4
    return shard_dir


def test_build_index_and_identify_roundtrip(catalog, index_dir):
    photo = Image.new("RGB", (64, 64))

    matches = Identifier(FixedEmbedder(E0), index_dir, catalog).identify(photo, k=3, views=1)

    # The reissue is another release of Abbey Road, so it collapses into the first match
    assert [m.mbid for m in matches] == [ABBEY, DEBUT, HOMOGENIC]
    assert matches[0].title == "Abbey Road"
    assert matches[0].artists == ("Beatles, The",)
    np.testing.assert_allclose([m.score for m in matches], [1.0, 0.6, 0.0], atol=0.01)


def test_identify_max_pools_views_and_uses_clusters(catalog, index_dir):
    photo = Image.new("RGB", (64, 64))
    embedder = FixedEmbedder(E2, E0, E2)

    one_view = Identifier(embedder, index_dir, catalog).identify(photo, k=2, views=1)
    three_views = Identifier(embedder, index_dir, catalog).identify(photo, k=2, views=3)
    clusters = {ABBEY: 1, ABBEY_REISSUE: 1, DEBUT: 1, HOMOGENIC: 2}
    clustered = Identifier(embedder, index_dir, catalog, clusters).identify(photo, k=3, views=3)

    assert one_view[0].mbid == HOMOGENIC
    assert one_view[1].score < 0.1  # Abbey Road only matches through the second view
    assert {m.mbid for m in three_views} == {HOMOGENIC, ABBEY}
    np.testing.assert_allclose([m.score for m in three_views], [1.0, 1.0], atol=0.01)
    assert {m.mbid for m in clustered} == {ABBEY, HOMOGENIC}


def test_build_index_rejects_wrong_dim(tmp_path):
    np.savez(tmp_path / "00.npz", uuids=np.array([ABBEY], dtype="U36"), embs=np.zeros((1, 3), np.float16))

    with pytest.raises(SystemExit, match="00.npz"):
        build_index([tmp_path / "00.npz"], 4, tmp_path, M=16, ef_construction=100)


def test_identify_widens_search_when_one_group_fills_the_neighbours(catalog, tmp_path):
    # 25 lookalikes of one cover outrank Debut, more than the first k*10 = 20 neighbours hold
    lookalikes = [f"{i:08x}-0000-4000-8000-000000000000" for i in range(25)]
    rng = np.random.default_rng(0)
    vectors = [unit(np.array(E0) + rng.normal(0, 0.01, 4)) for _ in lookalikes]
    np.savez(
        tmp_path / "00.npz",
        uuids=np.array([*lookalikes, DEBUT], dtype="U36"),
        embs=np.array([*vectors, unit([0.6, 0.8, 0, 0])], dtype=np.float16),
    )
    build_index([tmp_path / "00.npz"], 4, tmp_path, M=16, ef_construction=100)
    clusters = dict.fromkeys(lookalikes, 1)

    matches = Identifier(FixedEmbedder(E0), tmp_path, catalog, clusters).identify(Image.new("RGB", (8, 8)), k=2, views=1)

    assert [m.mbid for m in matches][1:] == [DEBUT]
