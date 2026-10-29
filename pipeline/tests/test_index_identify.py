import json

import numpy as np
import pytest
from PIL import Image

from pipeline.metadata.build_catalog_db import build
from pipeline.metadata.catalog import Catalog
from pipeline.gallery.exclusions import exclusions_hash
from pipeline.retrieval import build_index as build_index_module
from pipeline.retrieval.build_index import build_index, require_current_exclusions
from pipeline.retrieval.identify import Identifier

BEATLES = "b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d"
BJORK = "87c5dedd-371d-4a53-9f7f-80522fb7f3cb"
ABBEY = "11111111-1111-4111-8111-111111111111"
ABBEY_REISSUE = "11111111-1111-4111-8111-222222222222"
DEBUT = "22222222-2222-4222-8222-222222222222"
HOMOGENIC = "33333333-3333-4333-8333-333333333333"
RG_ABBEY = "88888888-8888-4888-8888-888888888880"

E0, E2 = [1, 0, 0, 0], [0, 0, 1, 0]


class FixedEmbedder:
    """Returns the given vectors in order, one per image, so each query view's vector is chosen."""

    name = "fixed"
    dim = 4
    input_px = 8
    signature = "fixed-8px-test"

    def __init__(self, *vectors):
        self.vectors = np.array(vectors, dtype=np.float32)

    def preprocess(self, img):
        raise NotImplementedError

    def embed(self, images):
        return self.vectors[: len(images)]


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def build_catalog(tmp_path, releases, artists):
    """A catalog.sqlite from releases in the slim shape, with no covers."""
    (tmp_path / "releases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in releases))
    (tmp_path / "artists.json").write_text(json.dumps(artists))
    (tmp_path / "covers.jsonl").write_text("")
    build(tmp_path / "releases.jsonl", tmp_path / "artists.json", tmp_path / "covers.jsonl", tmp_path / "catalog.sqlite")
    return Catalog(tmp_path / "catalog.sqlite")


@pytest.fixture
def catalog(tmp_path):
    releases = [
        {"title": "Abbey Road", "id": ABBEY, "artist_ids": [BEATLES], "release_group": RG_ABBEY},
        # Another title, same release group: still Abbey Road
        {"title": "Abbey Road (2019 Mix)", "id": ABBEY_REISSUE, "artist_ids": [BEATLES], "release_group": RG_ABBEY},
        {"title": "Debut", "id": DEBUT, "artist_ids": [BJORK]},
        {"title": "Homogenic", "id": HOMOGENIC, "artist_ids": [BJORK]},
    ]
    artists = {BEATLES: {"name": "Beatles, The"}, BJORK: {"name": "Björk"}}
    return build_catalog(tmp_path, releases, artists)


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
            signature=FixedEmbedder.signature,
        )
    assert build_index(sorted(shard_dir.glob("*.npz")), 4, shard_dir, M=16, ef_construction=100) == 4
    return shard_dir


def test_build_index_and_identify_roundtrip(catalog, index_dir):
    photo = Image.new("RGB", (64, 64))

    matches = Identifier(FixedEmbedder(E0), index_dir, catalog).identify(photo, k=3, views=1)

    # The 2019 mix is in Abbey Road's release group, so it collapses into the first match
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


def test_build_index_leaves_out_excluded_covers(catalog, index_dir):
    shards = sorted(index_dir.glob("*.npz"))
    assert build_index(shards, 4, index_dir, M=16, ef_construction=100, excluded={ABBEY, ABBEY_REISSUE}) == 2

    matches = Identifier(FixedEmbedder(E0), index_dir, catalog).identify(Image.new("RGB", (64, 64)), k=3, views=1)

    assert [m.mbid for m in matches] == [DEBUT, HOMOGENIC]


def test_an_index_built_under_another_exclusion_list_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(build_index_module, "load_exclusions", lambda gallery_dir: {ABBEY: "blank"})
    with pytest.raises(SystemExit, match="exclusion list"):
        require_current_exclusions(tmp_path)  # built before there was a list

    (tmp_path / "exclusions.txt").write_text(exclusions_hash({}))
    with pytest.raises(SystemExit, match="exclusion list"):
        require_current_exclusions(tmp_path)

    (tmp_path / "exclusions.txt").write_text(exclusions_hash({ABBEY: "blank"}))
    require_current_exclusions(tmp_path)


def test_build_index_rejects_wrong_dim(tmp_path):
    np.savez(tmp_path / "00.npz", uuids=np.array([ABBEY], dtype="U36"), embs=np.zeros((1, 3), np.float16),
             signature=FixedEmbedder.signature)

    with pytest.raises(SystemExit, match="00.npz"):
        build_index([tmp_path / "00.npz"], 4, tmp_path, M=16, ef_construction=100)


def test_build_index_skips_a_shard_with_no_covers(tmp_path):
    # hnswlib fails on zero rows only when they come first, so the empty prefix sorts first
    np.savez(tmp_path / "00.npz", uuids=np.zeros(0, dtype="U36"), embs=np.zeros((0, 4), np.float16),
             signature=FixedEmbedder.signature)  # a prefix the catalog has no covers for
    np.savez(tmp_path / "01.npz", uuids=np.array([ABBEY], dtype="U36"), embs=np.array([E0], dtype=np.float16),
             signature=FixedEmbedder.signature)

    assert build_index(sorted(tmp_path.glob("*.npz")), 4, tmp_path, M=16, ef_construction=100) == 1


def test_identify_widens_search_when_one_group_fills_the_neighbours(catalog, tmp_path):
    # 25 lookalikes of one cover outrank Debut, more than the first k*10 = 20 neighbours hold
    lookalikes = [f"{i:08x}-0000-4000-8000-000000000000" for i in range(25)]
    rng = np.random.default_rng(0)
    vectors = [unit(np.array(E0) + rng.normal(0, 0.01, 4)) for _ in lookalikes]
    np.savez(
        tmp_path / "00.npz",
        uuids=np.array([*lookalikes, DEBUT], dtype="U36"),
        embs=np.array([*vectors, unit([0.6, 0.8, 0, 0])], dtype=np.float16),
        signature=FixedEmbedder.signature,
    )
    build_index([tmp_path / "00.npz"], 4, tmp_path, M=16, ef_construction=100)
    clusters = dict.fromkeys(lookalikes, 1)

    matches = Identifier(FixedEmbedder(E0), tmp_path, catalog, clusters).identify(Image.new("RGB", (8, 8)), k=2, views=1)

    assert [m.mbid for m in matches][1:] == [DEBUT]


def test_identify_keeps_albums_without_a_release_group_apart(tmp_path):
    # "+" and "÷" have no release group here, so each stands alone
    ed, plus, divide = "44444444-4444-4444-8444-444444444444", "55555555-5555-4555-8555-555555555555", "66666666-6666-4666-8666-666666666666"
    build_catalog(tmp_path, [
        {"title": "+", "id": plus, "artist_ids": [ed]},
        {"title": "÷", "id": divide, "artist_ids": [ed]},
    ], {ed: {"name": "Ed Sheeran"}})
    np.savez(tmp_path / "00.npz", uuids=np.array([plus, divide], dtype="U36"),
             embs=np.array([E0, unit([0.9, 0.44, 0, 0])], dtype=np.float16), signature=FixedEmbedder.signature)
    build_index([tmp_path / "00.npz"], 4, tmp_path, M=16, ef_construction=100)

    matches = Identifier(FixedEmbedder(E0), tmp_path, Catalog(tmp_path / "catalog.sqlite")).identify(
        Image.new("RGB", (8, 8)), k=2, views=1
    )

    assert [m.title for m in matches] == ["+", "÷"]


def test_identify_keeps_same_titled_albums_apart(tmp_path):
    # Weezer's Blue and Green albums are both titled "Weezer" but are different release groups
    weezer, blue, green = "77777777-7777-4777-8777-777777777770", "77777777-7777-4777-8777-777777777771", "77777777-7777-4777-8777-777777777772"
    build_catalog(tmp_path, [
        {"title": "Weezer", "id": blue, "artist_ids": [weezer], "release_group": "88888888-8888-4888-8888-888888888881"},
        {"title": "Weezer", "id": green, "artist_ids": [weezer], "release_group": "88888888-8888-4888-8888-888888888882"},
    ], {weezer: {"name": "Weezer"}})
    np.savez(tmp_path / "00.npz", uuids=np.array([blue, green], dtype="U36"),
             embs=np.array([E0, unit([0.9, 0.44, 0, 0])], dtype=np.float16), signature=FixedEmbedder.signature)
    build_index([tmp_path / "00.npz"], 4, tmp_path, M=16, ef_construction=100)

    matches = Identifier(FixedEmbedder(E0), tmp_path, Catalog(tmp_path / "catalog.sqlite")).identify(
        Image.new("RGB", (8, 8)), k=2, views=1
    )

    assert [m.mbid for m in matches] == [blue, green]


def test_build_index_refuses_shards_from_different_preprocessing(tmp_path):
    # Prefix 01 was re-embedded after a preprocessing change; 00 still holds old vectors
    for prefix, signature in (("00", "fixed-8px-old"), ("01", FixedEmbedder.signature)):
        np.savez(tmp_path / f"{prefix}.npz", uuids=np.array([ABBEY], dtype="U36"),
                 embs=np.array([E0], dtype=np.float16), signature=signature)

    with pytest.raises(SystemExit, match="different embedders"):
        build_index(sorted(tmp_path.glob("*.npz")), 4, tmp_path, M=16, ef_construction=100)


def test_identify_refuses_an_index_from_another_embedder(catalog, index_dir):
    class Changed(FixedEmbedder):
        signature = "fixed-8px-changed"

    with pytest.raises(SystemExit, match="fixed-8px-test, not fixed-8px-changed"):
        Identifier(Changed(E0), index_dir, catalog)
