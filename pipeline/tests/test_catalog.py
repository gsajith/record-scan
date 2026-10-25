import json
import re

import pytest

from pipeline.metadata.build_catalog_db import build
from pipeline.metadata.catalog import Catalog, Release, normalize

BEATLES = "b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d"
BJORK = "87c5dedd-371d-4a53-9f7f-80522fb7f3cb"
ABBEY = "11111111-1111-4111-8111-111111111111"
DEBUT = "22222222-2222-4222-8222-222222222222"
TRIBUTE = "33333333-3333-4333-8333-333333333333"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("The Beatles", "beatles"),
        ("Beatles, The", "beatles"),
        ("Björk", "bjork"),
        ("Abbey Road (Remastered)", "abbey road remastered"),
        ("  Don't   Stop  ", "dont stop"),
        ("AC/DC", "ac dc"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def write_fake_dump(tmp_path):
    """Three releases in the slim dump shape, with an unknown artist id and mixed cover_id types."""
    releases = [
        {"title": "Abbey Road", "id": ABBEY, "artist_ids": [BEATLES], "format": "12\" Vinyl", "cover_id": 123},
        {"title": "Debut", "id": DEBUT, "artist_ids": [BJORK, "no-such-artist"], "cover_id": "456"},
        {"title": "Abbey Road Revisited", "id": TRIBUTE, "artist_ids": ["no-such-artist"]},
    ]
    artists = {
        BEATLES: {"name": "Beatles, The", "aliases": ["The Beatles"]},
        BJORK: {"name": "Björk"},
    }
    releases_json = tmp_path / "releases.json"
    artists_json = tmp_path / "artists.json"
    releases_json.write_text(json.dumps(releases))
    artists_json.write_text(json.dumps(artists))
    return releases_json, artists_json


@pytest.fixture
def catalog(tmp_path):
    releases_json, artists_json = write_fake_dump(tmp_path)
    out = tmp_path / "catalog.sqlite"
    assert build(releases_json, artists_json, out, delta_dir=None) == 3
    assert not out.with_name(out.name + ".tmp").exists()
    return Catalog(out)


def mbids(results):
    return [r.mbid for r in results]


def test_build_and_search_tmp_db(catalog):
    assert catalog.count() == 3
    assert mbids(catalog.search("bjork")) == [DEBUT]
    assert mbids(catalog.search("beatl")) == [ABBEY]
    assert mbids(catalog.search("the beatles abbey road")) == [ABBEY]
    assert set(mbids(catalog.search("abbey road"))) == {ABBEY, TRIBUTE}
    # Empty, punctuation-only and FTS-syntax queries never reach MATCH raw
    assert catalog.search("") == []
    assert catalog.search("!!! ...") == []
    assert catalog.search('"beatles OR NEAR(') == []

    assert catalog.release(ABBEY) == Release(
        mbid=ABBEY, title="Abbey Road", artists=("Beatles, The",),
        format='12" Vinyl', cover_id="123", release_group=None,
    )
    assert catalog.release(DEBUT).artists == ("Björk",)
    assert catalog.release(TRIBUTE).artists == ()
    assert catalog.release("missing") is None
    assert catalog.album_key(ABBEY) == "abbey road|beatles"


def test_find_exact(catalog):
    assert catalog.find("The Beatles", "abbey road") == [ABBEY]
    assert catalog.find("bjork", "DEBUT") == [DEBUT]
    assert catalog.find("Beatles", "Abbey Road Revisited") == []


def test_build_ingests_delta(tmp_path):
    releases_json, artists_json = write_fake_dump(tmp_path)
    delta = tmp_path / "delta"
    (delta / "releases").mkdir(parents=True)
    (delta / "artists").mkdir()
    new_artist = "44444444-4444-4444-8444-444444444444"
    new_release = "55555555-5555-4555-8555-555555555555"
    (delta / "artists" / "2026-11-01.jsonl").write_text(
        json.dumps({"id": new_artist, "name": "Sigur Rós", "aliases": []}) + "\n"
    )
    (delta / "releases" / "2026-11-01.jsonl").write_text(
        json.dumps({"title": "Ágætis byrjun", "id": new_release, "artist_ids": [new_artist],
                    "cover_id": 789, "release_group": "rg-1"}) + "\n"
        + json.dumps({"title": "Debut (Deluxe)", "id": DEBUT, "artist_ids": [BJORK],
                      "release_group": "rg-2"}) + "\n"
    )
    out = tmp_path / "catalog.sqlite"
    assert build(releases_json, artists_json, out, delta_dir=delta) == 4

    catalog = Catalog(out)
    assert catalog.release(new_release) == Release(
        mbid=new_release, title="Ágætis byrjun", artists=("Sigur Rós",),
        format=None, cover_id="789", release_group="rg-1",
    )
    assert mbids(catalog.search("sigur ros byrj")) == [new_release]
    # A delta row replaces the dump row with the same mbid
    assert catalog.release(DEBUT).title == "Debut (Deluxe)"
    assert catalog.release(DEBUT).release_group == "rg-2"


def test_missing_db_exits(reload_config, tmp_path):
    with pytest.raises(SystemExit, match="build_catalog_db"):
        Catalog(tmp_path / "nope.sqlite")
    # The default path comes from config at call time
    reload_config(RS_DATA_DIR=tmp_path)
    with pytest.raises(SystemExit, match=re.escape(str(tmp_path))):
        Catalog()
