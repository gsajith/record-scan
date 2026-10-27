import json
import re

import pytest

from pipeline.metadata.build_catalog_db import build
from pipeline.metadata.catalog import Catalog, Cover, Release, normalize

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


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def write_fake_dump(tmp_path):
    """Three releases in the slim shape with an unknown artist id, plus a covers list."""
    releases = [
        {"title": "Abbey Road", "id": ABBEY, "artist_ids": [BEATLES], "format": "12\" Vinyl"},
        {"title": "Debut", "id": DEBUT, "artist_ids": [BJORK, "no-such-artist"], "release_group": "rg-2"},
        {"title": "Abbey Road Revisited", "id": TRIBUTE, "artist_ids": ["no-such-artist"]},
    ]
    artists = {
        BEATLES: {"name": "Beatles, The", "aliases": ["The Beatles"]},
        BJORK: {"name": "Björk"},
    }
    covers = [  # TRIBUTE has none: its lookup found no usable image
        {"mbid": ABBEY, "cover_id": "123", "file": f"11/{ABBEY}.jpg", "primary": True},
        {"mbid": DEBUT, "cover_id": "457", "file": f"22/{DEBUT}-457.jpg", "primary": False},
        {"mbid": DEBUT, "cover_id": "456", "file": f"22/{DEBUT}.jpg", "primary": True},
    ]
    artists_json = tmp_path / "artists.json"
    artists_json.write_text(json.dumps(artists))
    return (write_jsonl(tmp_path / "releases.jsonl", releases), artists_json,
            write_jsonl(tmp_path / "covers.jsonl", covers))


@pytest.fixture
def catalog(tmp_path):
    out = tmp_path / "catalog.sqlite"
    assert build(*write_fake_dump(tmp_path), out) == 3
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


def test_build_takes_covers_and_release_groups(catalog):
    debut = catalog.release(DEBUT)
    assert (debut.cover_id, debut.release_group) == ("456", "rg-2")
    # Primary first, then extra fronts
    assert catalog.covers(DEBUT) == [Cover(DEBUT, "456", f"22/{DEBUT}.jpg", True),
                                     Cover(DEBUT, "457", f"22/{DEBUT}-457.jpg", False)]
    assert catalog.release(TRIBUTE).cover_id is None
    assert catalog.covers(TRIBUTE) == []


def test_cover_files_lists_a_prefix_folder_by_file_name(catalog):
    assert catalog.cover_files("22") == [f"22/{DEBUT}-457.jpg", f"22/{DEBUT}.jpg"]
    assert catalog.cover_files("11") == [f"11/{ABBEY}.jpg"]
    assert catalog.cover_files("33") == []  # TRIBUTE has no cover


def test_missing_db_exits(reload_config, tmp_path):
    with pytest.raises(SystemExit, match="build_catalog_db"):
        Catalog(tmp_path / "nope.sqlite")
    # The default path comes from config at call time
    reload_config(RS_DATA_DIR=tmp_path)
    with pytest.raises(SystemExit, match=re.escape(str(tmp_path))):
        Catalog()
