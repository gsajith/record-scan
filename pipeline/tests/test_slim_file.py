from pipeline.metadata.slim_file import slim_record

FLOYD = "83d91898-7763-47d7-b03b-b92132375c47"


def dump_line(**overrides):
    """One release in the JSON dump shape, trimmed to the keys slim_record reads plus noise."""
    line = {
        "id": "65ee4ee0-2eb7-33f9-b1b3-aeb8b192a66c",
        "title": "The Division Bell",
        "asin": "B000002A3T",
        "disambiguation": "",
        "barcode": "074646420027",
        "aliases": [{"name": "Division Bell"}],
        "media": [{"format": "CD", "tracks": [{"title": "Cluster One"}]}],
        "cover-art-archive": {"front": True, "artwork": True, "count": 19},
        "release-group": {"id": "90878b63-f639-3c8b-aefb-190bdf3d1790", "title": "The Division Bell"},
        "artist-credit": [
            {"name": "Pink Floyd", "joinphrase": "", "artist": {
                "id": FLOYD, "name": "Pink Floyd", "disambiguation": "",
                "aliases": [{"name": "Floyd"}, {"name": "The Pink Floyd"}],
            }},
        ],
    }
    return line | overrides


def test_slim_record_keeps_release_group():
    assert slim_record(dump_line()) == {
        "release": {
            "title": "The Division Bell",
            "id": "65ee4ee0-2eb7-33f9-b1b3-aeb8b192a66c",
            "artist_ids": [FLOYD],
            "asin": "B000002A3T",
            "format": "CD",
            "aliases": ["Division Bell"],
            "release_group": "90878b63-f639-3c8b-aefb-190bdf3d1790",
        },
        "artists": {FLOYD: {"name": "Pink Floyd", "aliases": ["Floyd", "The Pink Floyd"]}},
    }


def test_slim_record_skips_releases_without_art():
    assert slim_record(dump_line(**{"cover-art-archive": {"front": False, "artwork": False}})) is None


def test_slim_record_keeps_art_without_a_front_flag():
    # 2025 kept these too: an image may be the front without being typed "Front" yet
    assert slim_record(dump_line(**{"cover-art-archive": {"front": False, "artwork": True}})) is not None
