import numpy as np
import pytest
from PIL import Image, ImageDraw

from pipeline.gallery import exclusions
from pipeline.gallery.cluster_duplicates import VARIOUS_ARTISTS
from pipeline.gallery.exclusions import (MEASURES_DTYPE, excluded, exclusions_hash, load_exclusions, load_measures,
                                         measure_prefix, spread, write_exclusions)
from pipeline.tests.test_gallery import write_catalog
from pipeline.gallery.shard_and_hash import process_prefix


def mbid(i):
    return f"{i:08x}-0000-4000-8000-000000000000"


def noise(size, seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).resize(size, Image.Resampling.NEAREST)


def blank(size, color=(250, 250, 250)):
    return Image.new("RGB", size, color)


def test_spread_is_near_zero_for_a_flat_image_and_high_for_a_picture():
    motif = blank((500, 500), (200, 30, 30))
    ImageDraw.Draw(motif).ellipse((200, 200, 300, 300), fill=(250, 250, 250))  # a small motif on a plain color

    assert spread(blank((500, 500))) < 1
    assert 1 < spread(motif) < spread(noise((500, 500), 1))


@pytest.fixture
def gallery(tmp_path, reload_config):
    """Prefix 00 sharded from a tmp drive: a picture, a blank and a strip."""
    root = tmp_path / "images"
    (root / "00").mkdir(parents=True)
    noise((500, 500), 1).save(root / "00" / f"{mbid(0)}.jpg")
    blank((500, 500)).save(root / "00" / f"{mbid(1)}.jpg")
    noise((500, 116), 2).save(root / "00" / f"{mbid(2)}.jpg")
    reload_config(RS_IMAGE_ROOT=root)
    out = tmp_path / "gallery"
    process_prefix("00", [f"00/{mbid(i)}.jpg" for i in range(3)], out, workers=1)
    return out


def test_measures_join_each_covers_size_with_its_spread(gallery):
    assert measure_prefix("00", gallery) == 3

    rows = load_measures(gallery)

    assert rows["key"].tolist() == [mbid(0), mbid(1), mbid(2)]
    assert rows["width"].tolist() == [500, 500, 500] and rows["height"].tolist() == [500, 500, 116]
    assert rows["spread"][1] < 1 < rows["spread"][0]


def test_a_measured_prefix_is_skipped_until_its_shard_is_redone(gallery):
    measure_prefix("00", gallery)
    assert measure_prefix("00", gallery) == 0

    (gallery / "00.done").write_text((gallery / "00.done").read_text() + "0")  # the copy was redone

    with pytest.raises(ValueError, match="00 .*rerun"):
        load_measures(gallery)
    assert measure_prefix("00", gallery) == 3
    load_measures(gallery)


def test_measures_refuse_another_spread_recipe_or_an_unmeasured_prefix(gallery, monkeypatch):
    with pytest.raises(ValueError, match="00 .*rerun"):
        load_measures(gallery)
    measure_prefix("00", gallery)
    monkeypatch.setattr(exclusions, "SPREAD_RECIPE", "spread-other")

    with pytest.raises(ValueError, match="00 .*rerun"):
        load_measures(gallery)


PLACEHOLDER, PICTURE, SINGLE = 0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210, 0x0F0F_0F0F_0F0F_0F0F
RULE = {"blank_spread": 1.0, "strip_aspect": 2.5, "shared_release_groups": 5, "shared_first_artists": 10}


def cover(key, phash=None, width=500, height=500, spread=40.0):
    """A MEASURES_DTYPE row; by default a picture of its own."""
    return key, int(key[:8], 16) * 0x9E37_79B9 if phash is None else phash, width, height, spread


def test_placeholders_strips_and_blanks_are_excluded_but_small_covers_and_one_albums_copies_are_kept():
    placeholder = [mbid(i) for i in range(10)]  # one picture over ten unrelated albums
    album = [mbid(20), mbid(21)]  # an album and its reissue, one picture
    strip, flat, small = mbid(30), mbid(31), mbid(32)
    measures = np.array([cover(m, PLACEHOLDER) for m in placeholder] + [cover(m, PICTURE) for m in album]
                        + [cover(strip, width=500, height=116), cover(flat, spread=0.5),
                           cover(small, width=200, height=200)], dtype=MEASURES_DTYPE)
    artists = {m: [f"artist-{m}"] for m in placeholder} | {m: ["band"] for m in album}
    release_groups = {m: f"rg-{m}" for m in placeholder} | {m: "rg-album" for m in album}

    out = excluded(measures, artists, release_groups, **RULE)

    assert dict(zip(out["key"].tolist(), out["reason"].tolist())) == \
        {**{m: "shared" for m in placeholder}, strip: "strip", flat: "blank"}


def test_a_single_with_many_remixers_is_kept_since_only_first_artists_count():
    remixes = [mbid(i) for i in range(12)]  # one single's remix releases, each crediting another remixer
    measures = np.array([cover(m, SINGLE) for m in remixes], dtype=MEASURES_DTYPE)
    artists = {m: ["deadmau5", f"remixer-{i}"] for i, m in enumerate(remixes)}

    assert len(excluded(measures, artists, {m: f"rg-{m}" for m in remixes}, **RULE)) == 0
    # Various Artists is skipped, so a compilation counts its first named artist
    compilations = {m: [VARIOUS_ARTISTS, f"artist-{i}"] for i, m in enumerate(remixes)}
    assert len(excluded(measures, compilations, {m: f"rg-{m}" for m in remixes}, **RULE)) == 12


@pytest.fixture
def written(gallery, tmp_path, reload_config):
    """The gallery fixture measured, with a catalog, and its exclusion list written."""
    reload_config(RS_DATA_DIR=tmp_path)  # GALLERY_DIR is where the gallery fixture wrote
    write_catalog({mbid(i): ["x"] for i in range(3)})
    measure_prefix("00", gallery)
    write_exclusions(gallery)
    return gallery


def test_the_written_list_loads_back_with_a_reason_per_cover(written):
    assert load_exclusions(written) == {mbid(1): "blank", mbid(2): "strip"}


def test_loading_the_list_refuses_one_made_at_other_cutoffs_or_from_another_gallery_or_catalog(written, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(exclusions, "STRIP_ASPECT", 3.0)
        with pytest.raises(ValueError, match="stale"):
            load_exclusions(written)
    with monkeypatch.context() as m:
        m.setattr(exclusions, "EXCLUDE_RULE", "other-rule")
        with pytest.raises(ValueError, match="stale"):
            load_exclusions(written)

    write_catalog({mbid(i): ["y"] for i in range(3)})
    with pytest.raises(ValueError, match="stale"):
        load_exclusions(written)
    write_catalog({mbid(i): ["x"] for i in range(3)})
    load_exclusions(written)

    (written / "00.done").write_text((written / "00.done").read_text() + "0")  # the copy was redone
    with pytest.raises(ValueError, match="stale"):
        load_exclusions(written)


def test_the_list_hash_changes_with_any_key_or_reason():
    base = exclusions_hash({mbid(1): "blank"})

    assert base == exclusions_hash({mbid(1): "blank"})
    assert len({base, exclusions_hash({mbid(1): "strip"}), exclusions_hash({mbid(2): "blank"}),
                exclusions_hash({})}) == 4
