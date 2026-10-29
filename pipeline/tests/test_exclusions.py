import numpy as np
import pytest
from PIL import Image, ImageDraw

from pipeline.gallery import exclusions
from pipeline.gallery.exclusions import load_measures, measure_prefix, spread
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
