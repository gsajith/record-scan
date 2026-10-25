import numpy as np
import pytest
from PIL import Image

from pipeline.retrieval.embedders import load_embedder, query_views


def test_query_views_three_sizes():
    # 400x300: red border 50 px wide on the left and right, so the center square is all blue
    img = Image.new("RGB", (400, 300), (255, 0, 0))
    img.paste((0, 0, 255), (50, 0, 350, 300))

    crop, tight, padded = query_views(img)

    assert crop.size == (300, 300)
    assert tight.size == (270, 270)
    assert padded.size == (429, 429)  # 300 / 0.7
    for view in (crop, tight):
        assert view.getpixel((0, 0)) == (0, 0, 255)
    assert padded.getpixel((0, 0)) == (128, 128, 128)
    assert padded.getpixel((214, 214)) == (0, 0, 255)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["clip-vit-b-32"])
def test_each_embedder_unit_vectors(name):
    embedder = load_embedder(name)
    images = [Image.new("RGB", (500, 500), (200, 30, 30)), Image.new("RGB", (320, 480), (20, 90, 200))]

    embs = embedder.embed(images)

    assert embs.shape == (2, embedder.dim)
    assert embs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(embs[0], embs[1])


def test_query_views_apply_exif_rotation(tmp_path):
    # Stored sideways, red on the left, with the orientation tag phones write for a portrait shot
    img = Image.new("RGB", (400, 300), (0, 0, 255))
    img.paste((255, 0, 0), (0, 0, 200, 300))
    exif = Image.Exif()
    exif[0x0112] = 6  # display rotated 90 degrees clockwise
    img.save(tmp_path / "photo.jpg", exif=exif)

    crop = query_views(Image.open(tmp_path / "photo.jpg"))[0]

    # Upright, the red half is on top
    assert crop.getpixel((150, 10))[0] > 200
    assert crop.getpixel((150, 290))[2] > 200
