import copy

import numpy as np
import pytest
import torch
from PIL import Image

from pipeline.retrieval.embedders import BACKBONES, EMBEDDER_NAMES, exact_layer_norms, load_embedder, query_views


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
@pytest.mark.parametrize("name", EMBEDDER_NAMES)
def test_each_embedder_unit_vectors(name):
    from huggingface_hub import get_token  # HF_TOKEN, else the token `hf auth login` saved

    if BACKBONES[name].gated and get_token() is None:
        pytest.skip(f"{name} is gated on Hugging Face: set HF_TOKEN or run `hf auth login`")
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


class ChannelsFirstLayerNorm(torch.nn.LayerNorm):
    """Normalizes (batch, channels, h, w) over channels, like DINOv3 ConvNeXt's LayerNorm."""

    data_format = "channels_first"

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


@pytest.mark.parametrize("layer, shape", [(torch.nn.LayerNorm(96), (2, 5, 96)), (ChannelsFirstLayerNorm(96), (2, 96, 4, 4))])
def test_exact_layer_norms_matches_torch(layer, shape):
    torch.manual_seed(0)
    torch.nn.init.normal_(layer.weight)
    torch.nn.init.normal_(layer.bias)
    x = torch.randn(shape)
    expected = layer(x)

    patched = copy.deepcopy(layer)
    exact_layer_norms(patched)

    torch.testing.assert_close(patched(x), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple's MPS")
def test_exact_layer_norms_fixes_mps_on_flat_vectors():
    # A large mean and a tiny spread, as on a flat area of a cover: torch 2.8's fused MPS
    # layer_norm gets this wrong
    torch.manual_seed(0)
    x = 1 + 1e-3 * torch.randn(4, 96)
    layer = torch.nn.LayerNorm(96, eps=1e-6)
    expected = layer.double()(x.double()).float()

    exact_layer_norms(layer)

    torch.testing.assert_close(layer.float().to("mps")(x.to("mps")).cpu(), expected, rtol=1e-4, atol=1e-4)
