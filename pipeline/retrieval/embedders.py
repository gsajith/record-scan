"""Image backbones behind one interface, plus the query views every identify and eval run uses.

Only the CLIP ViT-B/32 baseline is wired up so far. Gallery covers and query photos go
through the same `embed`, so the two sides of a match are always preprocessed alike.
"""

from typing import Protocol

import numpy as np
import torch
from PIL import Image, ImageOps

from pipeline import config

EMBEDDER_NAMES = ("clip-vit-b-32",)

PAD_GRAY = (128, 128, 128)


class Embedder(Protocol):
    name: str
    dim: int
    input_px: int

    def preprocess(self, img: Image.Image) -> torch.Tensor: ...

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """(n, dim) float32, each row L2-normalized."""
        ...


class ClipEmbedder:
    """open_clip ViT-B-32 with the OpenAI weights (cached as timm/vit_base_patch32_clip_224.openai).

    The -quickgelu config matches how OpenAI trained them; plain ViT-B-32 swaps in GELU and
    shifts every vector.
    """

    name = "clip-vit-b-32"
    input_px = 224

    def __init__(self, device: str):
        import open_clip  # deferred: importing it pulls in timm and the hub client

        self.device = device
        self.model, _, self._preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        self.model.to(device).eval()
        self.dim = self.model.visual.output_dim

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self._preprocess(img.convert("RGB"))

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        embs = self.model.encode_image(batch).float()
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs.cpu().numpy()


def load_embedder(name: str, device: str | None = None, input_px: int | None = None) -> Embedder:
    if name not in EMBEDDER_NAMES:
        raise SystemExit(f"Unknown model {name!r}. Choose one of: {', '.join(EMBEDDER_NAMES)}")
    if input_px not in (None, ClipEmbedder.input_px):
        raise SystemExit(f"{name} runs at {ClipEmbedder.input_px} px only")
    return ClipEmbedder(device or config.device())


def query_views(img: Image.Image) -> list[Image.Image]:
    """The three query scales: center square crop, a 10% tighter crop, and the crop on gray padding.

    The padded view puts the cover at 70% of the canvas side. Callers max-pool scores across
    views. Phone photos are turned upright from their EXIF orientation first.
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    crop = _center_square(img, min(img.size))
    tight = _center_square(img, round(min(img.size) * 0.9))
    side = round(crop.width / 0.7)
    padded = Image.new("RGB", (side, side), PAD_GRAY)
    offset = (side - crop.width) // 2
    padded.paste(crop, (offset, offset))
    return [crop, tight, padded]


def _center_square(img: Image.Image, side: int) -> Image.Image:
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    return img.crop((left, top, left + side, top + side))
