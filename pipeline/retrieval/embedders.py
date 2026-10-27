"""Image backbones behind one interface, plus the query views every identify and eval run uses.

Gallery covers and query photos go through the same `embed`, so the two sides of a match
are always preprocessed alike. Each backbone uses the preprocessing on its model card.

The DINOv3 weights are gated: accept the license on Hugging Face, then `hf auth login` or
set HF_TOKEN.
"""

import functools
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from PIL import Image, ImageOps

from pipeline import config


@dataclass(frozen=True)
class Backbone:
    repo: str
    dim: int
    input_px: int
    resizable: bool = False  # DINOv3 uses RoPE, so it runs at any multiple of 16 px
    gated: bool = False


# Sizes and sources checked against each model's config.json and card on 2026-10-27
BACKBONES = {
    "clip-vit-b-32": Backbone("timm/vit_base_patch32_clip_224.openai", 512, 224),
    "dinov3-vits16": Backbone("facebook/dinov3-vits16-pretrain-lvd1689m", 384, 224, resizable=True, gated=True),
    "dinov3-vitsplus16": Backbone("facebook/dinov3-vits16plus-pretrain-lvd1689m", 384, 224, resizable=True, gated=True),
    "dinov3-convnext-tiny": Backbone("facebook/dinov3-convnext-tiny-pretrain-lvd1689m", 768, 224, resizable=True, gated=True),
    "dinov3-vitb16": Backbone("facebook/dinov3-vitb16-pretrain-lvd1689m", 768, 224, resizable=True, gated=True),
    "siglip2-b16-256": Backbone("google/siglip2-base-patch16-256", 768, 256),
    "pe-core-s16": Backbone("timm/vit_pe_core_small_patch16_384.fb", 512, 384),
}
EMBEDDER_NAMES = tuple(BACKBONES)


def exact_layer_norms(model: torch.nn.Module) -> None:
    """Swap every LayerNorm's forward for the plain formula. On MPS (torch 2.8) the fused
    layer_norm goes wrong when a vector has a large mean and a tiny spread: on flat areas of a
    cover, DINOv3 ConvNeXt's first LayerNorm came out 13% off CPU and float64, while the plain
    formula on MPS matches float64 to 1e-7."""
    for mod in model.modules():
        if isinstance(mod, torch.nn.LayerNorm):
            # _layer_norm handles one normalized dim, with weight and bias both or neither
            assert len(mod.normalized_shape) == 1 and (mod.weight is None) == (mod.bias is None), mod
            mod.forward = functools.partial(_layer_norm, mod)


def _layer_norm(mod: torch.nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    # DINOv3 ConvNeXt's LayerNorm subclass also normalizes (batch, channels, h, w) inputs
    channels_first = getattr(mod, "data_format", "channels_last") == "channels_first"
    dim = 1 if channels_first else -1
    x = x - x.mean(dim, keepdim=True)
    x = x * torch.rsqrt(x.pow(2).mean(dim, keepdim=True) + mod.eps)
    if mod.weight is None:
        return x
    shape = (-1, 1, 1) if channels_first else (-1,)
    return x * mod.weight.view(shape) + mod.bias.view(shape)


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
        if device == "mps":
            exact_layer_norms(self.model)
        self.dim = self.model.visual.output_dim

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self._preprocess(img.convert("RGB"))

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        embs = self.model.encode_image(batch).float()
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs.cpu().numpy()


class HfEmbedder:
    """A Hugging Face transformers backbone with the image processor from its repo.

    DINOv3 returns the final-norm CLS token for ViTs and the normed global average pool for
    ConvNeXt as `pooler_output`. SigLIP2 pools through its attention head in
    `get_image_features`.
    """

    def __init__(self, name: str, backbone: Backbone, input_px: int, device: str):
        from transformers import AutoImageProcessor, AutoModel  # deferred: a slow import

        self.name = name
        self.dim = backbone.dim
        self.input_px = input_px
        self.device = device
        self._processor = AutoImageProcessor.from_pretrained(backbone.repo, size={"height": input_px, "width": input_px})
        self.model = AutoModel.from_pretrained(backbone.repo).to(device).eval()
        if device == "mps":
            exact_layer_norms(self.model)
        self._siglip = name.startswith("siglip")

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self._processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        out = self.model.get_image_features(pixel_values=batch) if self._siglip else self.model(pixel_values=batch)
        embs = out.pooler_output.float()
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs.cpu().numpy()


class PeCoreEmbedder:
    """PE-Core S/16 at 384 px through timm. Keeps the head, a projection into the 512-d CLIP
    space, which is the image embedding on the model card.

    Preprocessing follows Meta's reference (perception_models get_image_transform), which
    squashes the whole image to 384x384 with bilinear resizing, as in training. timm's own
    config resizes the short side bicubically and center-crops instead.
    """

    name = "pe-core-s16"
    dim = 512
    input_px = 384

    def __init__(self, backbone: Backbone, device: str):
        import timm
        import torchvision.transforms as T

        self.device = device
        self.model = timm.create_model(backbone.repo.removeprefix("timm/"), pretrained=True).to(device).eval()
        # The attention pool feeds its learned query, expanded to the batch without a copy, into a
        # Linear. On MPS (torch 2.8) that aborts the process with "buffer is not large enough";
        # a contiguous copy fixes it and matches CPU output.
        self.model.attn_pool.q.register_forward_pre_hook(lambda _, args: (args[0].contiguous(),))
        if device == "mps":
            exact_layer_norms(self.model)
        self._preprocess = T.Compose([
            T.Resize((self.input_px, self.input_px), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return self._preprocess(img.convert("RGB"))

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        embs = self.model(batch).float()
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs.cpu().numpy()


def load_embedder(name: str, device: str | None = None, input_px: int | None = None) -> Embedder:
    """Load a backbone by name. Only DINOv3 takes an input_px other than its default."""
    if name not in BACKBONES:
        raise SystemExit(f"Unknown model {name!r}. Choose one of: {', '.join(EMBEDDER_NAMES)}")
    backbone = BACKBONES[name]
    px = input_px or backbone.input_px
    if px != backbone.input_px and not (backbone.resizable and px % 16 == 0):
        allowed = "any multiple of 16 px" if backbone.resizable else f"{backbone.input_px} px only"
        raise SystemExit(f"{name} runs at {allowed}, not {px}")
    device = device or config.device()
    try:
        if name == "clip-vit-b-32":
            return ClipEmbedder(device)
        if name == "pe-core-s16":
            return PeCoreEmbedder(backbone, device)
        return HfEmbedder(name, backbone, px, device)
    except OSError as e:  # transformers rewraps the hub's GatedRepoError as a plain OSError
        if backbone.gated and "gated" in str(e).lower():
            raise SystemExit(
                f"{name} is gated. Accept its license at https://huggingface.co/{backbone.repo}, "
                "then run `hf auth login` or set HF_TOKEN."
            ) from e
        raise


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
