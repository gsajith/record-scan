"""What gets exported: an embedder's forward pass from preprocessed pixels to a unit vector.

Preprocessing (resize and normalize) stays outside the export; the phone app does it. Each
export gets a `<file>.json` beside it naming the embedder signature that produced it, and
parity refuses an export whose signature no longer matches the embedder (Review focus 7).
"""

import json
from pathlib import Path

import torch

from pipeline import config
from pipeline.retrieval.embedders import Embedder, load_embedder


class EmbedderModule(torch.nn.Module):
    def __init__(self, embedder: Embedder):
        super().__init__()
        self.model = embedder.model  # registers the weights so torch.export lifts them
        self._features = embedder.features

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self._features(pixels).float(), dim=-1)


def load(name: str, px: int | None) -> tuple[Embedder, EmbedderModule, torch.Tensor]:
    """The embedder on CPU, its exportable module, and a one-image example input."""
    embedder = load_embedder(name, device="cpu", input_px=px)
    example = torch.zeros(1, 3, embedder.input_px, embedder.input_px)
    return embedder, EmbedderModule(embedder), example


def export_path(embedder: Embedder, variant: str) -> Path:
    """EXPORTS_DIR/<model>-<px>-<variant>, without a suffix: each exporter adds its own."""
    return config.EXPORTS_DIR / f"{embedder.name}-{embedder.input_px}-{variant}"


def sidecar(exported: Path) -> Path:
    return exported.with_name(exported.name + ".json")


def write_sidecar(exported: Path, embedder: Embedder, **extra: str) -> None:
    info = {"signature": embedder.signature, "torch": torch.__version__, **extra}
    sidecar(exported).write_text(json.dumps(info, indent=2) + "\n")


def check_sidecar(exported: Path, embedder: Embedder) -> None:
    path = sidecar(exported)
    found = json.loads(path.read_text())["signature"] if path.is_file() else None
    if found != embedder.signature:
        raise SystemExit(f"{exported.name} was exported by {found or 'an unknown embedder'}, "
                         f"not {embedder.signature}. Export it again.")
