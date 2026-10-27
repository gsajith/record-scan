"""How closely an exported model reproduces PyTorch: the mean cosine between their outputs.

    python -m pipeline.export.parity --model dinov3-vits16 --exported EXPORTS_DIR/dinov3-vits16-224-fp16.mlpackage

Scores 20 gallery covers from prefix 00 on the drive. The bar is 0.99.
"""

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pipeline import config
from pipeline.retrieval.embedders import EMBEDDER_NAMES


def load_exported(path: Path) -> Callable[[torch.Tensor], np.ndarray]:
    """A function running one (1, 3, px, px) input through a .mlpackage or a .pte file."""
    if path.suffix == ".mlpackage":
        import coremltools as ct

        mlmodel = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
        spec = mlmodel.get_spec().description
        name_in, name_out = spec.input[0].name, spec.output[0].name
        return lambda x: np.asarray(mlmodel.predict({name_in: x.numpy()})[name_out])
    if path.suffix == ".pte":
        from executorch.runtime import Runtime

        method = Runtime.get().load_program(path).load_method("forward")
        return lambda x: method.execute([x])[0].numpy()
    raise ValueError(f"Not an export: {path}")


@torch.no_grad()
def mean_cosine(reference: torch.nn.Module, exported: Path, inputs: torch.Tensor) -> float:
    """Run each input through both, one at a time as the phone does, and average the cosines."""
    run = load_exported(exported)
    want = reference.eval()(inputs).numpy()
    got = np.concatenate([run(x[None]).reshape(1, -1) for x in inputs])
    want = want / np.linalg.norm(want, axis=1, keepdims=True)
    got = got / np.linalg.norm(got, axis=1, keepdims=True)
    return float(np.sum(want * got, axis=1).mean())


def parity(model: str, exported: Path, images: list[Path], px: int | None = None) -> float:
    """Mean cosine between the PyTorch embedder on CPU and its export, on these images."""
    from pipeline.export.model import check_sidecar, load

    embedder, module, _ = load(model, px)
    check_sidecar(exported, embedder)
    inputs = torch.stack([embedder.preprocess(Image.open(p)) for p in images])
    return mean_cosine(module, exported, inputs)


def latency_ms(exported: Path, example: torch.Tensor, runs: int = 50) -> tuple[float, float]:
    """Median and p95 milliseconds per run on this Mac, after five warm-up runs."""
    run = load_exported(exported)
    for _ in range(5):
        run(example)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        run(example)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def gallery_sample(n: int = 20) -> list[Path]:
    """n covers spread evenly over prefix 00 on the drive, the same n on every run."""
    covers = sorted(p for p in (config.require_image_root() / "00").glob("*.jpg") if config.is_uuid(p.stem))
    return covers[:: len(covers) // n][:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=EMBEDDER_NAMES)
    parser.add_argument("--exported", required=True, type=Path)
    parser.add_argument("--px", type=int, help="input size, default the model's own")
    parser.add_argument("--n", type=int, default=20, help="gallery covers to score")
    args = parser.parse_args()

    score = parity(args.model, args.exported, gallery_sample(args.n), args.px)
    print(f"{args.exported.name}: mean cosine {score:.6f} on {args.n} covers ({'pass' if score >= 0.99 else 'FAIL'}, bar 0.99)")


if __name__ == "__main__":
    main()
