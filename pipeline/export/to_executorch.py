"""Export an embedder to an ExecuTorch program and time it on this Mac's CPU.

    python -m pipeline.export.to_executorch --model dinov3-vits16 --px 224 --backend xnnpack

Writes EXPORTS_DIR/<model>-<px>-xnnpack.pte. Only XNNPACK for now: the Qualcomm (QNN) backend
needs an ExecuTorch built from source against Qualcomm's SDK; the pip wheel lacks it.
"""

import argparse
import os
from importlib.resources import files
from pathlib import Path

import torch

from pipeline import config
from pipeline.retrieval.embedders import EMBEDDER_NAMES


def export_executorch(module: torch.nn.Module, example: torch.Tensor, out: Path) -> Path:
    """Lower to XNNPACK, ExecuTorch's CPU backend on both phones."""
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    # Serializing runs the flatc the wheel bundles, which ExecuTorch looks for on PATH: found only
    # when the venv is activated, not when its python is called directly
    os.environ.setdefault("FLATC_EXECUTABLE", str(files("executorch.data.bin") / "flatc"))
    program = torch.export.export(module.eval(), (example,))
    lowered = to_edge_transform_and_lower(program, partitioner=[XnnpackPartitioner()]).to_executorch()
    path = out.with_suffix(".pte")
    path.write_bytes(lowered.buffer)
    return path


def main() -> None:
    from pipeline.export.model import export_path, load, write_sidecar
    from pipeline.export.parity import latency_ms

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=EMBEDDER_NAMES)
    parser.add_argument("--px", type=int, help="input size, default the model's own")
    parser.add_argument("--backend", choices=("xnnpack",), default="xnnpack")
    args = parser.parse_args()

    config.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    embedder, module, example = load(args.model, args.px)
    path = export_executorch(module, example, export_path(embedder, args.backend))
    write_sidecar(path, embedder, backend=args.backend)
    median, p95 = latency_ms(path, example)
    print(f"{path.name}: median {median:.2f} ms, p95 {p95:.2f} ms over 50 runs (Mac CPU)")


if __name__ == "__main__":
    main()
