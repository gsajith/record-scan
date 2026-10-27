"""Export an embedder to a Core ML package and time it on this Mac.

    python -m pipeline.export.to_coreml --model dinov3-vits16 --px 224 [--precision fp32|fp16|fp16-attn32]

Writes EXPORTS_DIR/<model>-<px>-<precision>.mlpackage, then prints the median and p95 of 50
predictions with every compute unit allowed, and how many ops Core ML places on the Neural
Engine, the GPU and the CPU.
"""

import argparse
from collections import Counter
from pathlib import Path

import torch

from pipeline import config
from pipeline.export.model import export_path, load_exportable, write_sidecar
from pipeline.export.parity import latency_ms
from pipeline.retrieval.embedders import EMBEDDER_NAMES

# fp16 runs on the Neural Engine. DINOv3 ViTs' attention scores reach 2.5 million, past float16's
# 65,504, so fp16-attn32 keeps attention's matmuls and softmax in fp32, which moves them to the GPU.
PRECISIONS = ("fp32", "fp16", "fp16-attn32")


def export_coreml(module: torch.nn.Module, example: torch.Tensor, out: Path, precision: str = "fp32") -> Path:
    """Convert through torch.export, the same front end ExecuTorch uses, so both see one graph."""
    import coremltools as ct  # deferred: a slow import

    # coremltools converts only the ATen dialect, so lower the export's training dialect first
    program = torch.export.export(module.eval(), (example,)).run_decompositions({})
    compute_precision = {
        "fp32": ct.precision.FLOAT32,
        "fp16": ct.precision.FLOAT16,
        "fp16-attn32": ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in ("matmul", "softmax")),
    }[precision]
    mlmodel = ct.convert(
        program,
        convert_to="mlprogram",
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS17,
    )
    path = out.with_suffix(".mlpackage")
    mlmodel.save(str(path))
    return path


def placement(path: Path) -> Counter[str]:
    """Ops per compute device Core ML prefers, such as {"NeuralEngine": 410, "CPU": 3}."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    # The compiled copy lives in a temp folder that goes away with the MLModel, so keep it alive
    mlmodel = ct.models.MLModel(str(path))
    plan = MLComputePlan.load_from_path(path=mlmodel.get_compiled_model_path(), compute_units=ct.ComputeUnit.ALL)
    counts: Counter[str] = Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is not None:  # constants have no device
            counts[type(usage.preferred_compute_device).__name__.removeprefix("ML").removesuffix("ComputeDevice")] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=EMBEDDER_NAMES)
    parser.add_argument("--px", type=int, help="input size, default the model's own")
    parser.add_argument("--precision", choices=PRECISIONS, default="fp32")
    args = parser.parse_args()

    config.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    embedder, module, example = load_exportable(args.model, args.px)
    path = export_coreml(module, example, export_path(embedder, args.precision), args.precision)
    write_sidecar(path, embedder, precision=args.precision)
    median, p95 = latency_ms(path, example)
    print(f"{path.name}: median {median:.2f} ms, p95 {p95:.2f} ms over 50 predictions (compute units ALL)")
    print("ops by preferred device:", dict(placement(path).most_common()))


if __name__ == "__main__":
    main()
