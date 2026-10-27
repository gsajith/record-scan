"""Try fixes for DINOv3 ViT-S in Core ML fp16: iOS18's native attention op, and fp32 attention matmuls."""
import torch, coremltools as ct
from PIL import Image
from pipeline.export.model import load_exportable as load
from pipeline.export.parity import gallery_sample, mean_cosine, latency_ms
from pipeline.export.to_coreml import placement

emb, mod, ex = load("dinov3-vits16", 224)
x = torch.stack([emb.preprocess(Image.open(p)) for p in gallery_sample(20)])
prog = torch.export.export(mod.eval(), (ex,)).run_decompositions({})
variants = {
    "ios17-fp16": dict(minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT16),
    "ios18-fp16": dict(minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16),
    "ios17-fp16-attn32": dict(minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in ("matmul", "softmax"))),
}
for tag, kw in variants.items():
    try:
        m = ct.convert(prog, convert_to="mlprogram", **kw)
        out = f"{__import__('os').environ['L']}/{tag}.mlpackage"
        m.save(out)
        from pathlib import Path
        p = Path(out)
        print(tag, f"parity {mean_cosine(mod, p, x):.6f}", "latency %.2f / %.2f ms" % latency_ms(p, ex), dict(placement(p).most_common()), flush=True)
    except Exception as e:
        print(tag, "FAILED:", type(e).__name__, str(e)[:300], flush=True)
