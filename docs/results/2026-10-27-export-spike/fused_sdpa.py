"""Does Core ML's fused iOS 18 attention op fix DINOv3 ViT-S in fp16?

coremltools emits scaled_dot_product_attention only for an iOS 18 target when torch's call passes no
scale, and transformers always passes one (equal to the default), so fp16_fix.py's iOS 18 export got
plain matmuls. This registers an attention function that leaves the scale out.
"""
import os
from collections import Counter
from pathlib import Path
import torch, coremltools as ct
from PIL import Image
from transformers import AttentionInterface
from pipeline.export.model import load_exportable
from pipeline.export.parity import gallery_sample, mean_cosine, latency_ms
from pipeline.export.to_coreml import placement


def default_scale_sdpa(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    assert abs(scaling - query.shape[-1] ** -0.5) < 1e-9  # the default scale, so leaving it out is exact
    out = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register("default_scale_sdpa", default_scale_sdpa)
emb, mod, ex = load_exportable("dinov3-vits16", 224)
x = torch.stack([emb.preprocess(Image.open(p)) for p in gallery_sample(20)])
with torch.no_grad():
    before = mod(x)
emb.model.config._attn_implementation = "default_scale_sdpa"
with torch.no_grad():
    print("torch cosine to stock attention %.6f" % torch.nn.functional.cosine_similarity(before, mod(x)).min(), flush=True)
prog = torch.export.export(mod.eval(), (ex,)).run_decompositions({})
for prec in (ct.precision.FLOAT16, ct.precision.FLOAT32):
    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18, compute_precision=prec)
    p = Path(os.environ["L"]) / f"vits16-fused-{prec.name}.mlpackage"
    m.save(str(p))
    spec = m.get_spec().mlProgram.functions["main"]
    ops = Counter(op.type for op in next(iter(spec.block_specializations.values())).operations)
    print(prec.name, {k: ops[k] for k in ("scaled_dot_product_attention", "matmul", "softmax")},
          "parity %.6f" % mean_cosine(mod, p, x), "latency %.2f / %.2f ms" % latency_ms(p, ex), dict(placement(p).most_common()), flush=True)
