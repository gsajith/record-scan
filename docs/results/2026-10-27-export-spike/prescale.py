"""Scale the query before q @ k^T so the product stays in fp16 range; measure the scaled scores first."""
import os
from pathlib import Path
import torch, coremltools as ct
from PIL import Image
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from pipeline.export.model import load_exportable as load
from pipeline.export.parity import gallery_sample, mean_cosine, latency_ms
from pipeline.export.to_coreml import placement

peak = [0.0, 0.0]
def prescaled(module, query, key, value, attention_mask, scaling=None, **kw):
    q = query * scaling
    if not torch.compiler.is_compiling() and not isinstance(q, torch._subclasses.FakeTensor):
        peak[0] = max(peak[0], float((q @ key.transpose(-1, -2)).abs().max()))
        peak[1] = max(peak[1], float((query @ key.transpose(-1, -2)).abs().max()))
    return sdpa_attention_forward(module, q, key, value, attention_mask, scaling=1.0, **kw)
AttentionInterface.register("prescaled_sdpa", prescaled)

for name in ("dinov3-vits16", "dinov3-vitsplus16"):
    emb, mod, ex = load(name, 224)
    x = torch.stack([emb.preprocess(Image.open(p)) for p in gallery_sample(20)])
    with torch.no_grad():
        before = mod(x)
    emb.model.config._attn_implementation = "prescaled_sdpa"
    with torch.no_grad():
        after = mod(x)
    print(name, f"torch cos before/after prescale {torch.nn.functional.cosine_similarity(before, after).min():.6f}",
          f"peak |q.k| scaled {peak[0]:.0f}, unscaled {peak[1]:.0f}", flush=True)
    prog = torch.export.export(mod.eval(), (ex,)).run_decompositions({})
    m = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT16)
    p = Path(os.environ["L"]) / f"{name}-prescaled-fp16.mlpackage"; m.save(str(p))
    print(name, f"core ml fp16 prescaled parity {mean_cosine(mod, p, x):.6f}", "latency %.2f / %.2f ms" % latency_ms(p, ex), dict(placement(p).most_common()), flush=True)
