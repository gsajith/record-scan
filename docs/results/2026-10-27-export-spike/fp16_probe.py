"""Where does DINOv3 ViT-S break in float16? Max |activation| per layer in fp32, then torch fp16 vs fp32."""
import copy
import numpy as np, torch
from PIL import Image
from pipeline.export.model import load_exportable as load
from pipeline.export.parity import gallery_sample

for name in ("dinov3-vits16", "dinov3-convnext-tiny"):
    emb, mod, _ = load(name, 224)
    x = torch.stack([emb.preprocess(Image.open(p)) for p in gallery_sample(4)])
    peaks = {}
    hooks = [m.register_forward_hook(lambda m, i, o, n=n: peaks.__setitem__(n, max(peaks.get(n, 0), float(o.abs().max()) if torch.is_tensor(o) else 0)))
             for n, m in mod.named_modules() if n and isinstance(getattr(m, "forward", None), object) and not list(m.children())]
    with torch.no_grad():
        ref = mod(x)
    for h in hooks: h.remove()
    top = sorted(peaks.items(), key=lambda kv: -kv[1])[:6]
    print(name, "largest |activation| in fp32:", [(n, round(v)) for n, v in top])
    for dev in ("cpu", "mps"):
        m16 = copy.deepcopy(mod).half().to(dev)
        with torch.no_grad():
            out = m16(x.half().to(dev)).float().cpu()
        cos = torch.nn.functional.cosine_similarity(ref, out).mean().item()
        print(f"  torch fp16 on {dev}: mean cosine {cos:.6f}, nan={bool(out.isnan().any())}")
