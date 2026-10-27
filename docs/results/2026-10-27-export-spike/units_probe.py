"""Is PE-Core fp16 slower on the Neural Engine than fp32 on the GPU, or is that load noise?
Times each package with one compute-unit choice pinned, three times over, plus the ConvNeXt pair whose
op placement is identical."""
import time
import numpy as np, torch, coremltools as ct
from pipeline import config

def time_ms(path, px, units, runs=50):
    m = ct.models.MLModel(str(path), compute_units=units)
    name = m.get_spec().description.input[0].name
    x = {name: torch.zeros(1, 3, px, px).numpy()}
    for _ in range(5): m.predict(x)
    t = []
    for _ in range(runs):
        s = time.perf_counter(); m.predict(x); t.append((time.perf_counter() - s) * 1000)
    return np.median(t), np.percentile(t, 95)

X = config.EXPORTS_DIR
U = ct.ComputeUnit
cases = [("pe-core-s16-384-fp16", 384, U.CPU_AND_NE), ("pe-core-s16-384-fp16", 384, U.CPU_AND_GPU), ("pe-core-s16-384-fp16", 384, U.ALL),
         ("pe-core-s16-384-fp32", 384, U.CPU_AND_GPU),
         ("dinov3-convnext-tiny-224-fp16", 224, U.ALL), ("dinov3-convnext-tiny-224-fp16-attn32", 224, U.ALL)]
for rep in range(3):
    for stem, px, units in cases:
        med, p95 = time_ms(X / f"{stem}.mlpackage", px, units)
        print(f"round {rep}\t{stem}\t{units.name}\tmedian {med:.2f}\tp95 {p95:.2f}", flush=True)
