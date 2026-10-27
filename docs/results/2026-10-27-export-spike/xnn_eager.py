import os
from collections import Counter
from importlib.resources import files
from pathlib import Path
import torch
from PIL import Image
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from pipeline.export.model import load_exportable as load
from pipeline.export.parity import latency_ms, mean_cosine, gallery_sample

os.environ.setdefault("FLATC_EXECUTABLE", str(files("executorch.data.bin") / "flatc"))
emb, mod, ex = load("dinov3-vits16", 224)
emb.model.config._attn_implementation = "eager"
lowered = to_edge_transform_and_lower(torch.export.export(mod.eval(), (ex,)), partitioner=[XnnpackPartitioner()])
ops = Counter(str(n.target).rsplit(".", 1)[-1] for n in lowered.exported_program().graph.nodes if n.op == "call_function")
print("eager top-level ops:", dict(ops.most_common(8)), flush=True)
p = Path(os.environ["L"]) / "vits16-eager.pte"; p.write_bytes(lowered.to_executorch().buffer)
x = torch.stack([emb.preprocess(Image.open(q)) for q in gallery_sample(20)])
print("eager parity %.6f" % mean_cosine(mod, p, x), "latency median %.2f p95 %.2f ms" % latency_ms(p, ex), flush=True)
