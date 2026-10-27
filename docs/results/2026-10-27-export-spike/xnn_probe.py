"""Is the XNNPACK export fully delegated, and how does thread count change its speed?"""
import os
from collections import Counter
from importlib.resources import files
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import _unsafe_reset_threadpool
from pipeline.export.model import load_exportable as load, export_path
from pipeline.export.parity import latency_ms

os.environ.setdefault("FLATC_EXECUTABLE", str(files("executorch.data.bin") / "flatc"))
for name, px in (("dinov3-vits16", 224), ("pe-core-s16", 384)):
    emb, mod, ex = load(name, px)
    lowered = to_edge_transform_and_lower(torch.export.export(mod.eval(), (ex,)), partitioner=[XnnpackPartitioner()])
    ops = Counter(str(n.target).split(".")[-2] if "." in str(n.target) else str(n.target)
                  for n in lowered.exported_program().graph.nodes if n.op == "call_function")
    print(name, "top-level ops:", dict(ops.most_common()), flush=True)
    path = export_path(emb, "xnnpack").with_suffix(".pte")
    for threads in (1, 4, 8):
        _unsafe_reset_threadpool(threads)
        print(f"  {threads} threads: median %.2f, p95 %.2f ms" % latency_ms(path, ex), flush=True)
