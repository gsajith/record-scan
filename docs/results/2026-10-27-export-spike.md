# Export spike (plan Task 8, issue #27), 2026-10-27

Task 7 has not picked a shortlist, so the candidates are the three small DINOv3 models and
PE-Core S. All four convert to Core ML and to ExecuTorch XNNPACK. Each has at least one Core ML
path and the XNNPACK path at parity 0.99 or above, so none is dropped. Two findings shape
Phase B:

- **DINOv3 ViT-S and ViT-S+ break in plain Core ML fp16** (parity 0.16 and -0.04). Their
  attention scores reach 2.5 million (measured on ViT-S), and fp16 tops out at 65,504. Keeping attention's matmuls
  and softmax in fp32 (`fp16-attn32`) restores parity to 0.999995. Core ML then runs them on
  the GPU instead of the Neural Engine: 9.5 ms against 4.0 ms on this Mac. DINOv3 ConvNeXt
  (0.998) and PE-Core (0.9998) are fine in plain fp16.
- **XNNPACK lowering is fragmented.** DINOv3 ViT-S becomes 124 separate XNNPACK calls, with
  layer norms, the RoPE ops and the masking ops of PyTorch's fused attention running on
  ExecuTorch's slow portable kernels in between. Mac CPU times of 44 to 227 ms reflect that
  and a loaded machine, not a phone.

No phone was measured, and D13 (the oldest target phone) is still open, so the 30 ms p95 gate
cannot be applied yet.

## Provenance

- Code: 432e20b. The run started at 98448e7 with the export code uncommitted. That code was
  committed as 432e20b while the run went on, with no change to `pipeline/` in between.
- torch 2.9.1, coremltools 9.0, executorch 1.0.1, macOS 27.0.1, Apple M1 Max.
- Parity sample: 20 covers spread evenly over prefix 00 on the drive (`gallery_sample(20)` in
  `pipeline/export/parity.py`), from `0000172d-...` to `00f2970c-...`; sha1 of the 20 names,
  one per line: `00ea595de634`.
- Load average 15.8 at the start and 13.1 at the end, on 10 cores. Other sessions shared the
  Mac, so timings are noisy, CPU timings most of all. DINOv3 ConvNeXt's `fp16` and
  `fp16-attn32` packages place every op identically, yet timed 6.25 and 4.48 ms.

The commands, from the repo root, for each model (`px` is 224 for DINOv3 and 384 for PE-Core,
its only size):

```sh
python -m pipeline.export.to_coreml --model $m --px $px --precision fp32|fp16|fp16-attn32
python -m pipeline.export.to_executorch --model $m --px $px --backend xnnpack
python -m pipeline.export.parity --model $m --exported $EXPORTS_DIR/$m-$px-<variant>.<mlpackage|pte>
```

`to_coreml` times 50 predictions with every compute unit allowed, after five warm-up runs, and
counts the ops Core ML prefers to place on each device. `to_executorch` times 50 runs on the
Mac CPU through ExecuTorch's Python runtime. Parity is the mean cosine between the export and
the PyTorch embedder on CPU, one image at a time.

## Results

| Candidate | Path | Parity | Mac median ms | Mac p95 ms | Ops by device | Verdict |
|---|---|---|---|---|---|---|
| dinov3-vits16, 224 | Core ML fp32 | 1.000000 | 8.36 | 16.03 | GPU 526 | pass |
| | Core ML fp16 | 0.159275 | 4.02 | 4.27 | Neural Engine 522, CPU 5 | dropped: parity |
| | Core ML fp16-attn32 | 0.999995 | 9.46 | 12.83 | GPU 558, Neural Engine 17 | pass |
| | XNNPACK | 1.000000 | 65.94 | 97.98 | | pass |
| dinov3-vitsplus16, 224 | Core ML fp32 | 1.000000 | 8.47 | 15.11 | GPU 550 | pass |
| | Core ML fp16 | -0.042455 | 4.48 | 7.59 | Neural Engine 546, CPU 5 | dropped: parity |
| | Core ML fp16-attn32 | 0.999995 | 8.94 | 13.87 | GPU 599 | pass |
| | XNNPACK | 1.000000 | 83.65 | 127.08 | | pass |
| dinov3-convnext-tiny, 224 | Core ML fp32 | 1.000000 | 8.69 | 11.48 | GPU 187 | pass |
| | Core ML fp16 | 0.998397 | 6.25 | 11.34 | Neural Engine 154, GPU 33, CPU 1 | pass |
| | Core ML fp16-attn32 | 0.998397 | 4.48 | 5.90 | Neural Engine 154, GPU 33, CPU 1 | pass (no attention, same as fp16) |
| | XNNPACK | 1.000000 | 43.93 | 51.23 | | pass |
| pe-core-s16, 384 | Core ML fp32 | 1.000000 | 11.28 | 11.81 | GPU 546 | pass |
| | Core ML fp16 | 0.999824 | 16.81 | 17.02 | Neural Engine 546, CPU 1 | pass |
| | Core ML fp16-attn32 | 0.999972 | 11.91 | 12.26 | GPU 567, CPU 20, Neural Engine 11 | pass |
| | XNNPACK | 1.000000 | 226.92 | 325.41 | | pass |

Export sizes: fp32 packages and `.pte` files are 83 MB (ViT-S), 110 MB (ViT-S+), 106 MB
(ConvNeXt) and 91 MB (PE-Core); fp16 packages are half that.

## Why DINOv3 ViTs fail in fp16

In DINOv3 ViT-S, run in fp32 PyTorch over the 20 covers, the largest scaled attention score, `|q·k| / 8`, is
2,502,141, and the largest before scaling is 20,017,128. Float16's maximum is 65,504, so no
reordering of the math fits these scores in fp16. The largest activations elsewhere are
layer 0's query projection at about 2,000 and layer 6's residual at about 990, both within
range. PyTorch on CPU in fp16 still gives 0.999994, because its fp16 attention computes in
fp32 internally; PyTorch on MPS in fp16 gives NaN.

Tried and ruled out on DINOv3 ViT-S:

| Variant | Parity | Mac median ms | Ops by device |
|---|---|---|---|
| fp16, iOS 17 target | 0.159275 | 3.97 | Neural Engine 522, CPU 5 |
| fp16, iOS 18 target (native attention op) | 0.159275 | 3.90 | Neural Engine 522, CPU 5 |
| fp16, query scaled before `q·k` | 0.159275 | 3.97 | Neural Engine 522, CPU 5 |
| fp16, matmul and softmax in fp32 (`fp16-attn32`) | 0.999995 | 9.44 | GPU 558, Neural Engine 17 |

So on iPhone, DINOv3 ViTs run on the GPU or not at all, unless fine-tuning in Phase B tames
these scores. The Neural Engine stays open to ConvNeXt and PE-Core. Device timing has to tell
whether the GPU path still meets 30 ms on the oldest iPhone.

## Why XNNPACK is slow on the Mac

| Model | XNNPACK calls in the lowered graph | Ops left outside XNNPACK |
|---|---|---|
| dinov3-vits16 | 124 | 25 `native_layer_norm`, 24 `split_with_sizes_copy` and 53 `expand_copy` (RoPE), 24 each of `logical_not`, `any`, `where` and `full_like` (the fully masked row guard in fused attention), 125 `view_copy` |
| pe-core-s16 | 208 | 27 `native_layer_norm`, 48 `slice_copy`, 48 `unsqueeze_copy`, 38 `squeeze_copy`, the same attention guard ops, 130 `view_copy` |

Switching DINOv3 ViT-S to plain (eager) attention removes the guard ops but leaves 112 calls,
at 63 ms. Threads matter: ViT-S takes 163 ms on one thread, 65 on four and 67 on eight;
PE-Core takes 716, 281 and 242. Before Android timing means anything, the layer norms and
RoPE ops need to land inside XNNPACK, or the model needs another backend.

## Not measured

- **iPhone.** Needs Xcode's Core ML performance report with an iPhone connected: open each
  `.mlpackage` in `EXPORTS_DIR` and run the Performance tab on the device.
- **Android.** No phone was connected (`adb devices` listed none).
- **Qualcomm (QNN).** Not attempted on a phone. The pip wheel can't lower to it on the Mac:
  importing `executorch.backends.qualcomm.partition.qnn_partitioner.QnnPartitioner` fails with
  `ModuleNotFoundError: No module named 'executorch.backends.qualcomm.python'`. That module is
  built from source against Qualcomm's AI Engine Direct SDK.

## Warnings seen, quoted

- coremltools: `Torch version 2.9.1 has not been tested with coremltools. You may run into
  unexpected errors. Torch 2.7.0 is the most recent version that has been tested.` Every fp32
  and XNNPACK export still matched PyTorch at 1.000000.
- torchao: `Skipping import of cpp extensions due to incompatible torch version 2.9.1 for
  torchao version 0.14.0`. Only quantization would use them; this spike quantizes nothing.
- No conversion failed, so there are no unsupported-op or RoPE errors to quote. RoPE converts
  in both Core ML and ExecuTorch; under XNNPACK it runs outside the delegate.
