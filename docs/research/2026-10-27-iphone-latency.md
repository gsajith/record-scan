# iPhone latency for the encoder candidates, by chip generation (2026-10-27)

Question: which iPhones run a ConvNeXt-Tiny-sized encoder (28M params, 4.5 GFLOPs at 224 px) at or under 30 ms p95 per frame, and on which compute unit? The secondary candidates are a ViT-S/16-class model at 224 px and PE-Core S/16 at 384 px.

This note builds on `research_notes/On device album cover retrieval 2026/on_device_stack.md` (cited as "on_device_stack") and the Mac export spike in `docs/results/2026-10-27-export-spike.md` (cited as "export spike"). It does not repeat the MobileCLIP model-card table from on_device_stack.

Conventions. A link after a claim is its source. "Vendor figure" means the company that sells the hardware or library measured it. "Estimate" marks arithmetic done for this note, shown in full. No source found reports p95: papers give median, mean or minimum. Every p95 statement below is an estimate.

## Short answer

ConvNeXt-T itself measured 3.7 ms on an A14 (iPhone 12) Neural Engine, our DINOv3 ConvNeXt-T took 4.7 to 5.0 ms on an M1 Max Neural Engine that Apple's own figures put within about 20% of the A14's, and Apple's cross-generation figures show A15 to A18 Pro no slower, so ConvNeXt-Tiny on the Neural Engine clears 30 ms p95 with at least 2.5x headroom on A14 and newer (estimate: 4.4 to 6.1 ms median on A14). A13 (iPhone 11, SE 2nd gen, the oldest chip iOS 27 supports) has no published measurement of anything; a pessimistic ops/s-ratio estimate gives 10 to 13 ms median, still under 30 ms, but only a device run settles it. A plain ViT-S/16 also runs on the Neural Engine (DeiT-S 5.3 to 11.8 ms on A14), but DINOv3 ViT-S cannot, because its attention overflows fp16; its fp32-attention GPU path has no iPhone measurement, and two crude proxies put it at 23 to 61 ms on an A15, so it is the candidate most likely to miss 30 ms p95 on older phones. PE-Core S/16 at 384 px fits the Neural Engine in fp16 and estimates at 15 to 20 ms median on A14 to A17 Pro, so it clears 30 ms p95 only if the phone's tail stays within about 1.5x of median, which no source measures.

## 1. ConvNeXt-Tiny and ResNet-50 class models on named iPhones

All rows are Core ML on the Neural Engine in fp16 unless the row says otherwise.

| Model | Params, GFLOPs | iPhone (chip) | OS, toolchain | Latency | Statistic | Source |
|---|---|---|---|---|---|---|
| ConvNeXt-T, original layout (permute, channels-last LayerNorm, nn.Linear) | 28.6M, 4.5 | iPhone 12 Pro (A14) | iOS 16, coremltools 6.0 | 33.5 ms | median of 100 | [FastViT, arXiv 2303.14189, Table 14](https://arxiv.org/abs/2303.14189) (Apple authors) |
| ConvNeXt-T, channels-first LayerNorm and 1x1 convs (same math) | 29.0M, 4.5 | iPhone 12 Pro / Pro Max (A14) | iOS 16, coremltools 6.0 | 3.7 ms | median of 100 | [FastViT, Table 5 and Table 14](https://arxiv.org/abs/2303.14189) |
| ConvNeXt-T, original layout | 29.0M, 4.5 | iPhone 12 (A14) | iOS 16, CoreMLTools | 83.7 ms | mean | [EfficientFormerV2, arXiv 2212.08059, Table 2](https://arxiv.org/abs/2212.08059) |
| ConvNeXt-S | 50M, 8.7 | iPhone 13 (A15) | iOS 16.6.1, coremltools 7.1, MLProgram fp16 | 3.7 ms | median of means | [MobileNetV4, arXiv 2404.10518, Table 6 and appendix B](https://arxiv.org/abs/2404.10518) |
| ConvNeXt-S, channels-first | 50.0M, 8.7 | iPhone 12 Pro (A14) | iOS 16 | 5.4 ms | median | [FastViT, Table 14](https://arxiv.org/abs/2303.14189) |
| ResNet-50 | 25.5M, 4.1 | iPhone 12 (A14) | iOS 15, "all compute units" | 3.0 ms (CPU only: 29.4 ms) | mean of 1,000 | [EfficientFormer, arXiv 2206.01191, Table 1](https://arxiv.org/abs/2206.01191) |
| ResNet-50 | 25.5M, 4.1 | iPhone 12 (A14) | iOS 16 | 2.5 ms | mean | [EfficientFormerV2, Table 2](https://arxiv.org/abs/2212.08059) |
| ResNet-50 | 25.6M, 4.1 | iPhone 12 (A14) | Core ML | 2.69 ms | minimum round trip | [MobileOne, arXiv 2206.04040, Table 10](https://arxiv.org/abs/2206.04040) (Apple authors) |
| RSB-ResNet50 | 25.6M, 4.1 | iPhone 12 Pro Max (A14) | iOS 16, coremltools 6.0 | 2.6 ms | median of 100 | [FastViT, Table 5](https://arxiv.org/abs/2303.14189) |
| ResNet-50 fp16 | 25.6M, 4.1 | iPhone 14 Pro (A16) / iPhone 15 Pro (A17 Pro) | iOS 17, Xcode 15, compute units "all" | 1.52 / 1.38 ms | Xcode median | [coremltools quantization performance guide](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html) (vendor figure) |
| ResNet-50 via ExecuTorch Core ML fp16 | 25.6M, 4.1 | iPhone 15 Pro (A17 Pro) | ExecuTorch CI, AWS Device Farm, 2024 | 1.979 ms (XNNPACK Q8 on CPU: 37.021 ms) | mean | [ExecuTorch benchmark README at 6fa4233](https://github.com/pytorch/executorch/blob/6fa4233997b466c8c1a86e115267c6fabcccbe26/extension/benchmark/README.md) |
| DINOv3 ConvNeXt-T (ours), fp16 | 28M, 4.5 | M1 Max, not a phone | macOS 27.0.1, coremltools 9.0 | 4.7 to 5.0 ms; p95 5.90 ms in one run | median of 50 | [export spike](../results/2026-10-27-export-spike.md) |

What the rows say:

- A ConvNeXt-T-sized convnet runs in 2.5 to 3.7 ms on an A14 Neural Engine once its layout suits the engine. The same network in its original layout measured 33.5 ms (FastViT) and 83.7 ms (EfficientFormerV2) on the same chip generation. The fix in FastViT is a channels-first LayerNorm and 1x1 convolutions in place of `nn.Linear`, which removes the reshapes. [FastViT appendix B.1](https://arxiv.org/abs/2303.14189)
- Our DINOv3 ConvNeXt-T uses the slow layout. The Hugging Face block does `permute`, channels-last `LayerNorm`, two `nn.Linear` layers, then `permute` back (transformers 5.19.0, `models/dinov3_convnext/modeling_dinov3_convnext.py`, lines 86 to 127, in the project's `pipeline/.venv`). On the M1 Max it still timed 4.7 to 5.0 ms with 154 ops on the Neural Engine and 33 on the GPU ([export spike](../results/2026-10-27-export-spike.md)), so coremltools 9 handles it far better than coremltools 6 did. Whether those 33 GPU ops cost more on a phone GPU with 5 or 6 cores than on a 32-core M1 Max GPU ([Apple newsroom, M1 Pro and M1 Max](https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/)) is unmeasured.
- EfficientFormer reports that LayerNorm costs 10 to 20% of DeiT-S and PoolFormer-S24 latency on iPhone 12, and that reshapes, not attention, are what slow LeViT. [EfficientFormer, section 3](https://arxiv.org/abs/2206.01191)

## 2. ViT-S/16 and DeiT-S class models

| Model | Params, GFLOPs, input | iPhone (chip) | Compute unit, precision | Latency | Source |
|---|---|---|---|---|---|
| DeiT-S | 22.0M, 4.6, 224 | iPhone 12 Pro Max (A14), iOS 16 | Core ML, compute unit not stated | 5.3 ms median | [FastViT, Table 5](https://arxiv.org/abs/2303.14189) |
| DeiT-S | 22.5M, 4.5, 224 | iPhone 12 (A14), iOS 15 | "all compute units (NPU)" | 11.8 ms (CPU only: 41.0 ms) | [EfficientFormer, Table 1](https://arxiv.org/abs/2206.01191) |
| DeiT-S | 22.5M, 4.5, 224 | iPhone 12 (A14), iOS 16 | Neural Engine | 11.8 ms | [EfficientFormerV2, Table 2](https://arxiv.org/abs/2212.08059) |
| DeiT-tiny | 5.9M, 1.3, 224 | iPhone 12 (A14) | Core ML | 4.78 ms minimum round trip | [MobileOne, Table 9](https://arxiv.org/abs/2206.04040) |
| CLIP ViT-B/16 image encoder (plain ViT) | 86.2M, 17.5, 224 | iPhone 12 Pro Max (A14), iOS 17.0.3, coremltools 7.0 | Core ML | 11.5 ms | [MobileCLIP, arXiv 2311.17049, Table 7](https://arxiv.org/abs/2311.17049) (Apple authors) |
| Depth Anything V2 Small (ViT-S/14 encoder plus DPT head), fp16 | 24.8M, 518x396 | iPhone 12 Pro Max (A14), iOS 18.0 | dominant unit: Neural Engine | 31.10 ms | [apple/coreml-depth-anything-v2-small](https://huggingface.co/apple/coreml-depth-anything-v2-small) (vendor figure) |
| same | same | iPhone 15 Pro Max (A17 Pro) / iPhone 16 Pro (A18 Pro) | Neural Engine / compute units "all" | 33.90 / 26.21 ms | [same card](https://huggingface.co/apple/coreml-depth-anything-v2-small); [Apple Core ML models gallery](https://developer.apple.com/machine-learning/models/) (vendor figures) |
| CLIP ViT-B/32 image encoder via react-native-executorch | ~88M, 4.4, 224 | iPhone SE 3rd gen (A15) / iPhone 17 (A19) | Core ML fp16 | 5.71 / 3.06 ms forward (6.04 / 3.08 ms end to end, median) | [react-native-executorch benchmarks, imported.json](https://github.com/software-mansion/react-native-executorch/blob/a597acf69cba1f7c5ea7412d05b10ffb2782f363/docs/docs/04-benchmarks/imported.json) (library vendor's figures) |
| same | same | same | MLX int8 (Metal GPU) | 23.39 / 10.02 ms forward | same |
| same | same | same | XNNPACK fp32 (CPU) | 81.6 / 37.97 ms forward | same |
| DINOv3 ViT-S/16 (ours), fp16 | 22M, 224 | M1 Max, not a phone | Neural Engine 522 ops | 4.02 ms, but parity 0.16 (wrong output) | [export spike](../results/2026-10-27-export-spike.md) |
| DINOv3 ViT-S/16 (ours), fp16-attn32 / fp32 | same | M1 Max | GPU | 9.46 / 8.36 ms median | [export spike](../results/2026-10-27-export-spike.md) |

Do ViTs fall off the Neural Engine on iPhone?

- No source reports a plain ViT leaving the Neural Engine on iPhone. Apple lists the Neural Engine as the dominant compute unit for the ViT-S-based Depth Anything V2 on iPhone 12 Pro Max and 15 Pro Max. [apple/coreml-depth-anything-v2-small](https://huggingface.co/apple/coreml-depth-anything-v2-small)
- Our DINOv3 ViT-S ran on the Neural Engine too (522 ops) and gave wrong output, because its attention scores reach 2.5M against an fp16 maximum of 65,504. [export spike](../results/2026-10-27-export-spike.md) So the fall-off is ours: it comes from keeping attention in fp32.
- Core ML's typed-execution guide says the Neural Engine is the one compute unit barred when a model is typed float32: "Only the NE is barred as a compute unit when a model is typed with float 32 precision." [coremltools, Typed execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html) Any fp32 attention block therefore runs on GPU or CPU on every iPhone.
- An independent 2026 reverse-engineering study, measured on M1 and M5 silicon only, reports that the Neural Engine datapath has no fp32 margin: values above 65,504 saturate silently to infinity, and the multiply-accumulate output saturates at 32,768. It also reports that the engine's fused softmax subtracts the max first, so softmax itself does not overflow; the overflow risk sits in the `q.k` matmul. [Bryngelson, arXiv 2606.22283, section 3.7](https://arxiv.org/abs/2606.22283) (single author, not affiliated with Apple)
- Apple's two transformer-on-ANE write-ups give layout guidance (channels-first `(B, C, 1, S)` tensors, `nn.Linear` to `nn.Conv2d`, chunked per-head attention, split softmax) and say nothing about fp16 overflow in attention. [Apple ML Research, Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers); [Apple ML Research, Deploying Attention-Based Vision Transformers to Apple Neural Engine](https://machinelearning.apple.com/research/vision-transformers)

## 3. Generation curve

### Neural Engine specs from Apple

| Chip | First iPhone | Neural Engine as Apple states it | Source |
|---|---|---|---|
| A11 | iPhone 8 / X | 600 billion ops/s | [Apple newsroom, iPhone XS (2018)](https://www.apple.com/newsroom/2018/09/iphone-xs-and-iphone-xs-max-bring-the-best-and-biggest-displays-to-iphone/) |
| A12 | iPhone XS / XR | 8-core, 5 trillion ops/s; first Neural Engine open to Core ML | [same](https://www.apple.com/newsroom/2018/09/iphone-xs-and-iphone-xs-max-bring-the-best-and-biggest-displays-to-iphone/) |
| A13 | iPhone 11 | "faster Neural Engine"; no ops/s figure | [Apple newsroom, iPhone 11 (2019)](https://www.apple.com/newsroom/2019/09/apple-introduces-dual-camera-iphone-11/) |
| A14 | iPhone 12 | 16-core, 11 trillion ops/s | [Apple newsroom, iPhone 12 Pro (2020)](https://www.apple.com/newsroom/2020/10/apple-introduces-iphone-12-pro-and-iphone-12-pro-max-with-5g/) |
| A15 | iPhone 13 | 16-core, 15.8 trillion ops/s | [Apple newsroom, iPhone 13 Pro (2021)](https://www.apple.com/newsroom/2021/09/apple-unveils-iphone-13-pro-and-iphone-13-pro-max-more-pro-than-ever-before/) |
| A16 | iPhone 14 Pro | 16-core, "nearly 17 trillion" ops/s | [Apple newsroom, iPhone 14 Pro (2022)](https://www.apple.com/newsroom/2022/09/apple-debuts-iphone-14-pro-and-iphone-14-pro-max/) |
| A17 Pro | iPhone 15 Pro | "up to 2x faster"; the widely quoted 35 TOPS is not in the press release | [Apple newsroom, iPhone 15 Pro (2023)](https://www.apple.com/newsroom/2023/09/apple-unveils-iphone-15-pro-and-iphone-15-pro-max/) |
| A18 | iPhone 16 | 16-core, "up to 2x faster than A16" for ML models | [Apple newsroom, iPhone 16 (2024)](https://www.apple.com/newsroom/2024/09/apple-introduces-iphone-16-and-iphone-16-plus/) |
| A18 Pro | iPhone 16 Pro | 16-core, "faster and more efficient"; no figure | [Apple newsroom, iPhone 16 Pro (2024)](https://www.apple.com/newsroom/2024/09/apple-debuts-iphone-16-pro-and-iphone-16-pro-max/) |
| A19 Pro | iPhone 17 Pro | 16-core Neural Engine, plus Neural Accelerators in each of 6 GPU cores; no figure | [Apple newsroom, iPhone 17 Pro (2025)](https://www.apple.com/newsroom/2025/09/apple-unveils-iphone-17-pro-and-iphone-17-pro-max/) |
| A20 Pro | iPhone 18 Pro | "Dual 16-core" (32 cores), "double the AI processing power of A19 Pro"; 7-core GPU with Neural Accelerators | [Apple newsroom, iPhone 18 Pro (2026)](https://www.apple.com/newsroom/2026/09/apple-debuts-iphone-18-pro-and-iphone-18-pro-max/); [iPhone 18 Pro specs](https://www.apple.com/iphone-18-pro/specs/) |
| M1 / M1 Max (our Mac) | n/a | 16-core, 11 trillion ops/s; M1 Max GPU 32-core | [Apple newsroom, M1 (2020)](https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/); [M1 Pro and M1 Max (2021)](https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/) |

iOS 27 runs on iPhone 11 and later plus iPhone SE 2nd generation and later, so the oldest chip it supports is the A13. [Apple, iOS compatible devices](https://www.apple.com/os/ios/) Apple Intelligence on iOS 27 needs iPhone 15 Pro (A17 Pro) or any iPhone 16 or later. [same page](https://www.apple.com/os/ios/)

### Same model measured across generations

All are Apple's figures from its Core ML models gallery and Hugging Face model cards (vendor figures). Compute units "all"; the cards name the Neural Engine as the dominant unit.

| iPhone (chip) | FastViT-MA36 F16 (42.7M, 7.9 GMACs, 256 px) | DETR ResNet-50 seg. F16 | Depth Anything V2 Small F16 (ViT-S) |
|---|---|---|---|
| iPhone 12 Pro Max (A14) | 4.50 ms | 52 ms | 31.10 ms |
| iPhone 13 Pro Max (A15) | 4.47 ms | 51 ms | |
| iPhone 14 / 14 Plus (A15) | 4.56 ms (14 Plus) | 51 / 50 ms | |
| iPhone 15 Plus (A16) | 3.47 ms | 43 ms | |
| iPhone 15 Pro Max (A17 Pro) | 3.33 ms | 39 ms | 33.90 ms |
| iPhone 16 Pro (A18 Pro) | 2.78 ms | 34.32 ms | 26.21 ms |
| MacBook Pro M1 Max | 4 ms ("all") | 117 ms ("all") and 43 ms ("GPU") in the gallery; the HF card gives 43 ms with "CPU and Neural Engine" selected by hand | 33.48 ms ("all") and 32.78 ms ("GPU") in the gallery; 32.80 ms (Neural Engine) on the HF card |

Sources: [Apple Core ML models gallery](https://developer.apple.com/machine-learning/models/) for the iPhone 13 to 16 rows and M1 Max; [apple/coreml-FastViT-MA36](https://huggingface.co/apple/coreml-FastViT-MA36), [apple/coreml-detr-semantic-segmentation](https://huggingface.co/apple/coreml-detr-semantic-segmentation) and [apple/coreml-depth-anything-v2-small](https://huggingface.co/apple/coreml-depth-anything-v2-small) for the iPhone 12 Pro Max rows.

Two more cross-generation points:

- react-native-executorch, Core ML fp16, ExecuTorch forward time, iPhone SE 3rd gen (A15) then iPhone 17 (A19): CLIP ViT-B/32 image 5.71 then 3.06 ms; EfficientNetV2-S 3.88 then 3.5 ms. [react-native-executorch benchmarks](https://github.com/software-mansion/react-native-executorch/blob/a597acf69cba1f7c5ea7412d05b10ffb2782f363/docs/docs/04-benchmarks/imported.json) (library vendor's figures, release builds of 0.11.0-dev, September 2026, per [the benchmarks page](https://github.com/software-mansion/react-native-executorch/blob/a597acf69cba1f7c5ea7412d05b10ffb2782f363/docs/docs/04-benchmarks/01-benchmarks.mdx))
- Apple's gallery lists the 8-bit palettized DETR at 62.49 ms on iPad Pro 5th gen (M1) and 1,224 ms on iPad Pro 4th gen (A12Z). Apple gives no reason. It shows an A12-class chip can fall 20x off the curve for a given model. [Apple Core ML models gallery](https://developer.apple.com/machine-learning/models/)

What the curve shows:

- The M1 Max Neural Engine is within about 20% of an A14 iPhone's: A14 latency is 0.93 to 1.21x the M1 Max figure (Depth Anything V2 31.10 vs 33.48 ms; FastViT-MA36 4.50 vs 4 ms; DETR 52 vs 43 ms with CPU and Neural Engine). So the export spike's Neural Engine timings transfer to an iPhone 12 roughly one to one. The M1 Max GPU (32 cores) does not transfer: Apple lists 5 GPU cores for the A15 in iPhone 13 Pro, 6 for A17 Pro to A19 Pro, and 7 for A20 Pro (newsroom pages above).
- From A14 to A18 Pro, Apple's own models got 1.2 to 1.6x faster (FastViT-MA36 4.50 to 2.78 ms; DETR 52 to 34.32 ms; Depth Anything V2 31.10 to 26.21 ms). Over the same span Apple's marketing went from 11 trillion ops/s (A14) to "nearly 17 trillion" (A16) and then "up to 2x faster" (A17 Pro). Newer is not always faster: the attention-heavy Depth Anything V2 is slower on the A17 Pro (33.90 ms) than on the A14 (31.10 ms).
- No source in this search measured any model on an A12 or A13 iPhone.

### Which chips clear 30 ms p95 for ConvNeXt-T (estimate)

Inputs: measured A14 median 3.7 ms for channels-first ConvNeXt-T (FastViT); our DINOv3 ConvNeXt-T at 4.7 to 5.0 ms on the M1 Max Neural Engine (export spike); A14 latency at 0.93 to 1.21x the M1 Max's (table above). For p95, the export spike's worst p95/median ratio was 11.34/6.25 = 1.8 on a loaded Mac, so this note uses 2x as the p95 margin. That margin is an assumption, not a phone measurement.

| Chip | Median estimate | Arithmetic | p95 at 2x | Under 30 ms p95? |
|---|---|---|---|---|
| A14 | 4.4 to 6.1 ms | 4.7 x 0.93 = 4.4; 5.0 x 1.21 = 6.1 | 9 to 12 ms | Yes, with at least 2.5x headroom |
| A15, A16, A17 Pro, A18 Pro | up to 6.6 ms | Apple's gallery: these chips run its models at 0.62 to 1.09x of A14 latency; 6.1 x 1.09 = 6.6 | up to 13 ms | Yes |
| A19, A19 Pro, A20 Pro | below the A15 figure | react-native-executorch A19/A15 ratio 0.54 to 0.90 | below 13 ms | Yes (A20 Pro cold start is a separate problem, section 4) |
| A13 | 10 to 13 ms, as an upper bound | No A13 data. Pessimistic: assume latency scales with marketed ops/s and use the A12's 5 trillion as a floor for A13: 4.4 to 6.1 x (11 / 5) = 9.7 to 13.4 | 19 to 27 ms | Probably, but unverified |
| A12 | 10 to 13 ms if every op stays on the Neural Engine | same arithmetic | 19 to 27 ms | Unknown: the A12Z DETR row shows a 20x fall-off is possible. iOS 27 does not run on A12 phones. |

The ops/s ratio is a weak predictor in both directions: A14 to A15 is 1.44x in ops/s and 1.0x in FastViT-MA36 latency. Treat the A13 row as a sanity bound, not a forecast.

The same arithmetic for the secondary candidates:

- ViT-S/16 in plain fp16 on the Neural Engine (a model without DINOv3's overflow): measured 5.3 to 11.8 ms on A14 (DeiT-S rows). Under 30 ms p95 on A14 and newer with the 2x margin.
- DINOv3 ViT-S/16 on the GPU (fp16-attn32 or fp32): no iPhone GPU measurement exists. Proxy 1, core count: 9.5 ms on a 32-core M1 Max GPU x 32/5 = 61 ms on a 5-core phone GPU, or x 32/6 = 51 ms on 6 cores. That ignores clock and per-dispatch overhead, so treat it as pessimistic. Proxy 2, a similar-FLOPs ViT on the phone GPU: CLIP ViT-B/32 (4.4 GFLOPs vs ViT-S/16's 4.6) under MLX int8 runs 23.39 ms on A15 and 10.02 ms on A19. int8 MLX is not fp32 Core ML, so this proxy is optimistic. The two proxies put an A15 at 23 to 61 ms median. This is the candidate that most needs a phone run, and it likely fails 30 ms p95 on A13 to A15.
- PE-Core S/16 at 384 px, fp16 on the Neural Engine: 16.6 to 16.8 ms on the M1 Max Neural Engine (export spike) x 0.93 to 1.21 = 15 to 20 ms median on A14. The export spike's quieter PE-Core run had p95 17.02 against median 16.81, a 1.01x tail, but that was a Mac. At a 1.5x tail the A14 lands at 23 to 30 ms p95; at the 2x margin used above, 31 to 41 ms. Apple's only attention-heavy cross-generation row (Depth Anything V2) runs 1.09x the A14 time on A17 Pro and 0.84x on A18 Pro, so A15 to A17 Pro should look like A14, and A18 Pro gives 13 to 17 ms median. The Mac preferred its GPU for PE-Core (10.2 ms); a 5 or 6 core phone GPU is unlikely to repeat that, but no source measures it.

## 4. Cold start: first load and compile on the Neural Engine

How Core ML loads. On load, Core ML checks a cache for a device-specialized compile. A miss triggers an "uncached load" that segments the model across compute units and compiles each segment for its device. "For certain models, the uncached load can take a significant amount of time." The OS deletes the cache on a system update, when disk space runs low, or when the compiled model changes. The Core ML Instrument labels uncached loads "prepare and cache". [WWDC23, Improve Core ML integration with async prediction](https://developer.apple.com/videos/play/wwdc2023/10049/)

| Case | Device | Load | Compile | Prediction | Source |
|---|---|---|---|---|---|
| YOLOv3, compute units "all" (54 layers GPU, 32 Neural Engine) | an iPhone, model not named | about 400 ms median | about 940 ms | 22.19 ms | [WWDC22, Optimize your Core ML usage](https://developer.apple.com/videos/play/wwdc2022/10027/) |
| Xcode report in Apple's vision-transformer repo (594 ops, all on Neural Engine; model not named in the figure) | iPhone 15 Pro Max (A17 Pro), iOS 17.1 | 105.69 ms median | 151.24 ms median | 1.36 ms | [apple/ml-vision-transformers-ane, assets/device_measurements.png](https://github.com/apple/ml-vision-transformers-ane) |
| ResNet-50 via ExecuTorch Core ML fp16 | iPhone 15 Pro | 68.943 ms | | 1.979 ms | [ExecuTorch benchmark README at 6fa4233](https://github.com/pytorch/executorch/blob/6fa4233997b466c8c1a86e115267c6fabcccbe26/extension/benchmark/README.md) |
| CLIP ViT-B/32 via react-native-executorch Core ML fp16 | iPhone 17 (A19) / iPhone SE 3rd gen (A15) | 50 / 790 ms | | 3.06 / 5.71 ms | [react-native-executorch imported.json](https://github.com/software-mansion/react-native-executorch/blob/a597acf69cba1f7c5ea7412d05b10ffb2782f363/docs/docs/04-benchmarks/imported.json) (vendor figures; the page calls them "cold-start load times" but does not say whether the Core ML cache was cleared) |
| Four attention layers, 4,096 tokens, fp16, `.cpuAndNeuralEngine` | iPhone 18 Pro (A20 Pro) / iPhone 17 Pro, iOS 27.0.1 | 53.4 s / 0.8 s cold; 0.01 to 0.03 s cached | | 21.6 / 111.4 ms | [Apple Developer Forums thread 850058](https://developer.apple.com/forums/thread/850058), already summarized in on_device_stack section 1 |

Updates since on_device_stack. In thread 850058, other developers report first loads of over 60 s on iPhone 18 Pro against under 0.5 s on other iPhones. One sees an 11-minute first run of a Vision face-landmarks request on iPhone 18 Pro Max under iOS 27.2. An Apple DTS engineer replied that iOS 27.2 Beta 4 should improve load times "though it is not yet at parity with older devices". One developer measured a 120 s load falling to about 20 s on 27.2 beta 4. The thread also reports that the cache does not survive an app reinstall, and that `.cpuAndGPU` avoids the delay. [Apple Developer Forums thread 850058](https://developer.apple.com/forums/thread/850058) These are developer reports, not Apple measurements.

Core AI (new in iOS 27) offers ahead-of-time compilation. `xcrun coreai-build compile` turns a `.aimodel` into one `.aimodelc` per device architecture, and the app downloads the one matching `AIModel.deviceArchitectureName`. It "only compiles for devices that support Apple Intelligence, including iPhone or iPad with the A17 Pro chipset or later", and "the compiled asset still requires some specialization on the device". [Apple, Compiling Core AI models ahead of time](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time) `AIModel.specialize(contentsOf:options:)` lets the app pick when specialization happens, without reducing the work. [Apple, Managing model specialization and caching](https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching) One forum report on an M6 Mac found the precompiled Core AI path no faster than Core ML for an LSTM (16.3 s vs 12.8 s). [thread 850058](https://developer.apple.com/forums/thread/850058)

Published loads for models of this size range from 0.05 s to 0.8 s on A15 to A19, with uncached compiles of 0.15 s to 0.94 s where a source reports one. The A20 Pro regression is the outlier, and the reports so far involve attention and large activation maps; none tested a ConvNeXt.

## 5. ExecuTorch on iOS

- Core ML backend vs native Core ML. No published same-harness comparison was found. The closest pair is ResNet-50 fp16 on iPhone 15 Pro: 1.979 ms mean through ExecuTorch's benchmark app ([ExecuTorch README at 6fa4233](https://github.com/pytorch/executorch/blob/6fa4233997b466c8c1a86e115267c6fabcccbe26/extension/benchmark/README.md)) against 1.38 ms median in Xcode's performance tab ([coremltools guide](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)). That is 1.43x, but with different statistics, apps, iOS builds and model exports, so it does not isolate ExecuTorch overhead. react-native-executorch's numbers put ExecuTorch's own forward time within 0.02 ms of the end-to-end median for CLIP ViT-B/32 on iPhone 17 (3.06 vs 3.08 ms). [react-native-executorch imported.json](https://github.com/software-mansion/react-native-executorch/blob/a597acf69cba1f7c5ea7412d05b10ffb2782f363/docs/docs/04-benchmarks/imported.json)
- Precision control. ExecuTorch's Core ML `compute_precision` takes `FLOAT16` (default) or `FLOAT32` for the whole delegated graph, and the docs note "the ANE only supports FP16 precision". `compute_unit` takes ALL, CPU_ONLY, CPU_AND_GPU or CPU_AND_NE. `model_type=COMPILED_MODEL` compiles to `.mlmodelc` at export time, which the docs say "should improve the first time on-device model load time". [ExecuTorch Core ML partitioner docs](https://github.com/pytorch/executorch/blob/main/docs/source/backends/coreml/coreml-partitioner.md) The documented options have no per-op precision, so the export spike's `fp16-attn32` variant has no documented ExecuTorch route; DINOv3 ViT-S through ExecuTorch would go whole-model fp32, on GPU or CPU (inference from the docs).
- MPS backend. It does not matter any more. ExecuTorch deprecated it in 1.2 and deleted it on 2026-08-26; the named replacements are Core ML, plus a Metal backend for macOS GPU work. [pytorch/executorch PR 22181](https://github.com/pytorch/executorch/pull/22181)
- Published iOS benchmarks. ExecuTorch's AWS Device Farm benchmark jobs were removed on 2025-11-21 "due to performance data being unreliable on non-rooted devices" [pytorch/executorch PR 15433](https://github.com/pytorch/executorch/pull/15433), so the HUD dashboard ([hud.pytorch.org](https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fexecutorch)) gets no new iPhone runs. The dashboard sat behind a Vercel bot check during this search and could not be read.
- react-native-executorch. Yes, it exposes Core ML on iOS. v0.11.0 (released 2026-10-27) ships XNNPACK, Core ML, MLX and Vulkan as separately downloaded backends, selected with a `backends` list in the app config. Core ML is "Supported on iOS only", adds 0.4 MB on iOS, and the `imageEmbeddings` feature works with xnnpack, coreml, mlx and vulkan. [react-native-executorch native libraries doc](https://github.com/software-mansion/react-native-executorch/blob/a597acf69cba1f7c5ea7412d05b10ffb2782f363/docs/docs/03-core-and-advanced/08-native-libraries.md); [v0.11.0 release](https://github.com/software-mansion/react-native-executorch/releases/tag/v0.11.0) Across the 131 iPhone model pairs it publishes, Core ML fp16 forward time is 3.0x to 153x faster than XNNPACK fp32, with a median of 19x (computed for this note from the published rows; CLIP ViT-B/32 on A15 is 5.71 vs 81.6 ms). [imported.json](https://github.com/software-mansion/react-native-executorch/blob/a597acf69cba1f7c5ea7412d05b10ffb2782f363/docs/docs/04-benchmarks/imported.json)

## Gaps

- No measurement of any model on an A12 or A13 iPhone (XS, XR, 11, SE 2nd gen). The A13 verdict, and with it the choice of oldest supported iPhone, rests on the estimate in section 3.
- No iPhone GPU (Core ML `.cpuAndGPU`) latency for any ViT-S-class model. The DINOv3 ViT-S path is the least certain number in this note: 23 to 61 ms on A15 by two proxies that disagree by 2.6x.
- No p95 or tail latency on any iPhone from any source. Papers report median, mean or minimum, mostly with other apps closed.
- No published iPhone run of ConvNeXt-T in the original Hugging Face layout with a current toolchain. The 33.5 ms and 83.7 ms figures are from iOS 16 and coremltools 6. Our Mac result says coremltools 9 handles that layout well, but the 33 ops it placed on the M1 Max GPU are unmeasured on a phone GPU.
- No independent (non-Apple) cross-generation Neural Engine numbers. The A14 to A18 Pro curve is Apple's own; react-native-executorch's A15 and A19 rows come from the library's vendor.
- No image-encoder measurement on A19 Pro or A20 Pro. A20 Pro data is limited to forum load-time reports and one attention microbenchmark.
- Apple gives no Neural Engine ops/s figure for A13, A17 Pro (in the press release), A18, A18 Pro, A19, A19 Pro or A20 Pro.
- No same-harness comparison of ExecuTorch's Core ML backend against native Core ML.
- No sustained (thermal) throughput at 5 to 10 fps for any model on any iPhone.
- Cheapest way to close the first three gaps: run Xcode's Core ML performance report on an A13 or A14 iPhone and an A15 iPhone for the four `.mlpackage` exports, which the export spike already lists as the next step.
