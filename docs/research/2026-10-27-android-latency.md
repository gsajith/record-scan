# Android latency for a ConvNeXt-Tiny-sized encoder, 2026-10-27

Question: can a ConvNeXt-Tiny-sized image encoder (28.6M params, 4.5 GMACs at 224 px) run at or under 30 ms p95 per frame on Android, and on which backend? Secondary: ViT-S/16 at 224 px and PE-Core S/16 at 384 px.

This builds on [on_device_stack.md](research_notes/On%20device%20album%20cover%20retrieval%202026/on_device_stack.md) section 1 and does not repeat its Qualcomm CLIP table. A link after a claim is its source. "Vendor" means the vendor measured its own hardware. "Est." means arithmetic done for this note, with the inputs shown.

## Short answer

XNNPACK fp32 on the CPU is not hopeless on the newest Snapdragon flagship, but it does not clear 30 ms p95 with any margin. The closest measured proxy, CLIP ViT-B/32's image encoder (same compute as ConvNeXt-T), runs at a 24.6 ms median on a Galaxy S26 Ultra (Snapdragon 8 Elite Gen 5). The same model takes 96.6 ms on a Pixel 10 (Tensor G5) and 113 ms on a Galaxy S20+, and older papers put ConvNeXt-T itself at 340 to 356 ms on Pixel 6 and Snapdragon 8 Gen 1 CPUs. So XNNPACK fails on mid-range phones and on Pixels. Only the NPU clears 30 ms with a wide margin: Qualcomm measures ConvNeXt-T at 1.3 to 2.1 ms (fp16) on Snapdragon 8 Gen 3 and newer, and 2.7 to 3.5 ms (w8a16) on the mid-range Snapdragon 7 Gen 4. The GPU sits between the two: 8.2 ms on a Galaxy S24 with LiteRT, but 67 ms on a Pixel 10 GPU for the CLIP proxy.

## Model sizes used for scaling

The task's "4.5 GFLOPs" is multiply-accumulates (MACs) in the ConvNeXt paper's convention. OpenCLIP counts FLOPs as 2x MACs and lists image-tower GFLOPs at 224 px as follows. [OpenCLIP model_profile.csv](https://github.com/mlfoundations/open_clip/blob/main/docs/model_profile.csv)

| Image tower | Params (M) | GFLOPs (OpenCLIP count) |
|---|---|---|
| ConvNeXt-T (`convnext_tiny`) | 28.61 | 8.91 |
| ViT-B/32 (`ViT-B-32`) | 87.85 | 8.82 |
| ViT-S/16 (`ViT-S-16`) | 21.81 | 9.20 |
| ViT-B/16 (`ViT-B-16`) | 86.19 | 35.13 |

So CLIP ViT-B/32 is a same-compute proxy for ConvNeXt-T and ViT-S/16 (within 4%), though it carries 3x the weights. ResNet-50 is 4.1 GMACs and DeiT-S 4.5 GMACs. [EfficientFormerV2, arXiv 2212.08059, Table 2](https://arxiv.org/abs/2212.08059)

## Measured numbers

### ConvNeXt-T and same-compute models on the CPU

| Model | Phone (chipset) | Runtime | Precision, threads | Statistic | ms | Source |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 image encoder, 224 | Galaxy S26 Ultra (Snapdragon SM8850, 8 Elite Gen 5) | ExecuTorch XNNPACK via react-native-executorch 0.11.0-dev | fp32, threads not stated | median of 20 after 3 warm-ups, end to end from a 512x512 frame (mean 25.6, `model.execute` 25.2) | 24.6 | [RNE benchmarks](https://github.com/software-mansion/react-native-executorch/blob/main/docs/docs/04-benchmarks/01-benchmarks.mdx), [data](https://github.com/software-mansion/react-native-executorch/blob/main/docs/docs/04-benchmarks/imported.json), [method](https://github.com/software-mansion/react-native-executorch/blob/main/apps/benchmarks/BENCHMARK_SPEC.md) |
| same | Pixel 10 (Tensor G5) | same | same | same | 96.6 | same |
| same | Galaxy S20+ (Exynos 990) | same | same | same | 113.1 | same |
| EfficientNet-V2-S | Galaxy S26 Ultra / Pixel 10 / S20+ | same | fp32 | same | 73.9 / 233.4 / 287.7 | same |
| EfficientNet-V2-S | same three | same | int8 | same | 53.5 / 135.3 / 196.3 | same |
| ConvNeXt-T, 224 | Pixel 6 (Tensor G1) | XNNPACK | precision and threads not stated | "averaged over multiple runs" | 340.5 | [EfficientFormerV2, Table 2](https://arxiv.org/abs/2212.08059) |
| ResNet-50, 224 | Pixel 6 | XNNPACK | same | same | 167.5 | same |
| DeiT-S, 224 | Pixel 6 | XNNPACK | same | same | 218.2 | same |
| ConvNeXt-T, 224 | Snapdragon 8 Gen 1 CPU (phone not named) | TFLite | fp32, threads not stated | not stated | 356 | [EfficientViT, arXiv 2205.14756v3, Table 4](https://arxiv.org/abs/2205.14756v3) |
| ConvNeXt-S, 224 (8.7 GMACs) | Pixel 6 / Galaxy S23 (Snapdragon 8 Gen 2) | TFLite XNNPACK | int8, 1 thread on the fastest core | median of 5 means | 314.9 / 243.9 | [MobileNetV4, arXiv 2404.10518, Table 6](https://arxiv.org/abs/2404.10518) |
| ResNet-50 | Galaxy S22 5G, Android 13 (SoC not reported; QNN jobs were compiled for SM8450, 8 Gen 1) | ExecuTorch XNNPACK | int8 (`xnnpack_q8`), default threadpool | mean of 50 runs after 10 warm-ups; median over 116 CI runs, Aug to Oct 2025 | 12.8 (range 11.7 to 17.4) | [PyTorch HUD ExecuTorch dashboard](https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fexecutorch), aggregated for this note |
| ConvNeXt-T, w8a8 | Snapdragon 7 Gen 4 QRD (Qualcomm reference phone) | TFLite, mixed: 220 layers CPU, 164 NPU, 10 GPU | int8 | not stated | 11.7 | Vendor. [qualcomm/ConvNext-Tiny](https://huggingface.co/qualcomm/ConvNext-Tiny), [perf.yaml](https://github.com/qualcomm/ai-hub-models/blob/v0.65.0/src/qai_hub_models/models/convnext_tiny/perf.yaml) |

The RNE row is the only 2026 phone CPU number at this compute. RNE runs the same ExecuTorch XNNPACK runtime as our export path.

The last row is not a clean CPU number. Qualcomm's own perf file shows the layers split three ways.

ExecuTorch's dashboard has no XNNPACK fp32 config for Android vision models, only `xnnpack_q8` and `qnn_q8`. Its Device Farm jobs were removed on 2025-11-21 "due to performance data being unreliable on non-rooted devices", so there is no Android data after October 2025. [PR #15433](https://github.com/pytorch/executorch/pull/15433)

### GPU

| Model | Phone (chipset) | Runtime | Precision | Statistic | ms | Source |
|---|---|---|---|---|---|---|
| ConvNeXt-T (torchvision) | Galaxy S24 (chipset not stated) | LiteRT GPU, "partially delegated" | not stated | not stated | 8.2 | [LiteRT GPU page](https://developers.google.com/edge/litert/next/gpu) |
| ResNet-50 (torchvision) | same | LiteRT GPU, "fully delegated" | not stated | not stated | 6.9 | same |
| Swin-T (torchvision) | same | LiteRT GPU, partial | not stated | not stated | 164.4 | same |
| CLIP ViT-B/32 image encoder | Galaxy S26 Ultra / Pixel 10 / Galaxy S20+ | ExecuTorch Vulkan via RNE | fp16 | median of 20 | 17.4 / 66.6 / 236.3 | [RNE benchmarks](https://github.com/software-mansion/react-native-executorch/blob/main/docs/docs/04-benchmarks/01-benchmarks.mdx) |
| CLIP ViT-B/32 image encoder | OPPO Find X7 Ultra (8 Gen 3, Adreno 750) / Find X7 (Dimensity 9300, Mali-G720) | Transformer-Lite's own GPU engine | fp16 | not stated | 19.2 / 29.5 | [Transformer-Lite, arXiv 2403.20041, Table 3](https://arxiv.org/abs/2403.20041) |
| ConvNeXt-S | Galaxy S23 (Adreno 740) / Pixel 7 (Mali-G710) | TFLite GPU | fp16 | median of 5 means | 18.5 / 45.2 | [MobileNetV4, Table 6](https://arxiv.org/abs/2404.10518) |
| ConvNeXt (28.6M, 4.5 GMACs, so T) | OnePlus, Snapdragon 8 Gen 2 (Adreno 740) | MNN 271, DNNFusion 109, SmartMem 33.4; TFLite GPU "not supported" | fp16 | mean of 50 | 33.4 to 271 | [SmartMem, arXiv 2404.13528, Table 8](https://arxiv.org/abs/2404.13528) |
| ConvNeXt-T | Qualcomm SA8775P (automotive, not a phone) | TFLite GPU | float | not stated | 25.0 | Vendor. [qualcomm/ConvNext-Tiny](https://huggingface.co/qualcomm/ConvNext-Tiny) |

Est. ConvNeXt-T on the S23 and Pixel 7 GPUs, scaled from ConvNeXt-S by MACs (4.47 / 8.7 = 0.51): 18.5 x 0.51 = 9.5 ms and 45.2 x 0.51 = 23 ms.

The SmartMem row shows how much the GPU runtime matters: the same model on the same GPU spans 33 to 271 ms.

On the S20+, Vulkan (236 ms) is twice as slow as XNNPACK (113 ms).

### NPU

These are Qualcomm's own numbers, measured on its hosted devices.
- AI Hub "float" runs in fp16 on the HTP, because `fp16_relaxed_precision` is on by default.
- AI Hub profiles at BURST power with GUI-app scheduling, so a real app may run slower. Qualcomm measured a 9.7% average slowdown from the CLI tool on MobileNetV2 across 65 devices.
- Inference jobs run up to 100 iterations. The docs do not say which statistic is reported.

[AI Hub FAQ](https://workbench.aihub.qualcomm.com/docs/hub/faq.html), [AI Hub API options](https://workbench.aihub.qualcomm.com/docs/hub/api.html). Device names come from [perf.yaml](https://github.com/qualcomm/ai-hub-models/blob/v0.65.0/src/qai_hub_models/models/convnext_tiny/perf.yaml).

| Chipset (phone) | ConvNeXt-T TFLite fp16 | ConvNeXt-T QNN w8a16 | ConvNeXt-T TFLite w8a8 | ResNet-50 TFLite fp16 | Source |
|---|---|---|---|---|---|
| 8 Elite Gen 5 (Galaxy S26) | 1.34 | 1.21 | 0.78 | 1.11 | [ConvNext-Tiny](https://huggingface.co/qualcomm/ConvNext-Tiny), [ResNet50](https://huggingface.co/qualcomm/ResNet50) |
| 8 Elite (Galaxy S25) | 1.61 | 1.51 | 0.90 | 1.26 | same |
| 8 Gen 3 (Galaxy S24) | 2.13 | 2.18 | 1.19 | 1.59 | same |
| 8 Gen 1 (Galaxy S22 5G) | 8.46 | 3.96 | 2.38 | 3.47 | same |
| 7 Gen 4 (QRD reference phone) | no float row | 3.52 | 11.7 (mixed CPU/NPU, see above) | no float row | same |

Other NPU rows:
- ConvNeXt-S on a Pixel 8 EdgeTPU, int8: 3.7 ms. On a Pixel 4 Hexagon DSP it "Failed": "Hybrid models and ConvNext are DSP-incompatible". [MobileNetV4, Table 6](https://arxiv.org/abs/2404.10518)
- ResNet-50 through ExecuTorch QNN int8 was disabled in CI. Inception-v4 ran at 2.4 ms on the S22 and 1.47 ms on the S24 (`qnn_q8`, mean of 50). [HUD dashboard](https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fexecutorch), [issue #7892](https://github.com/pytorch/executorch/issues/7892)

### ViT-S/16 class

No ViT-S/16 or DeiT-S number on a 2024 to 2026 Android phone was found on any backend. Qualcomm publishes no ViT-S card.

| Model | Phone (chipset) | Runtime | Precision, threads | Statistic | ms | Source |
|---|---|---|---|---|---|---|
| DeiT-S | Pixel 6 | XNNPACK | not stated | averaged | 218.2 | [EfficientFormerV2, Table 2](https://arxiv.org/abs/2212.08059) |
| DeiT-S | Pixel 7 (Tensor G2) | PyTorch Mobile CPU | not stated | mean of 200 after 50 warm-ups | 212 | [LTMP, arXiv 2307.10780, Table 6](https://arxiv.org/abs/2307.10780) |
| DeiT-S | Galaxy S20 (Snapdragon 865) | authors' own framework, 8 cores | not stated | mean of 100+ | 113 | [SPViT, arXiv 2112.13890](https://arxiv.org/abs/2112.13890) |
| ViT-B/16 (86.6M), 224 | 8 Gen 3 / 8 Elite / 8 Elite Gen 5 | TFLite on NPU | fp16 | not stated | 6.76 / 4.75 / 3.61 | Vendor. [qualcomm/VIT](https://huggingface.co/qualcomm/VIT) |
| ViT-B/16, 224 | 7 Gen 4 QRD | QNN on NPU | w8a16 / w8a8 | not stated | 14.2 / 18.8 | same |
| SigLIP2 B/16 image encoder, 224 | 8 Gen 3 / 8 Elite | TFLite on NPU | fp16 | not stated | 5.88 / 4.13 | Vendor. [qualcomm/SigLIP2](https://huggingface.co/qualcomm/SigLIP2) |
| SigLIP2 B/16 image encoder, 224 | 7 Gen 4 QRD | QNN on NPU | w8a16 | not stated | 7.73 | same |

On Qualcomm's ViT-B/16 card, w8a8 is slower than fp16 on every flagship. For example, 8 Gen 3 TFLite takes 9.01 ms in w8a8 against 6.76 ms in fp16. [qualcomm/VIT](https://huggingface.co/qualcomm/VIT)

Correction to on_device_stack.md: its "OpenAI-Clip" table (10 to 21 ms) times a 150M-param, 571 MB model with a 77-token text context, so it includes the text tower. Image-only ViT-B/16 on the same chips is 4 to 7 ms (VIT and SigLIP2 rows above). [qualcomm/OpenAI-Clip](https://huggingface.co/qualcomm/OpenAI-Clip)

## Estimates for our three candidates

These estimates are labelled as such; nobody has measured them.

ConvNeXt-T, XNNPACK fp32, from the CLIP ViT-B/32 proxy:
- Equal compute gives 24.6 x 8.91 / 8.82 = 24.9 ms median on the S26 Ultra.
- On the Pixel 6, ConvNeXt-T ran 340.5 / 218.2 = 1.56x slower than DeiT-S at the same MACs (EfficientFormerV2). If that ratio carries over from DeiT-S to ViT-B/32, the S26 Ultra figure becomes 24.6 x 1.56 = 38 ms.
- Range: 25 to 38 ms median on the S26 Ultra, 97 to 151 ms on the Pixel 10, 113 to 176 ms on the S20+.
- p95 sits above the median, and our 1.0.1 export leaves `native_layer_norm` outside the delegate (next section). The 30 ms p95 gate is therefore at best marginal on the best 2026 phone and out of reach elsewhere.

ViT-S/16 at 224, XNNPACK fp32: 24.6 x 9.20 / 8.82 = 25.7 ms median on the S26 Ultra. ViT-S/16 has 197 tokens against ViT-B/32's 50, so attention takes a larger share and this is likely low. Our Mac run timed DINOv3 ViT-S/16 at 1.5x ConvNeXt-T (65.9 vs 43.9 ms). [export spike](../results/2026-10-27-export-spike.md)

PE-Core S/16 at 384: our Mac XNNPACK run took 3.4x ViT-S/16 (226.9 / 65.9). Applied to 25.7 ms, that gives about 88 ms on the S26 Ultra CPU. This only works on an NPU or a fast GPU.

## Pixel 2 (Snapdragon 835) and scaling to current phones

| Model | Device | Runtime | Precision, threads | ms | Source |
|---|---|---|---|---|---|
| ResNet-v1-50 | Pixel 2 | TFLite (2018) | fp32 / int8, 1 large core | 874 / 440 | [Krishnamoorthi, arXiv 1806.08342, Table 7](https://arxiv.org/abs/1806.08342) |
| Inception-v3 at 346 px (7.67 GMACs) | Pixel 2 | TFLite CPU / NNAPI fp16 / NNAPI int8 | threads not stated, average | 746 / 891 / 467 | [AI Benchmark 2019, arXiv 1910.06663, Table 4](https://arxiv.org/abs/1910.06663) |
| MobileNet v1 | Pixel 2 / Pixel 3 | TFLite GPU (OpenGL) | fp16, mean of 100 | 18 / 12.5 | [TFLite GPU, arXiv 1907.01989, Table 5](https://arxiv.org/abs/1907.01989) |

Est. ConvNeXt-T on the Pixel 2 GPU:
- Pixel 3 GPU throughput from Google's 2019 to 2020 tables is 85 to 132 GMAC/s: ResNet-v2-101 7.83 / 0.092, Inception-v3 5.73 / 0.056, Inception-v4 12.28 / 0.093. [hosted_models.md @b2a6276](https://github.com/tensorflow/tensorflow/blob/b2a6276/tensorflow/lite/g3doc/guide/hosted_models.md), [benchmarks.md @788bd31](https://github.com/tensorflow/tensorflow/blob/788bd3123cf2c473c7a7d1a48fdf486027e8fa96/tensorflow/lite/g3doc/performance/benchmarks.md)
- ResNet-50 at that throughput takes 4.11 / 132 to 4.11 / 85, so 31 to 48 ms.
- The Pixel 2 GPU is 18 / 12.5 = 1.44x slower than the Pixel 3, which gives 45 to 70 ms.
- ConvNeXt-T costs 8.2 / 6.9 = 1.19x ResNet-50 on the S24 GPU, which gives 54 to 83 ms. This assumes the 2017 OpenGL path delegates the model at all.

Est. Pixel 2 CPU: 7.67 GMACs / 0.746 s = 10.3 GMAC/s, so ConvNeXt-T would take 4.47 / 10.3 = 434 ms on 2019 TFLite. Against the S26 Ultra proxy (4.4 GMACs / 0.0246 s = 179 GMAC/s), the newest flagship CPU is about 17x the Pixel 2. That ratio mixes models, runtimes and years, so it is a rough guide only.

The AI Benchmark ranking site gives CPU-F scores of 21 for the Pixel 2, 68 for the Pixel 9 Pro and 157 for the Galaxy S25 Ultra, a 3.2x to 7.5x range. The site is run by the paper authors and is not peer-reviewed. [ai-benchmark.com ranking](https://ai-benchmark.com/ranking.html)

The Pixel 2 has no usable NPU path: NNAPI float ran slower than its CPU (891 vs 746 ms, above), and Snapdragon 835 is not on LiteRT's NPU list. The Pixel 2 can check that the pipeline runs end to end. It cannot tell us whether a current phone meets 30 ms, because it misses by 2x to 15x on every backend.

## NPU and GPU paths from ExecuTorch on Android, October 2026

The latest releases are ExecuTorch v1.5.1 (2026-09-23) and react-native-executorch v0.11.0 (2026-10-27). [ExecuTorch releases](https://github.com/pytorch/executorch/releases), [RNE releases](https://github.com/software-mansion/react-native-executorch/releases)

Qualcomm QNN:
- The runtime ships prebuilt: `org.pytorch:executorch-android-qnn` is on Maven Central for 1.0.0 to 1.5.1, plus a separate `com.qualcomm.qti:qnn-runtime` dependency. [Maven Central](https://repo1.maven.org/maven2/org/pytorch/executorch-android-qnn/maven-metadata.xml), [using-executorch-android.md](https://github.com/pytorch/executorch/blob/v1.5.1/docs/source/using-executorch-android.md)
- Lowering does not work on a Mac. The 1.5.1 `manylinux_2_28_x86_64` wheel contains `executorch/backends/qualcomm/python/`, and the `macosx_14_0_arm64` wheel does not; we checked both wheels' file lists. The docs list Ubuntu 22.04 x64, CentOS Stream 9, Windows and WSL as hosts. [backends-qualcomm.md](https://github.com/pytorch/executorch/blob/v1.5.1/docs/source/backends-qualcomm.md)
- ConvNeXt: "Convnext model fails to lower on QNN" was closed on 2025-09-17. QNN support for `native_layer_norm` merged on 2026-06-17. [issue #14049](https://github.com/pytorch/executorch/issues/14049), [PR #18990](https://github.com/pytorch/executorch/pull/18990)

MediaTek: Dimensity 9300 and 9400 only, Linux host, needs the NeuroPilot SDK wheels, built from source, no Maven artifact. [backends-mediatek.md](https://github.com/pytorch/executorch/blob/v1.5.1/docs/source/backends-mediatek.md)

Samsung Exynos 2500 and 2600: "currently not available in any of ExecuTorch's published packages", so it is built from source. [samsung-overview.md](https://github.com/pytorch/executorch/blob/v1.5.1/docs/source/backends/samsung/samsung-overview.md)

Vulkan:
- The runtime is prebuilt as `org.pytorch:executorch-android-vulkan`, and its op table lists `native_layer_norm` and `gelu`. [vulkan-op-support-table.csv](https://github.com/pytorch/executorch/blob/v1.5.1/docs/source/backends/vulkan/vulkan-op-support-table.csv)
- "Convnext model fails at runtime on Vulkan" (convnext_small) is still open. [issue #14053](https://github.com/pytorch/executorch/issues/14053)
- ExecuTorch publishes no Vulkan latency of its own. RNE's Vulkan rows above are the only ExecuTorch Vulkan vision numbers found.

XNNPACK: the 1.5.1 op table lists `gelu` but no layer norm. So the `native_layer_norm` that fell outside the delegate in our 1.0.1 export still falls outside it. [op-support.csv](https://github.com/pytorch/executorch/blob/v1.5.1/docs/source/backends/xnnpack/op-support.csv)

react-native-executorch:
- Its documented backends are `"xnnpack"`, `"coreml"`, `"mlx"` and `"vulkan"`. On Android that means XNNPACK and Vulkan only. [08-native-libraries.md](https://github.com/software-mansion/react-native-executorch/blob/main/docs/docs/03-core-and-advanced/08-native-libraries.md)
- A legacy TypeScript type mentions `'qnn'`, but no QNN library ships.

LiteRT as the alternative NPU path:
- LiteRT lists Qualcomm, MediaTek, Samsung Exynos, Intel and Google Tensor NPUs, and needs API level 31 or higher. [LiteRT NPU page](https://developers.google.com/edge/litert/next/npu)
- Google Tensor is AOT only ("doesn't yet support on-device (JIT) compilation"). Its SDK page lists only Tensor G5 and G6, and access is a beta sign-up. Pixel 8 and 9 (G3, G4) therefore have no LiteRT NPU path. [Tensor SDK](https://developers.google.com/edge/litert/next/tensor-sdk)
- NNAPI is deprecated as of Android 15. [Android NNAPI](https://developer.android.com/ndk/guides/neuralnetworks)

## XNNPACK against LiteRT CPU on the same model

No pair was found with the same model, phone, precision and era. ExecuTorch's published XNNPACK runs are int8 only (`xnnpack_q8`) on Galaxy S22/S24 and a rooted Pixel 3. Google's published LiteRT CPU tables stop at Pixel 3 and Pixel 4 (2019 to 2020) and use other models. The closest thing is the same `torchvision` ResNet-50: 12.8 ms in ExecuTorch `xnnpack_q8` on a Galaxy S22, against 167.5 ms in XNNPACK (likely fp32, through EfficientFormerV2's setup) on a Pixel 6. Precision, phone and software year all differ, so this is not a fair comparison.

## Gaps

- No measured ConvNeXt-T on a 2025 or 2026 phone CPU with XNNPACK fp32 or int8. The 25 to 38 ms figure is an estimate from a ViT-B/32 proxy.
- No p95 anywhere. RNE, the ExecuTorch HUD, Qualcomm AI Hub and the papers report median, mean or an unstated statistic. Sustained-load (thermal) latency over minutes of camera use was not measured by any source.
- No mid-range phone CPU or GPU number on current software for any 4 to 5 GMAC model. The only mid-range figures are Qualcomm's NPU rows on the Snapdragon 7 Gen 4 reference phone, plus one mixed CPU/NPU row.
- No MediaTek Dimensity or Exynos number for ConvNeXt-T on any backend. The only Dimensity row is Transformer-Lite's CLIP on Dimensity 9300 GPU.
- No ViT-S/16 or DeiT-S on any NPU or GPU.
- No ExecuTorch Vulkan or QNN measurement of ConvNeXt anywhere. The open Vulkan runtime bug makes Vulkan unverified for this architecture.
- Which statistic Qualcomm AI Hub reports is not documented.
- The S24 in Google's LiteRT GPU table has no stated chipset; it ships with either Exynos 2400 or Snapdragon 8 Gen 3 depending on region.
- RNE's XNNPACK thread count for 0.11 was not found. The legacy code used half the performant cores plus one.
