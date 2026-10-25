# Image embedding backbones for on-device album cover retrieval (state as of October 2026)

Baseline the project already knew (pre November 2025): open_clip ViT-B/32 with OpenAI weights, 512-d, frozen, hnswlib over 3.13M Cover Art Archive fronts. DINOv2 (2023), SigLIP 2 (Feb 2025), Perception Encoder (Apr 2025, T/S added Jul 2025), MobileCLIP2 (Aug 2025), DINOv3 (Aug 2025) and TIPSv1 (ICLR 2025) all predate November 2025 and count as "already knowable" baseline. Genuinely new since then: DINOv3 ConvNeXt distillation code (2025-11-20), TIPSv2 (CVPR 2026), PE-AV audio models (Dec 2025, not relevant), DINOv3 CHMv2 and FINO metadata-guided training (2026, not relevant to retrieval).

## Which encoders improve on DINOv2 and CLIP ViT-B/32 for instance retrieval, and which exist at 5 to 90M params?

### Takeaway
DINOv3 is the clear instance-retrieval leader at every size, and its distilled ViT-S/16 (21M), ViT-S+/16 (29M), ViT-B/16 (86M) and ConvNeXt T/S/B (29/50/89M) all fit the mobile range. Vision-language encoders (SigLIP 2, PE-Core) are much weaker on instance retrieval when frozen; MobileCLIP2 is the only family designed and benchmarked for phone latency, but nobody has published instance-retrieval numbers for it. No major new general-purpose vision backbone aimed at instance retrieval appeared between November 2025 and October 2026; TIPSv2 is the main new release and its smallest size is B/14.

### Cited findings
Model inventory (params are image encoder only unless noted):

| Family | Mobile-range sizes | Params | Release | Source |
|---|---|---|---|---|
| DINOv3 ViT (LVD-1689M, distilled) | S/16, S+/16, B/16 | 21M, 29M, 86M | Aug 2025 | [dinov3 README](https://github.com/facebookresearch/dinov3) |
| DINOv3 ViT larger | L/16, H+/16, 7B/16 | 300M, 840M, 6,716M | Aug 2025 | [dinov3 README](https://github.com/facebookresearch/dinov3) |
| DINOv3 ConvNeXt (distilled) | Tiny, Small, Base | 29M, 50M, 89M (Large 198M) | Aug 2025, distillation code 2025-11-20 | [dinov3 README](https://github.com/facebookresearch/dinov3) |
| MobileCLIP2 | S0, S2, B | 11.4M, 35.7M, 86.3M (text 63.4M) | 2025-08-28 | [ml-mobileclip README](https://github.com/apple/ml-mobileclip) |
| MobileCLIP (v1) | S0, S1, S2, B | 11.4M, 21.5M, 35.7M, 86.3M | 2024-06 | [ml-mobileclip README](https://github.com/apple/ml-mobileclip) |
| Perception Encoder PE-Core | T/16, S/16 (384 px), B/16 (224 px) | not stated in README | Apr 2025, T/S added 2025-07-11 | [perception_models README](https://github.com/facebookresearch/perception_models) |
| PE-Spatial (distilled from G) | T/16, S/16, B/16 (512 px) | not stated | 2025 | [perception_models README](https://github.com/facebookresearch/perception_models) |
| SigLIP 2 | B/16, B/32 (smallest) | ~86M vision; HF card lists 0.4B total for base-patch16-224 incl. text tower | Feb 2025 | [HF siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) |
| TIPSv1 | S/14-HR, B/14-HR | 21.6M, 85.7M | ICLR 2025 | [google-deepmind/tips](https://github.com/google-deepmind/tips) |
| TIPSv2 | B/14 (smallest) | 85.7M (text 109.6M) | CVPR 2026 | [google-deepmind/tips](https://github.com/google-deepmind/tips); [TIPSv2 arXiv 2604.12012](https://arxiv.org/pdf/2604.12012) |

- DINOv3 ViT-S/16 specs: embedding dim 384, patch 16, 4 register tokens, 6 heads, RoPE. [Lightweight distillation of SAM 3 and DINOv3, arXiv 2604.27128](https://arxiv.org/pdf/2604.27128) (via search snippet). ILIAS lists DINOv3-B at 768-d and DINOv3-L at 1024-d. [ILIAS project page](https://vrg.fel.cvut.cz/ilias/)
- In TIPSv2 only the ViT-g is pretrained; B, L and SO-400m are distilled from it. [TIPSv2 CVPR 2026 supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Cao_TIPSv2_Advancing_Vision-Language_CVPR_2026_supplemental.pdf) (via search snippet)
- The TIPSv2 smallest checkpoint is B/14; the S/14 size exists only for TIPSv1. [google-deepmind/tips](https://github.com/google-deepmind/tips)
- DINOv3 repo news after November 2025 is limited to CHMv2 canopy-height model (2026-03-10) and FINO metadata-guided training recipes (2026-06-12); no new general backbones. [dinov3 README](https://github.com/facebookresearch/dinov3)
- Perception Encoder's only post-November-2025 release is PE-AV / PE-A-Frame audio-visual models (2025-12-16). [perception_models README](https://github.com/facebookresearch/perception_models)
- MobileCLIP2 (2025-08-28) is still the newest family in Apple's repo. [ml-mobileclip README](https://github.com/apple/ml-mobileclip)
- A search for 2026 instance-retrieval backbones found only general multimodal embedding work (e.g. Ovis-Embedding, ObjEmbed arXiv 2602.01753), not Oxford/Paris-focused backbones. [ObjEmbed](https://www.alphaxiv.org/abs/2602.01753)

### Inferences
- For a pure image-to-image task, the text tower is dead weight, so the CLIP-style models' "params" that matter are image-only; MobileCLIP2-S0 at 11.4M is the smallest serious encoder in this list.
- DINOv3 ConvNeXt variants are likely easier to run on the Apple Neural Engine and LiteRT delegates than ViTs with RoPE, but nobody has published their retrieval accuracy (see next section).

### Gaps
- PE-Core T/16 and S/16 parameter counts and embedding dims are not on the README; the PE paper would have them.
- I found no 2026 Apple vision encoder newer than MobileCLIP2 / FastViTHD aimed at embeddings.
- SigLIP 2 B/16 exact vision-tower parameter count and file size are not stated on the HF card (only 0.4B total).

## How do they compare on instance-retrieval benchmarks?

### Takeaway
On Revisited Oxford-Hard, DINOv3 ViT-B/16 scores 58.5 mAP versus 20.2 for SigLIP 2 B and PE-Core B, and DINOv3 ViT-S/16 (21M) at 49.5 beats every VLM up to giant size. On the 2025 ILIAS benchmark, frozen OpenAI CLIP ViT-B scores 5.9 mAP@1k on mini-ILIAS against 26.4 for DINOv3-B, so the project's current backbone sits near the bottom of modern options. A light linear adaptation lifts VLMs a lot (PE-L 27.1 to 39.6), more than it lifts DINOv3.

### Cited findings
DINOv3 paper, Table 9 (instance recognition, largest models; Oxford/Paris Hard mAP, Met GAP, AmsterTime mAP). Model-to-row mapping was reconstructed by the fetch tool from table ordering; verify against the PDF. [DINOv3 arXiv 2508.10104](https://arxiv.org/html/2508.10104)

| Model | Oxford-H | Paris-H | Met (GAP) | AmsterTime |
|---|---|---|---|---|
| DINOv3 7B/16 | 60.7 | 87.1 | 55.4 | 56.5 |
| DINOv2 g/14 | 58.2 | 84.6 | 44.6 | 48.9 |
| AM-RADIOv2.5 g/14 | 47.5 | 85.7 | 30.5 | 23.1 |
| PE-Core G/14 | 32.7 | 68.9 | 10.6 | 23.1 |
| Web-DINO 7B/14 | 31.2 | 80.3 | 35.2 | 30.6 |
| AIMv2 3B/14 | 28.8 | 71.4 | 29.5 | 14.6 |
| EVA-CLIP 18B/14 | 27.1 | 65.6 | 0.5 | 18.9 |
| SigLIP 2 g/16 | 25.1 | 60.9 | 13.9 | 15.5 |
| Franca g/14 | 14.3 | 51.6 | 27.2 | 21.1 |

DINOv3 paper, Table 14 (distilled family, Oxford-Hard mAP). [DINOv3 arXiv 2508.10104](https://arxiv.org/html/2508.10104)

| Size | DINOv3 | DINOv2 | SigLIP 2 | PE-Core |
|---|---|---|---|---|
| S (21M) | 49.5 | 39.5 | n/a | n/a |
| S+ (29M) | 50.0 | n/a | n/a | n/a |
| B (86M) | 58.5 | 51.0 | 20.2 | 20.2 |
| L (300M) | 63.1 | 55.7 | 21.4 | 25.6 |
| SO400m | n/a | n/a | 23.0 | n/a |
| H+ (840M) | 64.5 | n/a | n/a | n/a |

- DINOv3 ConvNeXt table (Table 15) reports no instance-retrieval column, only IN-ReaL, IN-R, ObjectNet, ADE20k and NYU. [DINOv3 arXiv 2508.10104](https://arxiv.org/html/2508.10104)

ILIAS (CVPR 2025; 1,000 object instances, 1,232 queries, 100M YFCC distractors), image-to-image mAP@1k. "5M" is mini-ILIAS, "100M" is full. [ILIAS project page](https://vrg.fel.cvut.cz/ilias/); [ILIAS paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Kordopatis-Zilos_ILIAS_Instance-Level_Image_retrieval_At_Scale_CVPR_2025_paper.pdf)

| Model | Dim | Test res | Frozen 5M | Frozen 100M | Lin. adapt 5M | Lin. adapt 100M |
|---|---|---|---|---|---|---|
| OAI-CLIP-B (ViT-B/16) | 512 | 384 | 5.9 | 4.2 | 10.7 | 7.9 |
| OAI-CLIP-L | 768 | 384 | 9.0 | 7.0 | 15.8 | 11.9 |
| DINOv2-B | 768 | 724 | 14.3 | 11.5 | 15.0 | 12.1 |
| DINOv2-L | 1024 | 724 | 18.5 | 15.3 | 18.8 | 15.3 |
| DINOv3-B | 768 | 768 | 26.4 | 22.0 | 26.4 | 22.5 |
| DINOv3-L | 1024 | 768 | 31.1 | 26.5 | 32.9 | 28.3 |
| SigLIP2-B@384 | 768 | 512 | 18.4 | 15.0 | 27.5 | 22.6 |
| SigLIP2-B@512 | 768 | 724 | 18.6 | 15.4 | 28.6 | 23.5 |
| SigLIP2-L@512 | 1024 | 724 | 25.3 | 20.8 | 37.3 | 31.3 |
| PE-B | 1024 | 384 | 20.2 | 9.7 | 20.2 | 16.1 |
| PE-L@336 | 1024 | 512 | 27.1 | 22.0 | 39.6 | 33.4 |
| UDON-CLIP (ViT-B, UnED fine-tuned) | 768 | 384 | 8.3 | 5.9 | 9.2 | 6.7 |
| UNIC-L | 1024 | 512 | 11.4 | 8.9 | 15.3 | 11.7 |

- Best ILIAS full-scale result uses local-descriptor re-ranking: AMES on top of adapted SigLIP2-L@512 reaches 38.4 mAP@1k at 100M. [ILIAS project page](https://vrg.fel.cvut.cz/ilias/)
- ILIAS authors: local descriptors in re-ranking remain a key ingredient, especially with severe background clutter; a linear adaptation layer trained with multi-domain class supervision helps, especially for VLMs. [ILIAS paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Kordopatis-Zilos_ILIAS_Instance-Level_Image_retrieval_At_Scale_CVPR_2025_paper.pdf)
- A December 2025 patch-wise retrieval paper reports, on mini-ILIAS mAP@1k, DINOv3 global features 0.218, SigLIP linear-adapted 0.405, and DINOv3 with its finest local re-ranking 0.430. These global numbers do not match the ILIAS page's 26.4 for DINOv3-B, so the configurations differ. [Patch-wise Retrieval, arXiv 2512.12610](https://arxiv.org/pdf/2512.12610) (via search snippet)
- ILIAS does not list MobileCLIP, TIPS, or any 2026 model. [ILIAS project page](https://vrg.fel.cvut.cz/ilias/)
- TIPSv1 reports "UNED-KNN" 57.7 to 71.5 across sizes and ImageNet k-NN; TIPSv2 reports ImageNet k-NN 79.8 to 83.7; neither reports Oxford/Paris or ILIAS. [google-deepmind/tips](https://github.com/google-deepmind/tips)
- Product retrieval, frozen vs top-tuned family averages (likely mMP@5): SOP text-image 0.860 frozen / 0.882 top-tuned, SSL 0.834 / 0.866; Products-10k text-image 0.581 / 0.641, SSL 0.428 / 0.465. [Benchmarking Image Embeddings for E-Commerce, arXiv 2504.07567](https://arxiv.org/html/2504.07567)
- MobileCLIP2 reports only zero-shot classification: IN-1k top-1 71.5 (S0), 77.2 (S2), 79.4 (B); 38-dataset average 59.7, 64.1, 65.8. [ml-mobileclip README](https://github.com/apple/ml-mobileclip)

### Inferences
- The VLM vs SSL gap flips on product data: text-image models beat SSL on Products-10k (0.581 vs 0.428). Album covers sit between landmarks and products: flat, planar, text-heavy artwork. Text and logo content may favor VLMs; fine texture and layout matching favors DINO. This is exactly why the first evaluation must run on the project's own phone-photo queries.
- ILIAS and Oxford-H are much harder than album-cover matching (3D viewpoint change, clutter, tiny objects). A frozen CLIP ViT-B/32 working "OK" on covers is consistent with these numbers; the ranking between backbones is the transferable signal, not the absolute mAP.
- ILIAS evaluates DINOv3 at 768 px. Retrieval at 224 px on a phone will be lower; resolution sensitivity needs to be measured.
- PE-B drops from 20.2 to 9.7 when going from 5M to 100M distractors, a much steeper fall than DINOv3-B (26.4 to 22.0). At a 3.13M gallery, distractor robustness matters.

### Gaps
- No published Oxford/Paris or ILIAS numbers for DINOv3 ConvNeXt variants, MobileCLIP/MobileCLIP2, PE-Core T/S, TIPSv2, or SigLIP 2 B/32.
- No GLDv2 or Google Universal Image Embedding challenge numbers found for these 2025 to 2026 models.
- No published album-cover or artwork-from-phone-photo retrieval benchmark found. The closest artwork benchmark is the Met (DINOv3 7B 55.4 GAP vs DINOv2 g 44.6 vs SigLIP 2 g 13.9).

## Licenses, and do they allow use in a free consumer app?

### Takeaway
SigLIP 2, Perception Encoder and TIPS are Apache-2.0 and safe. DINOv3 uses the custom "DINOv3 License", which grants use, redistribution and derivatives with no commercial prohibition, but adds conditions (ship the license text, trade-control and military bans, no reverse engineering, terminable on breach). MobileCLIP weights use the Apple ML Research Model TOU, which needs a careful read before shipping in any product.

### Cited findings
- DINOv3 License grant (Section 1.a): "non-exclusive, worldwide, non-transferable and royalty-free limited license" to "use, reproduce, distribute, copy, create derivative works of, and make modifications to the DINO Materials." The text does not explicitly mention commercial use either way. [DINOv3 LICENSE.md](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- DINOv3 redistribution (1.b.i): distribution of the materials or derivatives must be under the Agreement, with "a copy of this Agreement" provided. [DINOv3 LICENSE.md](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- DINOv3 restrictions: comply with laws incl. trade control and privacy (1.b.iii); no reverse engineering or decompiling (1.b.iv); no end uses prohibited by trade controls, including military, warfare, nuclear, espionage, weapons (1.b.v). Attribution is required only in published research (1.b.ii). [DINOv3 LICENSE.md](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- DINOv3 derivatives: you own your derivative works and modifications, subject to Meta's ownership of the originals (5.a). Fine-tuned models fall under the Agreement when distributed (1.b.i). [DINOv3 LICENSE.md](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- DINOv3 termination: Meta may terminate on breach, after which you must delete and stop using the materials; licenses also terminate if you sue Meta for infringement over the materials (5.b, 6). [DINOv3 LICENSE.md](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- The DINOv3 License has no MAU threshold, no naming requirement, and no separate acceptable use policy. [DINOv3 LICENSE.md](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- Hugging Face hosts DINOv3 checkpoints such as facebook/dinov3-vitl16-pretrain-lvd1689m. [HF DINOv3 ViT-L](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- MobileCLIP / MobileCLIP2: code MIT, model weights "Apple ML Research Model TOU", data CC-BY-NC-ND. [ml-mobileclip README](https://github.com/apple/ml-mobileclip)
- Perception Encoder: Apache-2.0 for models and code per repo badges; a separate LICENSE.PE file exists. [perception_models README](https://github.com/facebookresearch/perception_models)
- SigLIP 2: Apache-2.0. [HF siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224)
- TIPS: "All software is licensed under the Apache License, Version 2.0" and "All other materials are licensed under ... CC BY 4.0." [google-deepmind/tips](https://github.com/google-deepmind/tips)

### Inferences
- DINOv3 in a free (or even paid) consumer app looks permitted: the grant covers use and distribution of derivatives, and nothing restricts commercial use. Practical obligations: bundle the DINOv3 License text in the app (e.g. an acknowledgements screen), and the no-reverse-engineering clause means a user decompiling the model is not your problem but you cannot encourage it. The "non-transferable" and termination-on-breach terms make it weaker than Apache-2.0.
- Shipping DINOv3 weights inside an app binary is "distribution" under 1.b.i, so the license-copy requirement applies even though nothing is open-sourced.
- MobileCLIP weights are the riskiest for a shipped app. My recollection is that the Apple ML Research Model license limits use to research purposes, which would rule out a public app, but I did not fetch its text (see gaps).
- The CC-BY-NC-ND data license (DataCompDR) does not bind the weights, but matters if the project trains on that data.

### Gaps
- The exact text of the Apple ML Research Model TOU (whether it permits non-research or commercial deployment) was not fetched. Read it on the MobileCLIP2 Hugging Face cards before choosing MobileCLIP.
- Whether Meta's DINOv3 Hugging Face gate adds terms beyond LICENSE.md was not checked.
- Contents of PE's LICENSE.PE file not checked against the Apache badge.

## On-device latency and supported input resolutions

### Takeaway
Only MobileCLIP publishes phone latency: image encoders run in 1.5 ms (S0), 3.6 ms (S2) and 10.4 ms (B) on an iPhone 12 Pro Max. No primary source publishes Core ML or LiteRT latency for DINOv3, SigLIP 2, PE or TIPS on a phone. All candidates fit the "well under a second" target by a wide margin at 224 to 256 px; resolution, not parameter count, will drive latency for ViTs.

### Cited findings
- MobileCLIP2 image-encoder latency: S0 1.5 ms, S2 3.6 ms, B 10.4 ms, S3 8.0 ms, S4 19.6 ms, L/14 57.9 ms; MobileCLIP-S1 2.5 ms. The README names only "iPhone12 Pro Max" as the measurement device. [ml-mobileclip README](https://github.com/apple/ml-mobileclip)
- DINOv3 ViT-S/16 on-disk size: 82.4 MB at fp32, 41.2 MB at fp16. [arXiv 2604.27128](https://arxiv.org/pdf/2604.27128) (via search snippet)
- DINOv3 ViT-S/16 CPU (not phone) single-thread latency: 34.8 ms on an 8x8 patch grid, 663 ms at 32x32. [CanViT, arXiv 2603.22570](https://arxiv.org/pdf/2603.22570) (via search snippet)
- A 2025 study measured 190 real ViTs and 102 CNNs across two frameworks and six mobile platforms, but its abstract gives no per-model numbers. [A Study on Inference Latency for ViTs on Mobile Devices, arXiv 2510.25166](https://arxiv.org/abs/2510.25166)
- Resolutions: DINOv3 examples use 256 (ImageNet eval transform), 768, 896, 1024 px; ILIAS ran DINOv3 at 768. [dinov3 README](https://github.com/facebookresearch/dinov3); [ILIAS](https://vrg.fel.cvut.cz/ilias/)
- PE-Core T/16 and S/16 are trained at 384 px, B/16 at 224 px, PE-Spatial T/S/B at 512 px. [perception_models README](https://github.com/facebookresearch/perception_models)
- SigLIP 2 B variants exist at 224, 256, 384, 512 px plus NaFlex; ILIAS lists SigLIP2-B@384 and @512. [ILIAS](https://vrg.fel.cvut.cz/ilias/)
- Core ML conversion practice: export PyTorch to ONNX or trace directly, convert with coremltools, profile on device with Xcode Instruments; some ops are unsupported and need swapping. [FaceLiVTv2, arXiv 2604.09127](https://arxiv.org/pdf/2604.09127); [RepViT, arXiv 2307.09283](https://arxiv.org/pdf/2307.09283) (via search snippets)

Approximate weight sizes, computed as params x 2 bytes (fp16) or x 1 byte (int8 weight quantization). These are my arithmetic, not published file sizes:

| Model | Params | fp16 | int8 |
|---|---|---|---|
| MobileCLIP2-S0 image | 11.4M | ~23 MB | ~11 MB |
| DINOv3 ViT-S/16 | 21M | ~42 MB (41.2 MB measured) | ~21 MB |
| DINOv3 ViT-S+/16 | 29M | ~58 MB | ~29 MB |
| DINOv3 ConvNeXt-T | 29M | ~58 MB | ~29 MB |
| MobileCLIP2-S2 image | 35.7M | ~71 MB | ~36 MB |
| DINOv3 ConvNeXt-S | 50M | ~100 MB | ~50 MB |
| DINOv3 ViT-B/16, SigLIP2-B, TIPSv2-B | 86M | ~172 MB | ~86 MB |

### Inferences
- A ViT-S/16 at 224 px has 196 patch tokens, similar compute to DeiT-S. On recent iPhone Neural Engines, that class of model typically runs in single-digit to low tens of milliseconds, which leaves the budget dominated by camera capture, cropping and ANN search, not the encoder. This is an estimate, not a measured number.
- ViT cost grows roughly with token count squared in attention, so moving DINOv3 from 224 to 448 px quadruples tokens. ConvNeXt scales linearly and tends to map well onto NPUs, which makes DINOv3 ConvNeXt-T worth benchmarking against ViT-S+ despite missing retrieval numbers.
- RoPE and register tokens in DINOv3 ViTs may need care during Core ML / LiteRT conversion; ConvNeXt avoids that.

### Gaps
- No published Core ML (iPhone 15/16/17) or Android LiteRT/NNAPI latency for any DINOv3, SigLIP 2, PE or TIPS model. The project needs its own measurement.
- MobileCLIP2 native input resolution is not stated in the README (I believe S0 to S2 use 256 px, unverified).
- The MobileCLIP latency table does not state precision, compute unit, or iOS version.

## Does fine-tuning a small backbone on domain data beat a frozen large one?

### Takeaway
There is no direct published head-to-head for exact-match artwork retrieval. The evidence points to "a cheap adaptation on top of a strong frozen backbone helps a lot, and retrieval-specific fine-tuning of older models does not catch up with newer frozen foundation models". UnED-fine-tuned ViT-B models (UDON) score 8.3 on mini-ILIAS while frozen DINOv3-B scores 26.4.

### Cited findings
- On ILIAS, UDON-CLIP (ViT-B fine-tuned for universal embedding) gets 8.3 / 5.9 mAP@1k (5M / 100M) versus frozen DINOv3-B at 26.4 / 22.0 and frozen SigLIP2-B@384 at 18.4 / 15.0. [ILIAS project page](https://vrg.fel.cvut.cz/ilias/)
- Linear adaptation on UnED gives large gains for VLMs: SigLIP2-B@384 18.4 to 27.5, PE-L@336 27.1 to 39.6, OAI-CLIP-B 5.9 to 10.7; DINOv3-B is unchanged at 26.4 and DINOv3-L rises 31.1 to 32.9. [ILIAS project page](https://vrg.fel.cvut.cz/ilias/)
- Universal Image Embedding challenge: all winning entries initialized from and fine-tuned a large CLIP ViT-H/14 or ViT-L/14 (LAION-2B). [Towards Universal Image Embeddings, arXiv 2309.01858](https://arxiv.org/pdf/2309.01858) (via search snippet)
- E-commerce benchmark: top-tuning frozen embeddings gives mean gains of about 5% for SSL and 3.9% for text-image models; text-image top-tuning matched or beat supervised full fine-tuning on four of six datasets; full fine-tuning is "consistently strong"; SSL pretrained models outperform fine-tuned ones on food2k, fashion, SOP and Products-10k. The paper does not compare fine-tuned-small vs frozen-large directly. [arXiv 2504.07567](https://arxiv.org/html/2504.07567)
- Older retrieval work shows freezing the embedding network and training only a normalization or projection on hard positives and mined hard negatives can adapt to illumination shift. [No Fear of the Dark, arXiv 1908.08999](https://arxiv.org/pdf/1908.08999)
- No published work found on album-cover recognition from phone photos.

### Inferences
- The best bet for Record Scan is a two-step plan. First, a strong frozen small backbone (DINOv3 ViT-S/S+). Second, a light projection head or a short fine-tune trained on synthetic phone-photo pairs: perspective warp, glare, sleeve wear, partial crops and shrink-wrap reflections applied to the 3.13M gallery images. Because the gallery is the class set (one image per release), this is self-supervised instance discrimination and needs no labels.
- A projection head also lets the project shrink the embedding (e.g. 384 to 128 or 256 dims), which cuts the hnswlib index for 3.13M items from ~4.8 GB at 384-d fp32 to ~1.6 GB at 128-d, relevant if the index ever moves on device.
- Distractor robustness matters at 3M items: ILIAS shows rankings change between 5M and 100M distractors (PE-B falls from 20.2 to 9.7), so evaluate against the full 3.13M gallery, not a subset.

### Gaps
- No direct study found of fine-tuned small (5 to 90M) backbone vs frozen large backbone on near-duplicate artwork retrieval.
- No public numbers on how DINOv3-S changes after domain fine-tuning for retrieval.

## Recommendation: candidates to evaluate first
Based on the findings above (my judgment, not a sourced claim):
1. DINOv3 ViT-S+/16 (29M, frozen, then with a learned projection). It has the best published instance-retrieval accuracy per parameter (Oxford-H 50.0, vs DINOv2-S 39.5 and SigLIP 2 B 20.2), ~29 MB at int8, and a license that allows app distribution with the license text bundled. Test ViT-S/16 (21M) and ViT-B/16 (86M, Oxford-H 58.5) alongside to bracket the size/accuracy curve.
2. DINOv3 ConvNeXt-Tiny (29M). Same training and license, likely friendlier to the Neural Engine and LiteRT, but no published retrieval numbers, so it needs measuring.
3. MobileCLIP2-S2 (35.7M, 3.6 ms on iPhone 12 Pro Max) or S0 (11.4M, 1.5 ms) as the latency floor and VLM counterpoint, since VLMs win on product-style data. Only if the Apple ML Research Model TOU permits a public app; otherwise substitute PE-Core S/16 (Apache-2.0).
