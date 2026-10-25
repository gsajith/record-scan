# On-device stack for album cover retrieval (state as of October 2026)

Scope: camera frame, crop, embed, nearest-neighbor search over about 3.13M album covers (3,128,310 cover JPEGs before any collapse to one cover per album), result. Earlier work used 512-d float16 CLIP embeddings and hnswlib on a Mac.

Convention in these notes: a link after a claim is its source. "Vendor claim" means the only source is the company selling the thing. Arithmetic in the size tables was computed for this note and is labelled as such.

## 1. Inference runtimes: per-frame latency for ViT-S/16 or ViT-B/16 at 224 px

### Takeaway
A ViT-B/16-class image encoder at 224 px runs in roughly 10 to 21 ms per frame on a 2024 to 2026 flagship NPU (Qualcomm AI Hub measured figures) and about 10 ms on an iPhone 12 Pro Max for Apple's ViT-B/16-sized MobileCLIP-B (Apple's own figure). Embedding is not the bottleneck for 5 to 10 fps; first-load compilation on the Neural Engine and mid-range Android devices are the real risks.

### Cited findings

Android, measured (Qualcomm AI Hub, OpenAI CLIP ViT-B/16, 224x224, NPU). The page reports a single "OpenAI-Clip" latency per device and does not say whether it covers the image encoder alone or image plus text. [Qualcomm AI Hub, qualcomm/OpenAI-Clip](https://huggingface.co/qualcomm/OpenAI-Clip)

| Chipset (phone) | TFLite float | QNN DLC float | QNN DLC w8a16 | ONNX float | ONNX w8a16 | Peak memory (MB) |
|---|---|---|---|---|---|---|
| Snapdragon 8 Elite Gen 5 (Galaxy S26 class) | 12.8 ms | 13.2 ms | 11.5 ms | 10.4 ms | 10.5 ms | up to ~500 to 630 |
| Snapdragon 8 Elite (Galaxy S25) | 17.1 ms | 16.6 ms | 15.3 ms | 14.4 ms | 14.0 ms | up to ~480 to 590 |
| Snapdragon 8 Gen 3 (Galaxy S24) | 21.1 ms | 21.4 ms | 18.8 ms | 18.5 ms | 16.9 ms | up to ~690 to 710 |
| Snapdragon 8 Gen 1 | 36.4 ms | 37.7 ms | n/a | 33.3 ms | 32.2 ms | up to ~700 to 815 |
| Snapdragon 7 Gen 4 (mid-range) | n/a | n/a | 37.0 ms | n/a | 41.1 ms | up to ~630 |
| Automotive SoCs on CPU fallback (SA8775P etc.) | 520 ms (CPU) | | | | | |

- A plain ViT (86.6M params, 224x224) on Qualcomm AI Hub is listed at 330 MB float, 86.2 MB w8a16 and 83.2 MB w8a8 on disk; that page did not show latency figures. [Qualcomm AI Hub, ViT](https://aihub.qualcomm.com/models/vit)
- GPU path, peer-reviewed: Transformer-Lite measured OpenAI clip-vit-base-patch32 image encoder at 19.2 ms on Snapdragon 8 Gen 3 and 29.5 ms on MediaTek Dimensity 9300, on the mobile GPU (paper from 2024, older figures). [Transformer-Lite, arXiv 2403.20041](https://arxiv.org/pdf/2403.20041)
- An earlier search summary of the same Qualcomm listing quoted TFLite 13.4 ms on S24 and ~7.9 ms on 8 Elite Gen 5, which conflicts with the 21.1 ms and 12.8 ms in the current table; the table fetched directly from the model card is what is reported above. [Qualcomm AI Hub, qualcomm/OpenAI-Clip](https://huggingface.co/qualcomm/OpenAI-Clip)

iOS (Core ML on the Neural Engine):
- Apple's MobileCLIP model card gives image-encoder latency on an iPhone 12 Pro Max (A14, 2020 device, so older hardware): MobileCLIP-S0 1.5 ms (11.4M params), S1 2.5 ms, S2 3.6 ms (35.7M), MobileCLIP-B / MobileCLIP2-B 10.4 ms (86.3M params, ViT-B/16 size), S3 8.0 ms, S4 19.6 ms, L/14 57.9 ms. Vendor figures; no third-party verification found. [apple/MobileCLIP2-S0 model card](https://huggingface.co/apple/MobileCLIP2-S0)
- Zero-shot ImageNet top-1 from the same card: MobileCLIP2-S0 71.5%, S2 77.2%, B 79.4%, S4 81.9%. [apple/MobileCLIP2-S0 model card](https://huggingface.co/apple/MobileCLIP2-S0)
- Apple says MobileCLIP-S0 reaches zero-shot performance similar to OpenAI ViT-B/16 while being 4.8x faster and 2.8x smaller. [Apple ML Research, MobileCLIP](https://machinelearning.apple.com/research/mobileclip)
- MobileCLIP ships an iOS demo app and Core ML artifacts for offline photo search. [MobileCLIP mirror on SourceForge](https://sourceforge.net/mirror/mobileclip/)
- No published Core ML latency figure for vanilla ViT-B/16 on a 2025 or 2026 iPhone was found. An Apple Developer Forums thread (2026) reports a four-layer split-einsum attention microbenchmark at 21.6 ms on iPhone 18 Pro vs 111.4 ms on iPhone 17 Pro, at 4,096 tokens (ViT-B/16 uses 197), and reports that the first Core ML model load on the Neural Engine on iPhone 18 Pro can take minutes versus seconds on 17 Pro (up to 65x slower). [Apple Developer Forums thread 850058](https://developer.apple.com/forums/thread/850058)

Export paths:
- coremltools 9.0 is the latest stable release (10 Nov 2025); a 9.1.dev1 pre-release is dated 3 Aug 2026; wheels cover Python 3.7 to 3.13. [PyPI coremltools](https://pypi.org/project/coremltools/)
- ExecuTorch 1.0 left beta, added Python and Maven package variants for the Qualcomm (QNN) backend; Android backends listed are XNNPACK (CPU), Vulkan (GPU), Qualcomm AI Engine (NPU) and MediaTek (NPU); docs advise starting with XNNPACK and adding hardware backends after. [ExecuTorch 1.0.0 release notes](https://newreleases.io/project/pypi/executorch/release/1.0.0); [ExecuTorch 1.0 Android backends](https://docs.pytorch.org/executorch/1.0/android-section.html)
- ExecuTorch 1.0 also documents a Core ML backend and iOS examples, so one PyTorch export can target both platforms. [ExecuTorch Core ML backend docs](https://docs.pytorch.org/executorch/1.0/desktop-coreml.html); [ExecuTorch iOS examples](https://docs.pytorch.org/executorch/1.2/ios-examples.html)
- Qualcomm AI Hub ships the same ViT-B/16 CLIP as TFLite, QNN DLC and ONNX assets with float and w8a16 variants, which is a ready export path for Android. [Qualcomm AI Hub, qualcomm/OpenAI-Clip](https://huggingface.co/qualcomm/OpenAI-Clip)

### Inferences
- At 5 to 10 fps the frame budget is 100 to 200 ms. A ViT-B/16 at 10 to 21 ms on 2024+ flagships, or 37 to 41 ms on a Snapdragon 7 Gen 4, leaves headroom. Embedding every frame is affordable on flagships; on mid-range phones, embed only when the crop is stable.
- MobileCLIP-S2 (3.6 ms on a 2020 iPhone) or a distilled ViT-S/16 student is the obvious choice if the retrieval eval shows it matches ViT-B/16 on covers. That also cuts memory and thermals.
- The iPhone 18 Pro first-load regression means the app should compile the model once at install or first launch and cache it, and should not assume load time is seconds.
- A single ExecuTorch export with Core ML and QNN/XNNPACK backends is the least-effort cross-platform path. Core ML via coremltools on iOS plus Qualcomm AI Hub TFLite/QNN on Android is the best-measured path.

### Gaps
- No measured ViT-S/16 (22M params) latency on any 2025/2026 phone was found.
- No independent iPhone 15/16/17 Core ML number for ViT-B/16 was found; all iPhone numbers are Apple's, on an A14.
- No measured ExecuTorch or ONNX Runtime Mobile ViT latencies on Pixel (Tensor G4/G5) or Exynos were found; Qualcomm numbers do not cover non-Snapdragon Android, which is a large share of devices.
- ai-edge-torch (PyTorch to LiteRT) status in 2026 was not verified.

## 2. Vector search on device at 1 to 3 million vectors

### Takeaway
Libraries that run on iOS and Android exist (USearch with Swift and Java bindings, faiss and hnswlib as C++, ObjectBox, sqlite-vec), but none publishes a measured phone benchmark at million scale. USearch has the best fit on paper: f16, i8 and 1-bit storage plus memory-mapped "view" serving. faiss is the only one with IVF-PQ/OPQ.

### Cited findings
- USearch lists iOS and Android as supported platforms; the Swift binding supports add/search/remove, save/load/view and filter predicates; the Java binding supports add/search/remove, save/load/view and batch operations. [USearch README](https://github.com/unum-cloud/usearch)
- USearch supports downcasting to f16, bf16, e5m2 and i8 (i8 only with cosine-like metrics) and b1 single-bit vectors (for Hamming/Jaccard only). [USearch README](https://github.com/unum-cloud/usearch)
- USearch can serve an index from a memory-mapped file with random access, without loading it into RAM, and uses 40-bit neighbor IDs, which it says are 37.5% smaller than 8-byte IDs. [USearch README](https://github.com/unum-cloud/usearch)
- USearch's published benchmarks are server-side: 100M 96-d f32 vectors indexed in 0.3 h vs 2.6 h for FAISS on Intel Sapphire Rapids. Vendor benchmark, and no phone figures. [USearch README](https://github.com/unum-cloud/usearch); [USearch README on jsDelivr](https://cdn.jsdelivr.net/npm/usearch@2.26.0/README.md)
- hnswlib is header-only C++11 with no other dependencies, so it compiles for iOS/Android. Its documentation shows float32 data only and lists l2, ip and cosine distances; it does not document int8 or fp16 storage. M "strongly affects memory consumption". [hnswlib README](https://github.com/nmslib/hnswlib)
- faiss reference numbers on SIFT1M (128-d, server CPU, 20 threads, so throughput rather than single-query phone latency): HNSW Flat R@1 0.874 at efSearch 16 (0.011 ms/query) and 0.992 at efSearch 256 (0.104 ms/query); HNSW with scalar quantizer R@1 0.728 at efSearch 16 and 0.972 at efSearch 256 (0.039 ms/query). [faiss wiki, Indexing 1M vectors](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors)
- The same faiss page lists the nmslib HNSW index on SIFT1M as "512 + 796 MB", i.e. graph overhead larger than the 512 MB of raw vectors at its settings. [faiss wiki, Indexing 1M vectors](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors)
- ObjectBox (Android, Java, Dart/Flutter, Swift) has an HNSW vector index and says it finds results "within millions of entries in a matter of milliseconds" and does not need all vectors in memory, fetching from disk when needed. Vendor claim only; its Android benchmarks were announced as "coming soon". [ObjectBox on-device vector database](https://objectbox.io/vector-database-for-ondevice-ai); [ObjectBox Android post](https://objectbox.io/author/markus/)
- sqlite-vec is brute-force only, stores float, int8 and binary vectors in vec0 tables, and recommends binary quantization with oversampling and float re-scoring for scale. [sqlite-vec binary quantization guide](https://https-alexgarcia_xyz.proxy.bugsmash.io/sqlite-vec/guides/binary-quant.html); [sqlite-vec README](https://cdn.jsdelivr.net/npm/@photostructure/sqlite-vec@1.2.0/README.md)
- One benchmark measured sqlite-vec full scan at 67.84 ms for 100,000 384-d vectors (desktop CPU), with an 8-bit quantized scan in a different extension being faster. [The State of Vector Search in SQLite](https://marcobambini.substack.com/p/the-state-of-vector-search-in-sqlite)
- A secondhand podcast summary cites 500 ms for 500K 768-d vectors in sqlite-vec and binary quantization to reach about 1M; not from a published benchmark. [How AI Is Built, local-first search episode](https://howaiisbuilt.transistor.fm/episodes/local-first-search-how-to-push-search-to-end-devices)

### Inferences
- Scaling the sqlite-vec number linearly (67.84 ms per 100K 384-d float) gives about 2.1 s per query at 3.13M. Brute-force float search in SQLite is out for realtime at this scale.
- For a fully on-device index, the candidates are: faiss IVF-PQ (smallest, most mature PQ, C++ only, you write the Swift/Kotlin bridge) or USearch HNSW with i8 or b1 storage served memory-mapped (ready bindings, but the vectors themselves stay large, see section 3).
- Memory-mapped serving matters on iOS because clean mmapped pages do not count the same way as dirty heap against app memory limits. This is general OS knowledge, not verified against a 2026 Apple source here.
- Apple and Google first-party vector APIs: no shipped on-device ANN API for third-party million-vector indexes was found in this research (search budget did not cover WWDC 2026 sessions specifically).

### Gaps
- No measured query latency or RSS for any library at 1 to 3M vectors on an iPhone or Android phone. This needs a device benchmark.
- No source checked for a faiss mobile build maintained in 2026 (CocoaPods/Maven); building from source is assumed.
- Whether Apple's Core Spotlight / Foundation Models or Android AICore expose a general vector index API in 2026 was not verified.

## 3. Size math for 3.13M vectors and download limits

### Takeaway
Raw float16 vectors are 2.4 to 3.2 GB and int8 is 1.2 to 1.6 GB; only PQ (100 to 230 MB) or binary codes (150 to 200 MB) fit comfortably in a phone download. Apple-hosted Background Assets and Play Asset Delivery can technically ship a few GB, but a multi-GB first-run download is a poor product.

### Cited findings
Size table, computed for this note at N = 3,128,310 (1 GB = 10^9 bytes). Graph overhead uses hnswlib-style level-0 links of 2M x 4-byte IDs plus about 12 bytes of header and label per element; this is an estimate, not a published figure.

| Encoding | Bytes per vector | Vectors only | Plus HNSW graph (M=16, ~0.44 GB) | Recall note |
|---|---|---|---|---|
| float32 512-d | 2048 | 6.41 GB | ~6.85 GB | baseline (hnswlib default) |
| float16 512-d | 1024 | 3.20 GB | ~3.64 GB | near-lossless |
| float16 384-d | 768 | 2.40 GB | ~2.84 GB | near-lossless |
| int8 / SQ8 512-d | 512 | 1.60 GB | ~2.04 GB | faiss HNSW-SQ R@1 0.972 vs 0.992 flat at efSearch 256 on SIFT1M |
| int8 / SQ8 384-d | 384 | 1.20 GB | ~1.64 GB | as above |
| binary 512-d | 64 | 0.20 GB | n/a (scan) | needs rescoring; no cover-domain recall found |
| binary 384-d | 48 | 0.15 GB | n/a | as above |
| IVF-PQ 64 B + 8 B id | 72 | 0.23 GB (+~34 MB for 16,384 x 512 fp32 centroids) | n/a | see below |
| IVF-PQ 32 B + 8 B id | 40 | 0.13 GB (+~34 MB centroids) | n/a | see below |

- Recall reference for SQ8 HNSW: faiss SIFT1M figures in section 2. [faiss wiki, Indexing 1M vectors](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors)
- PQ recall reference on SIFT1M from a third-party benchmark (Vearch): IVFPQ recall@1 0.6167, recall@10 0.9797, recall@100 0.9960; plain PQ 0.6274 and OPQ+PQ 0.6250 recall@1; HNSW 0.9792. The extracted table did not map each row to PQ32 or PQ64. [Vearch evaluation](https://vearch.readthedocs.io/en/latest/evaluation.html)
- faiss reports 1-recall@1 as its most sensitive measure and notes the gap between methods narrows for longer codes (32 and 64 bytes per code). [faiss wiki, Indexing 1G vectors](https://github.com/facebookresearch/faiss/wiki/Indexing-1G-vectors/0c492083ccfe55b4101ee670b4fd6016a1c7c413)
- iOS: maximum uncompressed app size is 4 GB, maximum executable __TEXT 500 MB (iOS 9+). Apple recommends Background Assets for larger content. [App Store Connect, Maximum build file sizes](https://developer.apple.com/help/app-store-connect/reference/maximum-build-file-sizes)
- Apple-hosted asset packs (Background Assets): 200 GB total and 200 packs per app record, shared across platforms; no per-pack limit is listed. [App Store Connect, Apple-hosted asset pack size limits](https://developer.apple.com/help/app-store-connect/reference/apple-hosted-asset-pack-size-limits)
- On-demand resources are deprecated as of iOS 27 / iPadOS 27 and Apple recommends migrating to Background Assets; the ODR page lists a 70 GB hosted limit for iOS 18+. [App Store Connect, On-demand resources size limits](https://developer.apple.com/help/app-store-connect/reference/on-demand-resources-size-limits/)
- iOS cellular download cap: Apple's 2015 statement put it at 100 MB; one secondary source says 200 MB on iOS 12+. No current Apple page confirming the cap was found. [AppleInsider 2015](https://appleinsider.com/articles/15/02/12/apple-increases-size-limit-of-app-store-downloads-to-4gb); [Codemia](https://codemia.io/knowledge-hub/path/max_size_of_an_ios_application)
- Google Play Asset Delivery: 1.5 GB per asset pack, 4 GB cumulative for modules plus install-time packs, 4 GB cumulative for fast-follow plus on-demand packs (10 GB for Play Partner Program for Games), max 100 asset packs per bundle. Older third-party figures (1 GB install-time, 512 MB per fast-follow pack) appear outdated. [Google Play Console help](https://support.google.com/googleplay/android-developer/answer/9302563?hl=ur); [Medium, older figures](https://medium.com/mindful-engineering/google-play-asset-delivery-in-android-dcc2059e5a63)

### Inferences
- Album covers are a near-duplicate retrieval task: the photo of a sleeve should sit far closer to its own gallery embedding than SIFT queries do to their neighbors. PQ's distortion therefore likely hurts less than the SIFT recall@1 of ~0.62 suggests. This has to be measured on the gold photo set; it is the single most important number for the on-device decision.
- A two-stage on-device design fits in about 0.35 to 0.45 GB: binary or IVF-PQ32 codes for candidate generation, then PQ64 or a mmapped int8 rescoring of the top few hundred. Rescoring with int8 brings back the 1.6 GB file, so PQ64 rescoring is the realistic variant.
- Collapsing to one cover per release group, plus dropping 384-d or binary, scales every row down linearly. Halving N to ~1.5M puts IVF-PQ64 at ~110 MB.
- Shipping: a 150 to 250 MB index fits one Play asset pack (1.5 GB) and one Apple-hosted asset pack. Use fast-follow on Android and an essential or prefetch Background Assets pack on iOS, with server fallback until it lands.
- Index updates (new releases) are a whole-file re-download unless the format is sharded by IVF list or prefix.

### Gaps
- No published recall for PQ, OPQ or binary codes on CLIP-like 512-d embeddings in a cover or product retrieval domain was found.
- Exact current iOS cellular download cap not confirmed from a 2026 Apple source.

## 4. Hybrid alternative: index on a server

### Takeaway
The server side is small: one CPU box with 4 to 16 GB RAM holds the whole index (6.9 GB float32 hnswlib, ~2 GB faiss HNSW-SQ8, ~0.25 GB IVF-PQ), sub-millisecond to low-millisecond search, about €21/month on Hetzner after the June 2026 increase, or roughly $22/month on Cloudflare Vectorize at 100K queries. Network round trip, not search, sets latency.

### Cited findings
- Hetzner CAX31 (8 ARM vCPU, 16 GB RAM, 160 GB SSD, 20 TB traffic) is listed at €20.99/month after Hetzner's 15 June 2026 price adjustment, up from €12.49 in February 2026. [wz-it.com Hetzner comparison](https://wz-it.com/blog/hetzner-cloud-oder-dedicated-server/); [whtop.com listing](https://www.whtop.com/amp/plans/hetzner.com/144091)
- Hetzner's cost-optimized CX and CAX plans showed "Currently not available" on Hetzner's page as of a 17 Aug 2026 re-check, while Sparecores saw CAX31 at $0.0386/hour in Nuremberg, Helsinki and Falkenstein in August 2026. Sources conflict on availability. [Findstack](https://findstack.com/resources/hetzner-price-increase-2026); [Sparecores CAX31](https://sparecores.com/server/hcloud/cax31)
- Cloudflare Vectorize (GA): Workers Paid includes 50M queried dimensions/month then $0.01 per million, and 10M stored dimensions then $0.05 per 100M; Workers Free includes 30M queried and 5M stored dimensions; no charge for index hours or CPU. [Cloudflare Vectorize pricing](https://developers.cloudflare.com/vectorize/platform/pricing/)
- Vectorize limits: max 1,536 dimensions per vector; max vectors per index reported as 10,000,000 by one table (another note says 5M; not confirmed against the live limits page); topK up to 50 with values/metadata, 100 without. [developersdigest Vectorize summary](https://www.developersdigest.tech/vector-databases/cloudflare-vectorize); [Cloudflare Vectorize pricing](https://developers.cloudflare.com/vectorize/platform/pricing/)
- Server search cost reference: faiss HNSW on SIFT1M at 0.04 to 0.1 ms/query (20-thread throughput) at R@1 0.97 to 0.99. [faiss wiki, Indexing 1M vectors](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors)

### Inferences
- Request payload: 512 x fp16 = 1,024 bytes (about 1.4 KB base64) or 384 x fp16 = 768 bytes. Response is a handful of IDs. Uploading the vector, not the image, keeps the photo on the device.
- Memory on the box: hnswlib only takes float32, so 3.13M x 512-d is 6.41 GB of vectors plus ~0.44 GB graph at M=16, about 6.9 GB. That fits a 16 GB CAX31. faiss HNSW-SQ8 (~2 GB) or USearch i8 would fit a 4 GB box. (Arithmetic from section 3.)
- Search time at 3M x 512-d single-threaded is likely 1 to 5 ms (extrapolated from SIFT1M, 4x the dimension and 3x the data). End-to-end latency is dominated by the mobile network round trip, typically tens of ms on good LTE/5G and worse on poor signal; no source measured this here.
- Vectorize cost for this gallery, computed from the pricing above and assuming the "(stored vectors + queries) x dimensions" monthly formula: stored 3.13M x 512 = 1.60B dimensions, about $0.80/month overage. At 100K queries/month, queried dimensions are about 1.65B, about $16/month. With the $5 Workers Paid base, roughly $22/month. Each extra 100K queries adds about $0.51. A self-hosted €21 box has flat cost and no 1,536-d or topK caps.
- The Workers Free tier (5M stored dimensions) holds only about 9,800 vectors at 512-d, so the free tier is not an option at this scale.

### Gaps
- Oracle Cloud Always Free ARM (historically 24 GB) and other free tiers were not checked for October 2026 availability.
- The Vectorize queried-dimension formula and the per-index vector limit should be confirmed on Cloudflare's live docs.
- No measured mobile RTT or Vectorize query latency was gathered.

## 5. Camera pipeline at 5 to 10 fps and sleeve isolation

### Takeaway
Both platforms can deliver frames well above 10 fps to app code (AVFoundation video data output on iOS, CameraX ImageAnalysis on Android). Apple's Vision rectangle detector can find a square sleeve without a custom model. ML Kit's document scanner cannot, because it is a Play-services-hosted capture flow that does not hand per-frame results to the app.

### Cited findings
- ML Kit Document Scanner runs as a prebuilt UI flow delivered through Google Play services, needs no camera permission in the app, runs fully on-device, and allows only limited options such as page limit and gallery import. [ML Kit Document scanner](https://developers.google.com/ml-kit/vision/doc-scanner?authuser=0)
- Apple Vision has VNDetectRectanglesRequest for rectangle detection. The fetch of Apple's page returned only the title, so its properties (aspect ratio range, quadrature tolerance, minimum size, maximum observations) and the Swift-native DetectRectanglesRequest successor are from the fetch tool's general knowledge and are unverified. [Apple Developer, VNDetectRectanglesRequest](https://developer.apple.com/documentation/vision/vndetectrectanglesrequest)

### Inferences
- iOS: set minimum and maximum aspect ratio near 1.0 (LP sleeves are square), run the rectangle request on every frame or every other frame, perspective-correct the quad to 224x224, and embed. This needs no custom detector for the common case of a sleeve held flat on a contrasting background.
- Android: no first-party realtime rectangle detector was found. Options are OpenCV contour plus approxPolyDP on ImageAnalysis frames, ML Kit Object Detection in stream mode (prominent-object box, unverified here), or the existing YOLOv8n detector in the project data folder (`models/yolov8n.pt`) fine-tuned on sleeves.
- A center-crop square guide in the viewfinder is the zero-effort fallback on both platforms and may be good enough given that the user aims the phone at the cover.

### Gaps
- No measured frame-rate or latency figures for VNDetectRectanglesRequest or CameraX ImageAnalysis were collected.
- ML Kit Object Detection and Tracking stream mode behavior on album sleeves was not verified.

## 6. Recommended architecture: fully on-device vs hybrid

### Takeaway
Start hybrid: embed on the phone (10 to 40 ms), send a 1 KB vector to a single €21/month CPU box running faiss or hnswlib, return the top-k. Move search on-device only if the gold-set eval shows IVF-PQ or binary codes at 150 to 250 MB keep top-1 accuracy within a point or two of float search.

### Cited findings
- On-device embedding is measured at 10 to 21 ms on 2024 to 2026 Snapdragon flagships and 37 to 41 ms on Snapdragon 7 Gen 4. [Qualcomm AI Hub, qualcomm/OpenAI-Clip](https://huggingface.co/qualcomm/OpenAI-Clip)
- A ViT-B/16-sized MobileCLIP-B image encoder runs in 10.4 ms on iPhone 12 Pro Max (vendor figure). [apple/MobileCLIP2-S0 model card](https://huggingface.co/apple/MobileCLIP2-S0)
- Play asset packs cap at 1.5 GB each, and Apple-hosted asset packs at 200 GB total, so a 0.2 GB index ships on both stores. [Google Play Console help](https://support.google.com/googleplay/android-developer/answer/9302563?hl=ur); [App Store Connect, Apple-hosted asset pack size limits](https://developer.apple.com/help/app-store-connect/reference/apple-hosted-asset-pack-size-limits)
- Float16 at 512-d costs 3.2 GB and int8 1.6 GB for 3.13M vectors (section 3 arithmetic), both too large for a casual app download.

### Inferences
- Latency budget, hybrid: crop 5 to 15 ms (estimate) + embed 10 to 40 ms + network 30 to 150 ms (estimate) + server search 1 to 5 ms (extrapolated) = about 50 to 200 ms per identification. That is near-realtime if the app sends one request when the crop is stable rather than every frame.
- Latency budget, on-device with IVF-PQ: crop + embed as above + PQ scan of a few IVF lists, likely single-digit ms (unmeasured) = under 60 ms on flagships, works offline, zero server cost. Cost: a 150 to 250 MB asset download, an unknown recall hit, and an index update story.
- The decision hinges on one experiment the project can run on the Mac today: build faiss IVF-PQ64, IVF-PQ32, OPQ variants and binary+rescore indexes over the embedded prefixes, query with the gold phone photos, and compare top-1/top-5 against exact float search. If PQ64 holds within ~1 to 2 points, ship on-device with server fallback; if not, stay hybrid.
- Hybrid also keeps the preview lookup (Deezer) online anyway, so "fully offline" buys less than it seems: the user needs a network to play the preview.

### Gaps
- No on-device search benchmark at 3M scale exists in the sources; the on-device latency figure above is an estimate.
- Battery and thermal cost of continuous embedding at 5 to 10 fps was not measured in any source found.
