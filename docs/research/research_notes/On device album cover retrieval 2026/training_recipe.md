# Training recipe for phone-photo to album-cover retrieval (2026)

Scope: how to train or fine-tune an embedding model that maps a phone photo of a record sleeve to one of 3.13M clean Cover Art Archive covers, and how to distill it into a phone-sized student. Researched 2026-10-24. Citation format: each fact ends with its source link. Anything without a link is in Inferences or Gaps and is my own reasoning or estimate.

Short version of the recommended plan (details and evidence below):

1. Build the eval harness and gold set first (real photos plus "no record" negatives). Score at release-group level.
2. Benchmark frozen teachers on clean gallery with no index-time augmentation: current CLIP ViT-B/32, DINOv3 ViT-B/L, SigLIP 2, SSCD ResNet-50.
3. Fine-tune the best teacher SSCD-style on synthesized photo/clean pairs: InfoNCE with large batch or memory bank, KoLeo entropy term, GeM pooling, 256 to 512 dim, PCA whitening.
4. Distill into a 10 to 30M parameter student. Precompute teacher embeddings once (MobileCLIP-style "reinforced" dataset). Train the student so its photo embedding lands on the teacher's clean-cover embedding (asymmetric retrieval), plus a relational KL term (RDCD style).
5. Rent GPUs for steps 3 and 4. The M1 is fine for the eval harness, small ablations and on-device export.

## 1. Frozen vs fine-tuned encoders, losses and head design for exact-instance retrieval

### Takeaway
Every strong result in instance retrieval and copy detection since 2021 fine-tunes or trains a task-specific head on top of a large pretrained backbone. Frozen general embeddings lose by large margins. Two loss families win: classification-style ArcFace or sub-center ArcFace on a small projection (GUIE 2022), and instance-contrastive InfoNCE with an entropy regularizer and heavy copy augmentations (SSCD, ISC 2021). For one-photo-per-cover data like Record Scan's, the SSCD-style contrastive recipe fits better, with GeM pooling, a linear projection to 256 to 512 dims, PCA whitening and score normalization.

### Cited findings

Google Universal Image Embedding (GUIE, Kaggle 2022, 64-d embeddings, mP@5; includes artwork and packaged goods):
- The challenge required a 64-d embedding, retrieval by kNN with k=5, scored by mean Precision@5, within a 9 hour inference limit. [Shao & Cui, 1st place](https://arxiv.org/pdf/2210.08473)
- The top six solutions all built on CLIP ViT-H/14 with multi-domain data. The frozen CLIP ViT-H/14 reference scored 62.1 on the private split against 72.8 for the winner, so fine-tuning added about 10.7 points. [GUIE challenge paper](https://arxiv.org/pdf/2309.01858)
- 1st place recipe: OpenCLIP ViT-H pretrained on LAION-2B, a linear projection to 64-d, ArcFace head, dropout 0.2 between the last two linear layers, SGD with momentum, weight decay 1.5e-4. Training data was Products-10K, Shopee, Met Artwork, Alibaba goods, H&M, GPR1200, GLDv2-Full and DeepFashion consumer-to-shop, with Products-10K used for a last fine-tuning pass. [Shao & Cui](https://arxiv.org/pdf/2210.08473)
- 1st place schedule: train only the last two fully connected layers to convergence for 6 epochs, then freeze that head and train only the backbone for 3 epochs. Freezing the head while tuning the backbone scored 0.678 private versus 0.677 for LP-FT, a negligible gap. [Shao & Cui](https://arxiv.org/pdf/2210.08473)
- 1st place resolution trick: fine-tuning at 280 then 290 px with 4-pixel patch overlap reached 0.728 private, the same as 336 px. Naive output averaging made ensembles worse than the best single model unless the models shared a frozen head. [Shao & Cui](https://arxiv.org/pdf/2210.08473)
- 3rd place: OpenCLIP ViT-H/14 with ArcFace, two-stage training, 0.692 private. 5th place: OpenCLIP ViT-H with a BatchNorm plus Linear head trained with ArcFace, 0.688 private. 6th place: CLIP LAION-2B with sub-center ArcFace, dynamic margins and PCA for dimensionality reduction. [3rd](https://cdnjs.deepai.org/publication/3rd-place-solution-for-google-universal-image-embedding), [5th](https://arxiv.org/pdf/2210.09495), [6th](https://ar5iv.labs.arxiv.org/html/2210.09377)
- 1st place found dynamic margin did not beat vanilla ArcFace with their very large class count. They suspected margins need to account for class count and that a 64-d space limits classification over many classes. [Shao & Cui](https://arxiv.org/pdf/2210.08473)

Copy detection (Meta ISC 2021 / DISC21, and SSCD):
- ISC 2021 had a descriptor track (256-d vectors, L2 distance) and an unconstrained matching track. [ISC 2021 results](https://arxiv.org/pdf/2202.04007)
- The organizers' findings: "Strong and non-standard image augmentations that mimic typical cases of image copies are very beneficial in the training." Instance-discrimination self-supervised learning mattered both for pretraining and as the main training stage. Score normalization had a large effect in both tracks. [ISC 2021 results](https://ar5iv.labs.arxiv.org/html/2202.04007)
- Descriptor track winner (Lyakaap-dt1, Shuhei Yokoo, DeNA): EfficientNetV2 with GeM pooling, then linear plus batch norm plus L2 norm; contrastive loss with a cross-batch memory bank; one view with enriched augmentations, one with conventional ones; negative embedding subtraction as post-processing; supervised ImageNet init. [ISC 2021 results](https://ar5iv.labs.arxiv.org/html/2202.04007)
- The hardest ISC transformations were severe crops and insertion onto unrelated images, especially combined with local pixel perturbations. [ISC 2021 results](https://ar5iv.labs.arxiv.org/html/2202.04007)
- SSCD architecture: ResNet-50 trunk, GeM pooling, a simple linear projection (the SimCLR MLP replaced), L2 norm, then PCA whitening learned on the DISC21 training set at full descriptor size. [SSCD, Pizzi et al., CVPR 2022](https://arxiv.org/pdf/2202.10261)
- SSCD loss: L = InfoNCE-mix + λ·L_KoLeo, where the KoLeo differential entropy term spreads descriptors apart. Standard setting λ = 30. Training became unstable above λ = 40. Entropy regularization alone added +17.4 µAP in the stepwise ablation. [SSCD](https://arxiv.org/pdf/2202.10261)
- SSCD training: SimCLR schedule, batch 4096, 224x224, LR 0.3 × N/256, weight decay 1e-6, 100 epochs, cosine schedule with linear warmup, LARS optimizer, temperature τ = 0.05 (SimCLR default is 0.1). Negatives and KoLeo neighbors come from the global batch across GPUs, with synchronized batch norm. Inference resizes the short edge to 288, larger than training. [SSCD](https://arxiv.org/pdf/2202.10261)
- GeM pooling at training time helped only together with the entropy term. With plain InfoNCE, average pooling trained better. [SSCD](https://arxiv.org/pdf/2202.10261)
- SSCD DISC21 results, µAP / µAP with score normalization: SimCLR trunk 13.1 / 33.9; SimCLR-CD (all changes but entropy, 2048-d) 39.8 / 56.8; SSCD ImageNet-trained 512-d with strong blur 50.4 / 64.5, advanced augs 55.5 / 71.0, plus mixup 56.8 / 72.2; SSCD DISC-trained 512-d plus mixup 61.5 / 72.5; SSCD-large ResNeXt-101 1024-d 63.7 / 75.3. [SSCD](https://arxiv.org/pdf/2202.10261)
- SSCD on Copydays strong subset: ResNet-50 512-d at short side 288 reached 86.6 mAP and 98.1 µAP, against DINO ViT-B/16 (1536-d) at 82.8 mAP / 92.3 µAP. [SSCD](https://arxiv.org/pdf/2202.10261)
- SSCD trained on a disjoint 1 million image DISC training set, which also served as the background set for score normalization. [SSCD](https://arxiv.org/pdf/2202.10261)

Follow-ups 2024 to 2026:
- RDCD (WACV 2025) distills a ResNet-50 teacher into an EfficientNet-B0 student with relational self-supervised distillation. It improves µAP by 5.0 / 4.9 / 5.9 points over the baseline at 64 / 128 / 256 dims. Its 64-d descriptor reaches µAP_SN 53.5, close to DINO ViT-B/16 at 1536-d (53.8). [RDCD](https://arxiv.org/pdf/2405.17928), [WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Kim_Relational_Self-Supervised_Distillation_with_Compact_Descriptors_for_Image_Copy_Detection_WACV_2025_paper.html)
- A 2025 manuscript, "Towards Image Copy Detection at E-commerce Scale" (DTML, dual-triplet metric learning), claims to beat DINO and SSCD descriptors on DISC2021. I saw only the abstract. [search result](https://sigport.org/node/7919)
- DINOv3 (August 2025) instance recognition, frozen features: the 7B teacher scores Oxford-Hard 60.7, Paris-Hard 87.1, Met GAP 55.4, AmsterTime 56.5. DINOv2 g/14 scores 58.2 / 84.6 / 44.6 / 48.9. SigLIP 2 g/16 scores only 25.1 / 60.9 / 13.9 / 15.5, and PEcore G/14 32.7 / 68.9 / 10.6 / 23.1. [DINOv3, Table 9](https://arxiv.org/html/2508.10104v1)
- MobileCLIP2 authors note that their DFNDR-2B models "do not always achieve state-of-the-art retrieval performance", which they attribute to the dataset's bias toward zero-shot classification. [MobileCLIP2](https://arxiv.org/html/2508.20691v1)

### Inferences
- Frozen CLIP ViT-B/32 (the November 2025 baseline) is an old, weak starting point for instance matching. The DINOv3 table shows CLIP-style text-aligned models (SigLIP 2, PEcore) far below DINO-family self-supervised models on instance retrieval and on Met, the artwork benchmark closest to album covers. Benchmark DINOv3 ViT-B and ViT-L frozen as the first new baseline.
- Choose InfoNCE over ArcFace for this data. Each cover has one clean image, so every "class" holds a single sample plus its synthetic views. ArcFace over 3.13M classes needs a 3.13M × d weight matrix (about 6.4 GB at 512-d fp32) and a partial-FC sampler. InfoNCE with a memory bank needs neither. If you try ArcFace, use sub-center ArcFace (designed for noisy labels, used by GUIE 6th place) because duplicate covers across releases act as label noise.
- Make the loss asymmetric. Anchor is the clean cover (the gallery side), positive is a synthetic photo of that cover, and negatives are other clean covers. That matches the deployment query distribution better than SimCLR's symmetric two-augmented-views setup. Keep one lightly augmented clean view per pair so the gallery embedding stays stable.
- Head: GeM pooling, one linear layer to 256 or 512 dims, L2 norm, PCA whitening learned on the gallery. GUIE shows 64-d works for mP@5. RDCD shows 64-d can approach a 1536-d DINO descriptor. 128 to 256 dims is the sensible range if the index has to fit on the phone or stay cheap on a server.
- Copy the GUIE winner's two-phase schedule (head first, then backbone with frozen head) when fine-tuning the teacher. It is cheap insurance against the head distorting pretrained features.

### Gaps
- No paper found that reports results on album covers or record sleeves specifically. The closest public precedents are a 2008 Stanford EE368 CD-cover-from-camera-phone project and an AWS Rekognition tutorial, neither with comparable metrics. [EE368](https://web.stanford.edu/class/ee368/Project_08/index.html), [AWS tutorial](https://community.aws/content/2mISVlhhsupsSxJVsNEoEKayaBY)
- I did not get the exact score progression for the GUIE 1st place stages beyond the ablations quoted above, or the GPU hours it used.
- I did not verify the DTML 2025 margins or find a current DISC21 leaderboard for 2025 to 2026.
- I did not find a head-to-head of triplet with hard negatives versus InfoNCE versus SupCon on DISC21 in the sources read. SSCD and the ISC winner both used InfoNCE-style losses.

## 2. Copy-detection-style augmentation to simulate phone photos of flat artwork

### Takeaway
The published copy detection recipes use strong blur, crops, rotation, text and emoji overlays, JPEG, and mixup/cutmix, plus insertion onto unrelated backgrounds. None of them targets camera capture of a physical sleeve. The phone-photo terms (perspective, glare, shadows, wear, hands, white balance) have to be added from general-purpose library transforms, and their ranges must be tuned against the real gold photos.

### Cited findings
- SSCD augmentation sets, applied incrementally. Base: horizontal flip, random crop, color jitter, grayscale, Gaussian blur. Strong blur: 50% large-radius Gaussian blur with σ in [1, 5]. Advanced: 10% rotation, 10% text, 20% emoji, 20% JPEG compression. Mixed: 2.5% mixup, 2.5% cutmix. [SSCD Table 1](https://arxiv.org/pdf/2202.10261)
- SSCD details: half the rotations are multiples of 90 degrees and half are unconstrained. Text overlays use random font, string, opacity, size and color. Emoji are random size. JPEG quality is sampled randomly. The authors argue blur invariance gives a low-frequency bias that reduces sensitivity to high-frequency noise in real copies. [SSCD](https://arxiv.org/pdf/2202.10261)
- Stronger augmentations added about +6 µAP to SSCD (50.4 to 56.8 on ImageNet-trained 512-d models). [SSCD](https://arxiv.org/pdf/2202.10261)
- ISC 2021: every top method used augmentation sets richer than standard classification augmentation. Severe crops and pasting onto unrelated images were the hardest cases. [ISC 2021 results](https://ar5iv.labs.arxiv.org/html/2202.04007)
- The ISC 2021 descriptor winner fed one strongly augmented and one conventionally augmented view per pair. [ISC 2021 results](https://ar5iv.labs.arxiv.org/html/2202.04007)
- AugLy (Meta) is MIT licensed and provides meme formatting, text and emoji overlays, and an overlay_onto_screenshot augmentation. Its README lists the Image Similarity Challenge (which produced DISC21) among its uses. It does not advertise a perspective transform. [AugLy](https://github.com/facebookresearch/AugLy)
- Albumentations Perspective defaults: scale=(0.05, 0.1), the standard deviation of corner displacement as a fraction of image size, with keep_size=True. Affine and ShiftScaleRotate (shift ±0.0625, scale ±0.1, rotate ±45 degrees by default) are also available. [Albumentations geometric docs](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/transforms/)
- The maintained package is now AlbumentationsX, dual licensed AGPL-3.0-only or commercial. "Earlier releases retain the permissions that accompanied them." It ships RandomSunFlare, RandomShadow, PlasmaShadow, ISONoise, ImageCompression, MotionBlur and Defocus. [AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX)

### Inferences
Proposed photo simulator for Record Scan. The ranges are my starting points, not published values. Tune each one by comparing histograms of simple statistics (blur level, mean brightness, glare area, skew) between synthetic and real gold photos.

- Detector crop error and background. Paste the warped cover onto a background photo (tables, shelves, carpets, other sleeves) so it fills 55 to 95% of the frame, then crop 70 to 100% of the cover area with 0 to 15% offset. This mirrors the ISC "insertion onto unrelated images" hard case and the YOLO crop step in the pipeline.
- Perspective: corner jitter σ of 0.03 to 0.12 of image size (around the Albumentations default 0.05 to 0.1), plus in-plane rotation of ±15 degrees. A 90 degree rotation at p ≈ 0.05 covers sideways shots.
- Lighting and color: brightness and contrast ±0.3, saturation ±0.3, hue ±0.03, gamma 0.7 to 1.5, and a white balance shift (per-channel gain 0.85 to 1.15) to mimic tungsten versus daylight.
- Specular glare from shrink-wrap and glossy board: additive bright elliptical blobs and streaks covering 2 to 25% of the area at p ≈ 0.4 (RandomSunFlare or a custom blob), occasionally clipping to white.
- Shadows: RandomShadow or PlasmaShadow at p ≈ 0.3, including the phone-and-hand shadow shape.
- Blur and sensor: defocus radius 1 to 4 px and motion blur kernel 3 to 11 px at the 224 to 256 training size, ISONoise, JPEG quality 40 to 95. Keep SSCD's 50% Gaussian blur σ 1 to 5 as a baseline arm.
- Hand occlusion: skin-toned blobs or real hand cutouts entering from an edge, up to 20% of the area, at p ≈ 0.25.
- Sleeve wear: ring wear (a darker or lighter ring at 70 to 90% of sleeve width), creases, corner dings, sun fade (desaturate plus yellow shift), and price or hype stickers made with the SSCD-style text overlay. Stickers matter because their text can fool a text-sensitive model.
- Use kornia for GPU-batched geometric and photometric ops if CPU augmentation bottlenecks rented GPUs. Keep heavy compositing (backgrounds, hands, stickers) on CPU workers or pre-render it.
- Licensing: if the app may become commercial, pin the last MIT albumentations release or use kornia and AugLy (MIT). AlbumentationsX is AGPL or paid.
- A bold option worth one experiment: render sleeves in Blender (one textured, slightly bent board under a few HDRI lights with a glossy clearcoat) for 50k to 100k realistic glare and perspective samples. The Blender MCP is already installed. Do this only if the 2D simulator plateaus on the gold set.

### Gaps
- No primary source found that publishes augmentation parameter ranges tuned for photographed flat artwork such as posters, book covers or sleeves.
- I did not fetch the kornia or AugLy API docs for exact parameter signatures. Check them before implementing.
- I did not find the last MIT-licensed albumentations version number. Check PyPI history before pinning.

## 3. Distillation into a 5 to 30M parameter student

### Takeaway
Two published patterns transfer directly. MobileCLIP's "reinforced dataset" precomputes teacher outputs once so student training never runs the teacher. RDCD-style relational distillation keeps copy-detection quality at 64 to 256 dims in an EfficientNet-B0. Good student inits that fit the budget already exist: the MobileCLIP2-S0 image tower (11.4M params, 1.5 ms on an iPhone 12 Pro Max) and the DINOv3 ViT-S (21M) and ConvNeXt-Tiny (29M) distillations. The big labs' compute numbers (hundreds of GPUs) come from general pretraining. Task-specific distillation on 3M covers is a hobby-scale job.

### Cited findings
- MobileCLIP2 students: S0 image encoder 11.4M params, 1.5 ms image latency, 71.5% ImageNet zero-shot; S2 35.7M, 3.6 ms, 77.2%; B 86.3M, 10.4 ms, 79.4%. Latency was measured on an iPhone 12 Pro Max. [MobileCLIP2](https://arxiv.org/html/2508.20691v1)
- MobileCLIP2 loss: L_total = (1 − λ)·L_CLIP + λ·L_distill, with λ = 1.0 for the final models (pure distillation). L_distill is a KL divergence between row-wise softmaxes of the teacher and student similarity matrices, in both directions, averaged over K teachers. Each teacher has its own temperature (logit scale 60 or 70). The authors report the logit scale is "not a sensitive hyperparameter" within about 5 points. [MobileCLIP2](https://arxiv.org/html/2508.20691v1)
- MobileCLIP2 teachers: an ensemble of DFN2B-CLIP-ViT-L-14-s39b and DFN2B-CLIP-ViT-L-14 at 224 px. The reinforced dataset DFNDR-2B holds 1.9B samples (162 TB) with 5 synthetic captions per image and stored teacher outputs. [MobileCLIP2](https://arxiv.org/html/2508.20691v1)
- MobileCLIP2 compute: 13B seen samples on 32×8 A100-40GB or 16×8 H100-80GB; final recipe 200k iterations at global batch 65,536; ablations on 4×8 H100 at batch 8192. GPU hours are not reported. The paper is internally inconsistent on per-teacher logit scales and on augmentation count (2 versus 30). [MobileCLIP2](https://arxiv.org/html/2508.20691v1)
- TinyCLIP (ICCV 2023) uses affinity mimicking (the student matches the teacher's image-text similarity distributions) and weight inheritance from the teacher. ViT-8M/16 trained on YFCC-15M reaches 41.1% ImageNet zero-shot, beating original CLIP ViT-B/16 by 3.5 points with 8.9% of its parameters. Weight inheritance speeds training 1.4 to 7.8× versus from scratch. [TinyCLIP](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_TinyCLIP_CLIP_Distillation_via_Affinity_Mimicking_and_Weight_Inheritance_ICCV_2023_paper.html)
- DINOv3 distills a fixed 6.7B ViT-7B teacher into ViT-S (21M), ViT-S+ (29M), ViT-B (86M), ViT-L (300M), ViT-H+ (840M) and ConvNeXt T/S/B/L (29M / 50M / 89M / 198M). It runs multiple students at once, sharing teacher inference across GPUs, so each added student costs only its own training. Schedule: 1M iterations, 250k cooldown, then a high-resolution phase. [DINOv3](https://arxiv.org/html/2508.10104v1)
- DINOv3 Oxford-Hard retrieval retained by students: ViT-S 49.5, ViT-S+ 50.0, ViT-B 58.5, ViT-L 63.1, ViT-H+ 64.5, against 60.7 for the 7B teacher. ViT-S keeps about 82% of the teacher's score; ViT-B keeps about 96%. [DINOv3](https://arxiv.org/html/2508.10104v1)
- RDCD: ResNet-50 teacher to EfficientNet-B0 student via relational self-supervised distillation, compact descriptors at 64 to 256 dims, gains of 5.0 / 4.9 / 5.9 µAP over its baseline, and a 64-d µAP_SN of 53.5 versus DINO ViT-B/16 1536-d at 53.8. [RDCD](https://arxiv.org/pdf/2405.17928)

### Inferences
- Recommended student recipe:
  1. Teacher: the fine-tuned model from section 1 (or frozen DINOv3 ViT-L if fine-tuning does not pay off).
  2. Precompute and store teacher embeddings for every clean cover, the gallery targets. At 3.13M × 512 fp16 that is about 3.2 GB.
  3. Train the student on synthetic photos. Loss 1 is cosine or L2 regression of student(photo) onto teacher(clean cover), which is asymmetric distillation. Loss 2 is InfoNCE where the student photo embedding must pick its teacher clean embedding out of a large bank of other teacher clean embeddings (a memory bank is cheap because the targets are precomputed). Loss 3 is an optional RDCD/MobileCLIP-style KL between the student's and teacher's in-batch similarity distributions.
  4. Then decide between two deployment modes. In asymmetric mode the gallery stays in teacher space and only the query runs the student. That keeps the gallery quality of the big model and needs no re-embedding when the student changes. In symmetric mode the student embeds the gallery too, which is needed only if the index must be rebuilt on-device. Start asymmetric.
- Student candidates in priority order: MobileCLIP2-S0 image tower (11.4M, measured 1.5 ms on iPhone), DINOv3 ConvNeXt-Tiny (29M, CNN-friendly for Core ML and NNAPI), DINOv3 ViT-S (21M). EfficientNet-B0 has published copy-detection distillation results (RDCD) if a plain CNN is wanted.
- Expect a ViT-S-class student to keep roughly 80 to 95% of the teacher's recall on clean-ish photos, based on the DINOv3 Oxford-Hard retention above. That is a guess from a related benchmark, not a measurement on covers. The gap is likely widest on glare-heavy and text-only covers.
- This replaces the 13.6-vectors-per-image index-time augmentation. Robustness moves into the query encoder, so the index holds one vector per cover (or one per deduplicated cover group).

### Gaps
- No source found that reports task-specific (instance retrieval) distillation cost in GPU hours for a 3M-image gallery. MobileCLIP2, TinyCLIP and DINOv3 report hardware or speedups, not hours, and all are general pretraining.
- I did not find the original MobileCLIP (CVPR 2024) compute figures in the sources read.
- DINOv3 student scores on Met (the artwork benchmark) were not in the extracted text. Only Oxford-Hard was.

## 4. Evaluation design for this domain

### Takeaway
Score at the level the user cares about (the right album, ideally the right release group) and treat identical covers across releases as correct. Use recall@1 and recall@5 on real photos as headline metrics. Add µAP over a pool that includes "no record" and "record not in catalog" queries, because µAP measures calibration across queries, which is what a lock threshold depends on. SSCD and ISC both show score normalization and global calibration matter as much as ranking.

### Cited findings
- DISC21 uses micro average precision (µAP), a single precision-recall curve over all queries pooled. That rewards descriptors whose distances are comparable across queries. SSCD reports µAP both raw and after score normalization. [SSCD](https://arxiv.org/pdf/2202.10261)
- SSCD reports that its method improves µAP far more than per-query mAP on Copydays, which it reads as "better distance calibration". On high-resolution images it saw better mAP but worse µAP. [SSCD](https://arxiv.org/pdf/2202.10261)
- Score normalization had "a significant impact" in ISC 2021. The descriptor winner used negative embedding subtraction. SSCD uses similarity normalization against a background set of 1M training images. [ISC 2021 results](https://ar5iv.labs.arxiv.org/html/2202.04007), [SSCD](https://arxiv.org/pdf/2202.10261)
- DISC21 deliberately includes queries with no match in the reference set, which is why µAP rather than recall is the metric. [ISC 2021 results](https://arxiv.org/pdf/2202.04007)

### Inferences
- Gold set: 200 to 500 photos of Gautham's own records, several per sleeve, deliberately varied (lamp, daylight, glare, angle, hand on sleeve, shelf background, worn copies, shrink-wrapped copies). Add 100 to 200 "no record" frames (desk, shelf spines, a CD, a book cover, a screen) and some records known to be missing from Cover Art Archive. Label each photo with its MusicBrainz release MBID and let the harness map that to release group.
- Correctness rule: a hit is correct if the top result is in the same release group, or if its cover image is a near-duplicate of the true cover (teacher cosine above a threshold calibrated by eye, or pHash Hamming distance under about 8 of 64 bits). Report strict (exact release) and lenient (release group or identical art) numbers side by side. Strict release accuracy is unachievable for identical sleeves, so it should not gate decisions.
- Deduplicate the gallery once. Cluster covers by teacher embedding plus pHash, collapse each cluster to one index entry that lists all its releases, and rank within the cluster by metadata (vinyl format first, then release date or popularity). This also shrinks the index and removes false negatives from training.
- Metrics: recall@1 and recall@5 (lenient) on real photos; the same on a 10k to 50k synthetic query set built with a held-out augmentation seed (catches regressions quickly on the M1); µAP over real plus no-record queries; per-slice recall for text-only covers, glare-heavy photos, dark covers and photos with occlusion.
- Gallery for eval: always search the full 3.13M (or deduplicated) gallery. Scores on a 100k subset will overstate accuracy because hard negatives (same artist, same label house style) are missing.
- Lock threshold: use cosine top-1 score after normalization, plus the margin between top-1 and the best result from a different cover cluster (not the next duplicate). Pick the threshold on the gold set to reach about 99% precision on accepted locks, then report the recall at that point. Require agreement over 3 to 5 consecutive frames before locking, which cuts false locks on video at low cost.
- Avoid training on the gold photos. If more real data is needed later, collect a separate fine-tuning set.

### Gaps
- No published benchmark exists for this exact task, so there is no external target number. The gold set defines success.
- I found no source on pHash thresholds for "same artwork, different scan" across Cover Art Archive uploads. The Hamming cutoff above is a starting guess to tune by inspecting pairs.

## 5. Compute realism: M1 Mac versus rented GPUs, and 2026 prices

### Takeaway
Use the M1 for the eval harness, synthetic-query generation, small ablations on 50k to 200k covers, and Core ML export. Rent one GPU (RTX 4090 or A100/H100) for embedding 3.13M covers with a big teacher and for the student training runs. At 2026 marketplace prices a full teacher-plus-student cycle should cost tens of dollars, not hundreds. The bigger practical cost is moving 250 GB of JPEGs off the Seagate drive.

### Cited findings
- RunPod official pricing page (updated July 2026): H100 80GB from $1.99/hr, A100 80GB from $1.19/hr, RTX 4090 24GB from $0.34/hr, billed per second. Third-party trackers list Secure Cloud prices higher, for example RTX 4090 around $0.69 to $0.74 and H100 PCIe around $2.89. [RunPod pricing](https://www.runpod.io/gpu-cloud/pricing), [Hivenet guide](https://www.hivenet.com/post/runpod-pricing-complete-guide-to-gpu-cloud-costs)
- Modal per-second rates (converted to hourly): H100 about $3.95, A100 80GB about $2.50, A100 40GB about $2.10, L40S about $1.95, A10 about $1.10, L4 about $0.80, T4 about $0.59. The Starter plan includes $30 of free compute per month. [Modal pricing](https://modal.com/pricing)
- Vast.ai marketplace: H100 about $0.90 to $1.87/hr in a mid-2026 review; RTX 4090 about $0.14 to $0.50/hr across sources. Listed prices are the cheapest offer at a given moment, not guaranteed. [fast.io review](https://www.fast.io/resources/vast-ai-review-2026.md), [Hostinger comparison](https://www.hostinger.com/tutorials/?p=158956), [tech-insider](https://tech-insider.org/runpod-vs-lambda-vs-vast-ai-2026/)
- Lambda on-demand H100: quoted between $2.99 and $3.99/hr depending on source and date. One source says Lambda offers no RTX 4090. [fast.io review](https://www.fast.io/resources/vast-ai-review-2026.md), [DeployBase](https://deploybase.ai/articles/vast-ai-vs-lambda)
- For scale reference, SSCD trained 100 epochs on 1M images at batch 4096 across multiple GPUs with synchronized batch norm, about 100M sample-steps. [SSCD](https://arxiv.org/pdf/2202.10261)

### Inferences
These are back-of-envelope estimates to replace with measured throughput in a one-hour pilot.
- Data prep (do this first, on the Mac): resize all covers to 256 px short side, JPEG quality 90, pack into WebDataset tar shards. At roughly 15 to 25 KB each that is about 50 to 80 GB instead of 250 GB. Upload once to the GPU provider's persistent volume or an object bucket. At 20 to 50 Mbit/s upstream, 60 GB takes about 3 to 7 hours. Do the resize from the external drive in one sequential pass.
- Teacher gallery embedding: a ViT-B or ViT-L forward pass at 224 px in fp16 on one H100 or 4090 should handle somewhere around 1,000 to 3,000 images per second if JPEG decoding keeps up. That puts 3.13M covers at well under an hour of GPU time per teacher, or a few dollars. The same job on the M1 over MPS is likely 10× slower or more, an overnight run per model. Fine for a single frozen baseline, painful for iteration.
- Teacher fine-tuning (SSCD-style, 1M covers × 10 to 30 epochs = 10M to 30M samples): roughly 5 to 20 GPU hours on an A100 or H100, so about $10 to $60 on RunPod or Vast community pricing. Large batch contrastive loss needs memory. If one GPU cannot hold batch 4096, use a cross-batch memory bank like the ISC winner.
- Student distillation with precomputed teacher targets (3.13M covers × 5 to 10 epochs at 224 px, a 10 to 30M student): roughly 3 to 10 hours on one RTX 4090 or A100, about $2 to $15. Expect CPU augmentation to be the bottleneck. Pick pods with 16 or more vCPUs or move geometric ops to kornia on GPU.
- M1 Mac with 64 GB is practical for: the eval harness and FAISS search over 3.13M × 256-d vectors (about 3.2 GB fp32); generating synthetic query sets; fine-tuning only a linear head on cached features; quick student ablations on 50k to 200k covers; Core ML conversion and on-device latency tests.
- Use Modal's $30/month credit for short pilot jobs such as throughput checks and teacher embedding. Use Vast or RunPod community pods for the multi-hour training runs, where Modal's hourly rate is about 2× higher.

### Gaps
- Google Colab Pro 2026 pricing and GPU availability were not checked.
- No measured M1 (MPS) throughput numbers were found for ViT-B, ConvNeXt-T or MobileCLIP2-S0. The November 2025 CLIP ViT-B/32 embedding run on prefixes 00 and 01 may have timing logs in `~/Projects/record-scan-data/_archive-mbdump-2025-11/` worth checking.
- All GPU throughput and hour estimates above are my own and unverified. Run a 1,000-step pilot to measure images per second before budgeting.

## Domain pitfalls (cross-cutting notes)

### Cited findings
- Severe crops and insertion onto unrelated images were the hardest transformations in ISC 2021, ahead of pixel-level edits. [ISC 2021 results](https://ar5iv.labs.arxiv.org/html/2202.04007)
- CLIP-style models trail DINO-family models badly on Met artwork instance recognition (SigLIP 2 g/16 13.9 GAP, EVA CLIP 0.5, DINOv3 7B 55.4). [DINOv3 Table 9](https://arxiv.org/html/2508.10104v1)
- Commercial vinyl identifier apps such as "Vinyl Identifier: Scan It" and "Vinlyo" exist on iOS. Some ask for up to four images (front, back, label, vinyl) and none describes its matching method. [App listing](https://apps.appfollow.io/ios/vinyl-identifier-scan-it/6757922322?country=at), [Vinlyo](https://apps.appfollow.io/ios/record-scan-identify-vinlyo/6757200678?country=bb)

### Inferences
- Many releases share one cover. Collapse duplicates before training, as in section 4. Exclude same-cluster items from negatives or the loss will push identical art apart, which damages the embedding. Group by MusicBrainz release group first, then merge near-duplicate art across groups (compilations and bootlegs reuse art).
- Reissues with small differences (a new logo, a "180g" banner, a remaster sticker, a changed catalog number) are genuine near-duplicates with different identities. Mine them as hard negatives only where metadata says they differ in a way the user cares about. Otherwise treat them as duplicates and let metadata pick the release.
- Text-heavy and minimalist covers (typography only, plain black, plain white, label house-style series) are the main failure slice. Mine hard negatives with the teacher's kNN across release groups so the model learns fine typography. Consider an OCR fallback (on-device text recognition matched against title and artist) for low-margin results. Treat sticker text as noise in augmentation so the model does not key on it.
- Glare and shrink-wrap: specular blobs can erase large regions. Train with glare occlusion up to about 25% of the area and evaluate the glare slice separately. Multi-frame locking lets the user tilt the record to move the glare.
- Sleeve wear and fading shift color globally. Keep color jitter wide and do not rely on color histograms.
- Square crop assumption: the detector crop may include the record edge, the inner sleeve or a hand. Train with crop errors in both directions (too tight and too loose).
- Gallery quality: Cover Art Archive front images vary (scans with borders, photos of sleeves, low-res uploads). Some gallery entries are already photo-like. Augmentation applied to them stacks distortions, so cap augmentation strength for gallery images that are not clean scans, or filter them out of the training anchors.
