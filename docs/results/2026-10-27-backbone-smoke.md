# Six more backbones, smoke run on prefix 00

Issue #18. Embedded at git SHA eea47dd; after rebasing onto main (#16's eval and its release-group changes to identify), the lookups were rerun at 10f1d07 with the same results. M1 Mac on MPS, covers read from the Seagate drive over USB. No
eval config applies: these runs check that each model works end to end, not how accurate it
is. Accuracy on phone photos is Task 7's sweep, run with #16's eval.

## Embedding from the drive

```
python -m pipeline.retrieval.embed --model NAME --prefixes 00
python -m pipeline.retrieval.build_index --model NAME
```

| Model | Dim | Input px | Covers | Seconds | Covers per second |
|---|---|---|---|---|---|
| dinov3-vits16 | 384 | 224 | 12,076 | 165 | 73 |
| dinov3-vitsplus16 | 384 | 224 | 12,076 | 147 | 82 |
| dinov3-convnext-tiny | 768 | 224 | 12,076 | 106 | 114 |
| dinov3-vitb16 | 768 | 224 | 12,076 | 176 | 69 |
| siglip2-b16-256 | 768 | 256 | 12,076 | 181 | 67 |
| pe-core-s16 | 512 | 384 | 12,076 | 237 | 51 |

The first model's run read covers off the disk. Later runs found most of them in the macOS
file cache, so the rates are not a fair speed comparison. Each run also skipped the 3,358
extra fronts (`<mbid>-<cover_id>.jpg`) that #21 added to prefix 00, as embed.py does until
Task 5 lists covers from the catalog.

## Self-match

The same three Cover Art Archive `front-500` covers as the CLIP smoke run (00159325,
0069e86a, 00bdec8d), through
`python -m pipeline.retrieval.identify FILE --model NAME --k 3`:

| Model | Top-1 correct | Top-1 scores | Next album's score |
|---|---|---|---|
| dinov3-vits16 | 3 of 3 | 1.000, 1.000, 1.000 | 0.48, 0.67, 0.54 |
| dinov3-vitsplus16 | 3 of 3 | 1.000, 1.000, 1.000 | 0.49, 0.64, 0.53 |
| dinov3-convnext-tiny | 3 of 3 | 1.000, 1.000, 1.000 | 0.57, 0.73, 0.68 |
| dinov3-vitb16 | 3 of 3 | 1.000, 1.000, 1.000 | 0.39, 0.57, 0.50 |
| siglip2-b16-256 | 3 of 3 | 1.000, 1.000, 1.000 | 0.74, 0.68, 0.60 |
| pe-core-s16 | 3 of 3 | 1.000, 1.000, 1.000 | 0.67, 0.69, 0.64 |

Finding an exact copy says the path works, nothing about recall. The gap to the next album
is not comparable across models either, since each spreads its scores differently.

## MPS against CPU

The first run of this smoke test found two bugs in torch 2.8's MPS backend:

- PE-Core aborted the process ("MPSNDArray ... buffer is not large enough"). Its attention
  pool feeds its learned query, expanded to the batch without a copy, into a Linear. A
  contiguous copy fixes it.
- DINOv3 ConvNeXt came out at cosine 0.995 to CPU on real covers, against 0.99996 on random
  noise. The fused MPS `layer_norm` is wrong on vectors with a large mean and a tiny spread,
  which flat areas of a cover produce: synthetic vectors with mean 1 and spread 0.001 are off
  by up to 0.64, and with mean 0 they match. The same formula written out on MPS matches
  float64 to 1e-7, so every embedder now runs LayerNorm that way on MPS.

After both fixes, on 200 prefix-00 covers (every 60th file), in batches of 50:

| Model | Lowest cosine, MPS vs CPU |
|---|---|
| all seven | 1.000000 |

The prefix-00 embeddings from the first run were deleted and rebuilt with the fixed code;
the numbers above are from the rebuild.
