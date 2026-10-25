# CLIP baseline smoke run, prefixes 00 and 01

Issue #15, git SHA 752b5ba, M1 Mac on MPS, covers read from the Seagate drive over USB. No
eval config applies: these runs check that the path works, not how accurate it is.

## Embedding from the drive

```
python -m pipeline.retrieval.embed --model clip-vit-b-32 --prefixes 00
python -m pipeline.retrieval.embed --model clip-vit-b-32 --prefixes 00 01   # interrupted once in 01, then rerun
python -m pipeline.retrieval.build_index --model clip-vit-b-32
```

| Prefix | Covers | Seconds | Covers per second |
|---|---|---|---|
| 00 | 12,076 | 234 | 52 |
| 01 | 12,194 | 285 | 43 |

No unreadable files. The index holds 24,270 vectors. The drive limits the rate: on the
rerun of prefix 01, covers already in the macOS file cache embedded several times faster
than covers read from the disk. At 45 per second the full 3.1M covers would take about
19 hours from the drive, which is why later embedding runs read Task 5's shards on the
internal disk.

## Self-match

Cover Art Archive `front-500` images for three prefix-00 releases (00159325, 0069e86a,
00bdec8d), run through `python -m pipeline.retrieval.identify FILE --model clip-vit-b-32`:

| Index | Top-1 correct | Top-1 scores |
|---|---|---|
| clip-vit-b-32 (this run) | 3 of 3 | 1.000, 1.000, 1.000 |
| clip-vit-b-32-aug13 (2025-11 shards, `--index`) | 3 of 3 | 0.922, 0.964, 0.925 |

The 2025-11 shards were built with open_clip's plain `ViT-B-32` config (GELU). This run
uses `ViT-B-32-quickgelu`, which matches the OpenAI weights. The old path reproduces the
2025-11 vectors at cosine 1.000; mixing the two configs costs 0.04 to 0.08 in score.

## HNSW recall on the 2025-11 index

500 random rows plus Gaussian noise (sigma 0.015, cosine about 0.95 to the source row),
seed 0, k=50, compared against brute force over 329,960 vectors:

| ef | Top-1 album recall | ms per query |
|---|---|---|
| 64 | 0.996 | 1.0 |
| 128 (default) | 0.998 | 1.9 |
| 256 | 0.998 | 2.8 |
| 1024 | 0.998 | 8.8 |

Produced by this script, run from the repo root with the pipeline venv:

```python
import numpy as np, hnswlib, time
D = "/Users/gsajith/Projects/record-scan-data/embeddings/clip-vit-b-32-aug13"
labels = np.load(D + "/labels.npy")
e = np.concatenate([np.load(f"{D}/{p}.npz", allow_pickle=True)["embs"].astype(np.float32) for p in ["00", "01"]])
e /= np.linalg.norm(e, axis=1, keepdims=True)
rng = np.random.default_rng(0)
q = e[rng.choice(len(e), 500, replace=False)] + rng.normal(0, 0.015, (500, 512)).astype(np.float32)
q /= np.linalg.norm(q, axis=1, keepdims=True)
truth = labels[np.argmax(q @ e.T, axis=1)]
idx = hnswlib.Index(space="cosine", dim=512)
idx.load_index(D + "/index.hnsw", max_elements=len(labels))
for ef in (64, 128, 256, 512, 1024):
    idx.set_ef(ef)
    t = time.perf_counter()
    ids, _ = idx.knn_query(q, k=50, num_threads=1)
    ms = (time.perf_counter() - t) / len(q) * 1000
    print(ef, np.mean(labels[ids[:, 0]] == truth), ms)
```

## One phone photo

Gautham's photo of The Dark Side of the Moon (12" vinyl, 01aa589c), shot on an iPhone and
converted from HEIC with `sips`, then run through
`python -m pipeline.retrieval.identify IMG_9772.jpg --model clip-vit-b-32 [--views 1]`:

| Views | Rank of true album | Its score | Next score |
|---|---|---|---|
| 3 | 1 | 0.836 | 0.790 (The Beatles, The Beatles) |
| 1 | 1 | 0.769 | 0.730 (Patrick R. Pärk) |

Wish You Were Here ranked third with three views. One photo against 24,270 covers says the
path works on real input, nothing about recall.
