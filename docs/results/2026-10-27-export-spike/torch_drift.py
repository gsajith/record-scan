"""Compare torch 2.9.1 vectors with the torch 2.8 prefix-00 embeddings on 200 covers per model."""
import numpy as np
from PIL import Image

from pipeline import config
from pipeline.retrieval.embedders import EMBEDDER_NAMES, load_embedder

root = config.require_image_root()
for name in EMBEDDER_NAMES:
    old = np.load(config.EMB_DIR / name / "00.npz")
    uuids = [str(u) for u in old["uuids"]]
    picks = list(range(0, len(uuids), 60))[:200]
    imgs = [Image.open(root / "00" / f"{uuids[i]}.jpg").convert("RGB") for i in picks]
    emb = load_embedder(name, device="mps")
    new = np.concatenate([emb.embed(imgs[i:i + 50]) for i in range(0, len(imgs), 50)])
    ref = old["embs"][picks].astype(np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    cos = np.sum(new * ref, axis=1)
    print(f"{name}\tn={len(picks)}\tmin={cos.min():.6f}\tmean={cos.mean():.6f}", flush=True)
