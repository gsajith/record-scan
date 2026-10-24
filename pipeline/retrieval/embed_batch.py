"""
embed_batch.py
---------------
Generates augmented embeddings for all cover art images using CLIP (ViT-B/32).
Each image is augmented (rotations, perspective, flips) to make retrieval robust
to real-world photos. Each augmented embedding shares the same UUID.
"""

import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip
import cv2
import random

IMAGE_ROOT = "/Volumes/Seagate Portabl/record scan Pictures/record_scan_cover_images_split"
OUT_DIR = os.path.join(os.getcwd(), "embeddings")
os.makedirs(OUT_DIR, exist_ok=True)
MANIFEST = os.path.join(OUT_DIR, "manifest.json")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_name = "ViT-B-32"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
model.to(device)
model.eval()

BATCH_SIZE = 32

# --- augmentations ---
def generate_augmentations(img_bgr):
    """
    Generates realistic augmentations for flat album scans that simulate
    how a photo of the album might differ even after perspective correction.
    """

    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    aug_imgs = [img_bgr]

    # 1️⃣ Mild rotation & perspective variation (residual rectification errors)
    for angle in random.sample([-15, -10, 10, 15], 3):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        aug_imgs.append(rotated)

    for dx, dy in [(0.03*w, 0.03*h), (-0.03*w, 0.03*h)]:
        pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        pts2 = np.float32([[dx,dy],[w-dx,dy],[dx,h-dy],[w-dx,h-dy]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        aug_imgs.append(cv2.warpPerspective(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE))

    # 2️⃣ Lighting variance (simulate glare, exposure shifts)
    for img in list(aug_imgs):
        alpha = random.uniform(0.85, 1.25)  # contrast
        beta  = random.randint(-25, 25)     # brightness
        aug_imgs.append(cv2.convertScaleAbs(img, alpha=alpha, beta=beta))

    # optional glare overlay
    if random.random() < 0.4:
        glare = np.zeros_like(img_bgr, dtype=np.uint8)
        mask = np.zeros((h, w), np.uint8)
        x, y = random.randint(0, w), random.randint(0, h//2)
        cv2.ellipse(mask, (x, y),
                    (random.randint(60,150), random.randint(20,100)),
                    random.randint(0,180), 0, 360, 255, -1)
        glare[:,:,0] = glare[:,:,1] = glare[:,:,2] = mask
        aug_imgs.append(cv2.addWeighted(img_bgr, 1.0, glare, 0.4, 0))

    # 3️⃣ Slight top/bottom clipping (simulate imperfect square detection)
    if random.random() < 0.5:
        crop_frac = random.uniform(0.9, 0.95)
        crop_h = int(h * crop_frac)
        if random.random() < 0.5:
            aug_imgs.append(cv2.resize(img_bgr[0:crop_h, :], (w, h)))      # top cut
        else:
            aug_imgs.append(cv2.resize(img_bgr[h-crop_h:, :], (w, h)))    # bottom cut

    # 4️⃣ Blur and noise (camera softness, compression)
    if random.random() < 0.4:
        aug_imgs.append(cv2.GaussianBlur(img_bgr, (3,3), random.uniform(0.5,1.2)))
    if random.random() < 0.3:
        noise = np.random.normal(0, 5, img_bgr.shape).astype(np.uint8)
        aug_imgs.append(cv2.add(img_bgr, noise))

    return aug_imgs


def load_manifest():
    return json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}


def save_manifest(m):
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f)
    os.replace(tmp, MANIFEST)


def save_shard(prefix, uuids, embs):
    out_tmp = os.path.join(OUT_DIR, f"{prefix}.tmp.npz")
    out_final = os.path.join(OUT_DIR, f"{prefix}.npz")
    np.savez_compressed(out_tmp, uuids=np.array(uuids, dtype=object), embs=embs.astype(np.float16))
    os.replace(out_tmp, out_final)


def process_prefix(prefix):
    prefix_dir = os.path.join(IMAGE_ROOT, prefix)
    if not os.path.isdir(prefix_dir):
        return

    files = [f for f in os.listdir(prefix_dir) if f.endswith(".jpg")]
    manifest = load_manifest()
    if manifest.get(prefix, {}).get("done"):
        print(f"[✓] {prefix} already processed")
        return

    uuids, embs_list = [], []
    batch_imgs, batch_ids = [], []

    for fname in tqdm(files, desc=f"Prefix {prefix}"):
        uuid = os.path.splitext(fname)[0]
        if manifest.get(prefix, {}).get("processed", {}).get(uuid):
            continue

        fpath = os.path.join(prefix_dir, fname)
        try:
            pil = Image.open(fpath).convert("RGB")
        except Exception:
            manifest.setdefault(prefix, {}).setdefault("processed", {})[uuid] = "err"
            save_manifest(manifest)
            continue

        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        patch = cv2.resize(img_bgr, (224, 224))

        # --- generate augmentations ---
        for aug in generate_augmentations(patch):
            pil2 = Image.fromarray(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB))
            tensor = preprocess(pil2)
            batch_imgs.append(tensor)
            batch_ids.append(uuid)

        if len(batch_imgs) >= BATCH_SIZE:
            batch_tensor = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                emb = model.encode_image(batch_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                emb_np = emb.cpu().numpy().astype(np.float32)
            for uid, e in zip(batch_ids, emb_np):
                uuids.append(uid)
                embs_list.append(e)
            save_shard(prefix, uuids, np.stack(embs_list))
            batch_imgs, batch_ids = [], []

        manifest.setdefault(prefix, {}).setdefault("processed", {})[uuid] = "ok"
        save_manifest(manifest)

    # final flush
    if batch_imgs:
        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            emb = model.encode_image(batch_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb_np = emb.cpu().numpy().astype(np.float32)
        for uid, e in zip(batch_ids, emb_np):
            uuids.append(uid)
            embs_list.append(e)

    if embs_list:
        save_shard(prefix, uuids, np.stack(embs_list))

    manifest.setdefault(prefix, {})["done"] = True
    save_manifest(manifest)
    print(f"[✓] Finished prefix {prefix}")


def main():
    prefixes = sorted([d for d in os.listdir(IMAGE_ROOT) if len(d) == 2])
    prefixes = ['00', '01']  # for testing, process only two prefixes
    os.makedirs(OUT_DIR, exist_ok=True)
    for prefix in prefixes:
        process_prefix(prefix)


if __name__ == "__main__":
    main()
