"""
query.py
--------
Query your album cover HNSW index using a photo (including skew/perspective correction).
Supports debug mode with preprocessing visualization (--show-preprocess).
"""

import json
import cv2
import numpy as np
from PIL import Image
import torch
import clip
import hnswlib
import argparse

# -----------------------------
# CONFIG
# -----------------------------
INDEX_PATH = "cover_hnsw.index"
META_PATH = "index_metadata.json"        # HNSW label -> UUID
UUID_TO_META_PATH = "../uuid_to_metadata.json"
ARTISTS_TO_META_PATH = "../release_slim_artists.json"
EMB_DIM = 512

# -----------------------------
# PREPROCESSING HELPERS
# -----------------------------
def robust_unskew_query_image(img_path, show=False):
    """
    Detects album cover region in a query image using contour + aspect ratio + relative area filtering,
    applies a perspective warp to get a front-on view, then center-crops and normalizes lighting.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image at {img_path}")

    img_h, img_w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]  # top 50 largest

    best_score = 0
    best_rect = None

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        print(len(approx))
        if len(approx) != 4:
            continue
        rect = approx.reshape(4,2)

        # Compute width, height, area, aspect ratio
        w = np.linalg.norm(rect[0]-rect[1])
        h = np.linalg.norm(rect[0]-rect[3])
        area = w*h
        aspect_ratio = w/h if h > 0 else 0

        # Skip tiny rectangles
        if area < 0.2 * img_w * img_h:  # at least 20% of image area
            continue

        # Score = area weighted by closeness to square
        score = area * (1.0 - abs(aspect_ratio - 1))
        if score > best_score:
            best_score = score
            best_rect = rect

    if best_rect is None:
        # fallback: center crop
        min_dim = min(img_h, img_w)
        crop = img[(img_h-min_dim)//2:(img_h+min_dim)//2, (img_w-min_dim)//2:(img_w+min_dim)//2]
        if show:
            cv2.imshow("Fallback Crop", crop)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    # Order points tl, tr, br, bl
    rect_ordered = np.zeros((4,2), dtype="float32")
    s = best_rect.sum(axis=1)
    rect_ordered[0] = best_rect[np.argmin(s)]
    rect_ordered[2] = best_rect[np.argmax(s)]
    diff = np.diff(best_rect, axis=1)
    rect_ordered[1] = best_rect[np.argmin(diff)]
    rect_ordered[3] = best_rect[np.argmax(diff)]

    tl, tr, br, bl = rect_ordered
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))

    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect_ordered, dst)
    warp = cv2.warpPerspective(img, M, (maxW, maxH))

    # Center-crop to square
    h, w = warp.shape[:2]
    min_dim = min(h, w)
    crop = warp[(h-min_dim)//2:(h+min_dim)//2, (w-min_dim)//2:(w+min_dim)//2]

    # Lighting normalization
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l,a,b))
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if show:
        cv2.imshow("Warped + Cropped + Normalized", normalized)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    img_rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# -----------------------------
# METADATA LOADING
# -----------------------------
def load_metadata():
  with open(UUID_TO_META_PATH) as f:
      return json.load(f)
def load_artists_metadata():
  with open(ARTISTS_TO_META_PATH) as f:
      return json.load(f)
# -----------------------------
# MAIN QUERY LOGIC
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Query album cover images")
    parser.add_argument("image_path", type=str, help="Path to query image")
    parser.add_argument("top_k", type=int, nargs="?", default=5, help="Number of top results")
    parser.add_argument("--show-preprocess", action="store_true", help="Show preprocessing steps")
    args = parser.parse_args()

    img_path = args.image_path
    top_k = args.top_k
    show_preprocess = args.show_preprocess

    print("🔄 Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model.eval()

    print("📂 Loading HNSW index and metadata...")
    p = hnswlib.Index(space='cosine', dim=EMB_DIM)
    p.load_index(INDEX_PATH)
    with open(META_PATH) as f:
        label_to_uuid = json.load(f)
    uuid_map = load_metadata()
    artists_map = load_artists_metadata()

    print("🧹 Preprocessing query image...")
    pil_img = robust_unskew_query_image(img_path, show=show_preprocess)
    image_input = preprocess(pil_img).unsqueeze(0)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        query_vec = image_features.cpu().numpy().astype("float32")

    print("🔍 Searching...")
    labels, distances = p.knn_query(query_vec, k=top_k)

    print(f"\nTop-{top_k} results for {img_path}:")
    for rank, (label, dist) in enumerate(zip(labels[0], distances[0])):
        uuid = label_to_uuid[str(label)]
        m = uuid_map.get(uuid, {})
        title = m.get("title", "Unknown")

        # Resolve artist names from artist_ids
        artist_ids = m.get("artist_ids", [])
        artists = []
        for aid in artist_ids:
            a = artists_map.get(aid)
            if a:
                artists.append(a.get("name", "Unknown artist"))
        artists_str = ", ".join(artists) if artists else "Unknown artist"
        print(f"{rank+1}. {title} — {artists_str} (distance: {dist:.4f})")

if __name__ == "__main__":
    main()
