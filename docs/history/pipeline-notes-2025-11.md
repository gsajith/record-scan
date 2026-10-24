# Record Retrieval Project Notes

### Overview

This repo powers a large-scale album cover retrieval system using embeddings generated from Cover Art Archive images.

### Dataset

- ~3.2 million JPG images (~100 KB each)
- Directory layout: `/Volumes/Seagate Portabl/record scan Pictures/record_scan_cover_images_split/<prefix>/<uuid>.jpg`
- `<prefix>` = first 2 characters of each UUID (to prevent filesystem slowdown)
- UUID → album metadata stored separately in `uuid_to_metadata.json`

### Environment

- Running locally on M1 Mac with 64 GB RAM, no discrete GPU
- No external APIs or cloud services
- Local embedding + search only

### Model

- `open-clip` model: **ViT-B/32**, pretrained on OpenAI CLIP weights
- Output dimension: 512
- Normalization: L2 per vector
- Embeddings stored as float16 (≈1 KB per image → ~3 GB total)

### Embedding Pipeline

- Script: `embed_batch.py`
- Processes one two-character prefix directory at a time
- Resumable: `manifest.json` tracks progress (per prefix, per file)
- Each shard written as `<prefix>.npz` containing:
  - `uuids`: list of UUID strings
  - `embs`: float16 normalized vectors

### Indexing

- Script: `build_index.py`
- Uses **hnswlib** for approximate nearest neighbor (ANN) search
- Parameters:
  - space: cosine
  - dim: 512
  - M: 48
  - ef_construction: 200
- Output:
  - `cover_hnsw.index` → ANN index
  - `index_metadata.json` → index ID → UUID mapping

### Query

- Script: `query.py`
- Loads index + metadata
- Given an image path:
  - Preprocess (perspective correction)
  - Generate embedding
  - Query hnswlib for top-K results
  - Return UUIDs and album metadata

### Helper Scripts

#### resume_failed_prefixes.py

Automatically reprocesses any incomplete or failed prefix directories from `manifest.json`.
Use it after a crash, disconnect, or partial run. Only to be used in rare cases of failures.

#### verify_embeddings.py

Sanity-checks `.npz` shards to confirm integrity before building the index.
Ensures all arrays have consistent shape and dimension (512).

### Storage Summary

| Component  | Count  | Size (approx) | Notes             |
| ---------- | ------ | ------------- | ----------------- |
| Images     | 3.2 M  | 250 GB        | External drive    |
| Embeddings | 3.2 M  | ~3 GB         | Float16           |
| Index      | 3.2 M  | ~10–15 GB     | hnswlib graph     |
| Metadata   | 1 JSON | <1 GB         | UUID → album info |

### Decisions / Tradeoffs

- ✅ Open-source, offline only
- ✅ CLIP (ViT-B/32) chosen for speed vs accuracy balance
- ✅ hnswlib (simple, good recall, easy persistence)
- 🚫 No FAISS/Pinecone yet — can be swapped later
- ✅ float16 on disk, float32 in memory
- ✅ Sharded by prefix for parallel safety and resumability

### Next Steps

1. **Test small batch** (e.g. 1000 images) to verify embeddings + retrieval.
2. Adjust Canny / area thresholds in `detect_and_warp_cover()` if poor alignment.
3. After embedding all shards, run `build_index.py`.
4. Validate retrieval results with sample photos.
5. Add:
   - Video frame sampler (OpenCV, 1 fps or motion-based)
   - Optional re-ranker or fine-tuned model later
   - Simple Flask/Streamlit UI to visualize matches
6. Consider compressing index or migrating to FAISS IVF+PQ when serving remotely.

---

**Author decisions summary**

- OUT_DIR: current working directory
- Device: MPS (Apple GPU) or CPU fallback
- Batch size: 32
- All runs resumable via `manifest.json`

This file exists for Cursor/Copilot to understand:

- How the system is structured
- Which scripts serve which roles
- What assumptions and defaults have been made
- What logical next steps are in the roadmap
