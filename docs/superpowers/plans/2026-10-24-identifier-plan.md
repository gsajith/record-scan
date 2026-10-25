# Record Scan identifier plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Dispatch implementers and reviewers with model `opus` (Gautham's standing preference).

**Goal:** Build the record identifier the research report specifies: one vector per deduplicated cover, a query encoder trained on simulated phone photos, a phone-sized student, served hybrid first and on-device if a compressed index proves itself. This plan supersedes Tasks 4 to 13 of `docs/superpowers/plans/2026-10-24-record-scan-recovery.md`; Tasks 1 to 3 of that plan are done and its Parts A to F remain the project inventory.

**Architecture:** Phase A builds the measuring instruments (labeler, gold and wild sets, catalog search, retrieval core, gallery pass, catalog update job, export spike) and runs the frozen-backbone sweep. Nothing in Phase A trains a model. Phases B to D (hybrid v1, simulator, head, fine-tune, distillation, index compression, lock calibration) are gated on Phase A's numbers and are outlined here; their tasks get written at full detail after Step 4's measurements exist, because the model and the dimensions they depend on are unknown until then.

**Tech stack:** Python 3.12, PyTorch 2.8 (MPS), Hugging Face transformers and timm for backbones, open_clip for the CLIP baseline, hnswlib and faiss-cpu, WebDataset shards, SQLite FTS5, imagehash, ffmpeg. Next.js 16 (the `annotator/` app, repurposed as the labeler) with better-sqlite3 and zod. coremltools 9 and ExecuTorch 1.x for the export spike.

**Spec:** `docs/research/reports/On device album cover retrieval 2026.md` (the twelve gated steps and the evidence behind them) plus the research notes beside it. Decisions D8 to D12 below extend the recovery plan's D1 to D7. Where this plan and the report differ, this plan wins; it records why.

## Global constraints

- Everything from the recovery plan's Global constraints still applies: never `git add` data, agents delete nothing but regenerable artifacts, Conventional Commits, one venv at `pipeline/.venv`, `require_image_root()` before touching the drive, never `os.makedirs` under `/Volumes`, TypeScript without `any`.
- Data layout under `RS_DATA_DIR` (default `~/Projects/record-scan-data`): `metadata/` (JSON, `catalog.sqlite`), `gold/` and `wild/` (each with `photos/`, `inbox/`, `labels.jsonl`), `gallery256/` (shards, hashes, clusters), `delta/` (covers and metadata fetched since the dump), `embeddings/<model>/`, `exports/`, `lightbank/`. Config exposes each as a constant.
- Pipeline modules read `config.X` at call time, never `from pipeline.config import X` at import. Tests generate fixtures into `tmp_path`; no JSON or image fixtures are committed except tiny generated ones under `pipeline/tests/fixtures/` created by `conftest.py`.
- One label record format, defined once in `pipeline/gold/labels.py` and mirrored in `annotator/lib/labels.ts`, with a shared test vector so the two never drift.
- Gold photos and the `eval` half of the wild set never enter any training, simulator fitting, or background bank. The split is a deterministic hash, assigned once, shown read-only.
- External services: Cover Art Archive thumbnails may be hot-linked by the labeler for display only. The MusicBrainz web service gets at most one request per second with a `User-Agent` naming the project and a contact address. Internet Archive search uses the scrape API with cursor resume.
- Every measured number lands in `docs/results/<date>-<topic>.md` with the command that produced it, so the report's decision gates are checked against written evidence.

## Review focus

1. Two labels for one image: the labeler appends a new line and the loader keeps the latest line per `sha1`; a re-label never silently keeps the old answer. Test in Task 2.
2. Split drift: the Python and TypeScript `split_for(sha1)` return the same answer for the same input; a shared vector of five sha1 values is asserted in both suites. Tests in Tasks 2 and 3.
3. Catalog search misses on accents and case: "bjork" finds Björk and "the beatles" finds "Beatles, The"; FTS5 is built with `unicode61 remove_diacritics 2`. Test in Task 1.
4. A corrupt JPEG or an unmounted drive mid-run: the gallery pass logs and skips the file, finishes the prefix, and resumes per prefix on rerun; it never creates folders under `/Volumes`. Test in Task 5.
5. Rate-limit violations: the delta fetcher sleeps at least one second between MusicBrainz requests, backs off on 503, and resumes from its cursor file after a crash. Test in Task 6.

---

## Part A: decisions added by this plan

- **D8 Labeler is local-only.** The `annotator/` app runs on the Mac, writes files under `RS_DATA_DIR`, and has no cloud backend. Couch use goes through Tailscale. Supabase code is removed. (Gautham, 2026-10-24.)
- **D9 Candid internet photos are allowed for evaluation and simulator fitting, with a license filter and a one-time split.** Sources: Flickr Creative Commons, Wikimedia Commons, Pexels, Unsplash, Creative Commons YouTube frames. No Instagram, nothing unlicensed. Half (by hash) feeds simulator statistics and backgrounds; half is evaluation. Nothing from either half is used to train weights unless its license is CC0 or equivalent and Gautham says so.
- **D10 Gallery working copy at 256 px short side, JPEG quality 90, in WebDataset tar shards keyed by MBID.** One pass over the drive produces it together with a 64-bit pHash per cover. Later steps read the shards, not the drive.
- **D11 Catalog updates come from the Cover Art Archive's Internet Archive index, not from a new dump.** A monthly job lists items added or updated since the last run, downloads new front covers, and fetches metadata from the MusicBrainz web service. A full dump re-slim stays an annual job and is where the release-group id gets added.
- **D12 Mobile shell framework: proposed Expo (React Native) with VisionCamera frame processors and `react-native-executorch` for one codebase on both platforms; native Swift and Kotlin if the export spike shows the React Native path cannot hit the latency bar.** Not needed in Phase A: iOS device timing comes from Xcode's Core ML performance report on a connected iPhone, which needs no app. Gautham confirms or overrides before Phase B's Task 12.

## Part B: human tasks (only Gautham can do these)

- **H1** Mount the Seagate drive before Tasks 5, 8 (distractors) and anything in Phase B.
- **H2** Gold set, revised from the recovery plan's H2: 60 to 100 of your own records, a ten-second hand-held video of each (varying angle, distance, light, with and without your hand on it), plus 100 to 200 negatives (walls, shelves of spines, books, CDs, screens) and a few records you know are missing from Cover Art Archive. Drop the videos into `~/Projects/record-scan-data/gold/inbox/`; Task 2's sampler turns them into frames and Task 3's labeler labels them.
- **H3** Light bank and geometry target. (a) A glossy white 12-inch square (a blank glossy sleeve, or glossy photo paper mounted on board; matte paper shows shadows and color cast but no specular glare, and real sleeves are mostly glossy laminate). Photograph it 100 to 200 times in every place records live for you: shelf, desk lamp, window light, dark room, colored bulb, phone flash, a record store. Shoot half the frames with the card inside a clear plastic outer sleeve, since store copies are often wrapped. (b) One geometry target: a printed cover with ArUco markers in the four corners, shot 50 times at the tilts and distances you would really hold a phone, which gives ground-truth corners for Step 3. (c) Optional and cheap: 10 to 20 real catalog covers printed on glossy paper at 12 inches, a "printed" set that can be photographed in any environment with a known answer; it is scored as its own set, never mixed into gold. Drop everything into `~/Projects/record-scan-data/lightbank/inbox/`.
- **H4** Wild set: collect 300 to 1,000 licensed candid photos of people holding or flipping through records (see D9) by pasting or dropping them into the labeler, with source URL and license. The labeler refuses a wild image without both.
- **H5** Accept the DINOv3 license on Hugging Face (`facebook/dinov3-*` are gated), run `huggingface-cli login` on the Mac, and put the token in the pipeline `.env` as `HF_TOKEN`. Also read Apple's MobileCLIP2 model terms and say yes or no to including it in the sweep.
- **H6** Confirm or override D12 before Phase B.
- **H7** Before Phase B: an account on one GPU provider (RunPod or Modal) with a small balance; the whole plan budgets under $100.

## Part C: layout added by this plan

```
record-scan/
  pipeline/
    config.py                      + CATALOG_DB, WILD_DIR, GALLERY256_DIR, DELTA_DIR, EXPORTS_DIR, LIGHTBANK_DIR, RESULTS_DIR
    metadata/
      catalog.py                   SQLite-backed Catalog: release(), search(), album_key(), find()
      build_catalog_db.py          JSON + delta -> catalog.sqlite (FTS5)
      fetch_new_covers.py          Internet Archive delta -> covers + metadata (Task 6)
    gold/
      labels.py                    Label schema, split_for(), load_labels()
      sample_frames.py             ffmpeg video -> frames into inbox/
    gallery/
      shard_and_hash.py            drive -> gallery256 shards + phash (Task 5)
      cluster_duplicates.py        phash -> clusters.parquet
    retrieval/
      embedders.py                 7 backbones behind one Protocol, query_views()
      embed.py                     shards or uuid lists -> npz
      build_index.py, identify.py  hnsw index, album-level identify
      eval.py                      gold/wild eval with lenient hits, slices, negatives
      sweep.py                     Step 2 runner -> docs/results/
    export/
      to_coreml.py, to_executorch.py, parity.py   Task 8
    tests/
  annotator/                       the labeler (Task 3), Supabase removed
  docs/results/                    measured numbers, one file per run
```

---

## Phase A: instruments and the frozen sweep (Steps 0 to 4, plus the catalog update)

Dependency order: Task 1 -> Task 2 -> Task 3; Task 4 after Task 1; Task 5 and Task 6 after Task 1 and independent of each other; Task 7 after Tasks 2, 4 and 5; Task 8 after Task 4; Task 9 after Task 7. Tasks 3, 4, 5, 6 can run as parallel workers once Tasks 1 and 2 land, since they touch disjoint directories.

### Task 1: catalog in SQLite with full-text search

**Files:**
- Create: `pipeline/metadata/catalog.py`, `pipeline/metadata/build_catalog_db.py`, `pipeline/tests/test_catalog.py`
- Modify: `pipeline/config.py` (new constants listed in Part C), `pipeline/requirements.txt` (no new deps; sqlite3 is stdlib)

**Interfaces (produces):**
```python
# pipeline/metadata/catalog.py
@dataclass(frozen=True)
class Release: mbid: str; title: str; artists: tuple[str, ...]; format: str | None; cover_id: str | None; release_group: str | None
def normalize(s: str) -> str          # casefold, strip accents (NFKD), drop punctuation, collapse spaces, move a trailing ", the" and drop a leading "the "
class Catalog:
    def __init__(self, db_path: Path | None = None): ...           # default config.CATALOG_DB; raises SystemExit with a hint to run build_catalog_db if missing
    def release(self, mbid: str) -> Release | None
    def search(self, query: str, limit: int = 20) -> list[Release]  # FTS5 bm25 over title + artist names, prefix matching on the last token
    def album_key(self, mbid: str) -> str                           # normalize(title) + "|" + normalize(first artist)
    def find(self, artist: str, title: str) -> list[str]            # exact normalized match, used by eval
    def count(self) -> int

# pipeline/metadata/build_catalog_db.py   CLI: python -m pipeline.metadata.build_catalog_db [--out PATH] [--delta DIR]
def build(releases_json: Path, artists_json: Path, out: Path, delta_dir: Path | None) -> int
    # schema: releases(mbid PK, title, primary_artist, artist_ids_json, format, cover_id, release_group) ; artists(id PK, name, aliases_json)
    #         release_fts USING fts5(mbid UNINDEXED, title, artists, tokenize="unicode61 remove_diacritics 2")
    # returns row count; writes to out + ".tmp" then os.replace
```
Load `slim_releases_with_covers.json` and `release_slim_artists.json` whole (about 6 GB of RAM, fine on the Mac), insert in one transaction with `executemany`, then build the FTS table. Also ingest `delta/releases/*.jsonl` when present (Task 6's output, which carries `release_group`).

- [ ] Step 1: Failing tests: `test_normalize` ("The Beatles" -> "beatles"; "Beatles, The" -> "beatles"; "Björk" -> "bjork"; "Abbey Road (Remastered)" keeps its words), `test_build_and_search_tmp_db` (three fake releases written to `tmp_path` JSON, build, `search("bjork")` returns the Björk release, `search("beatl")` prefix-matches, `release(mbid)` round-trips), `test_find_exact`, `test_missing_db_exits`.
- [ ] Step 2: Run, FAIL.
- [ ] Step 3: Implement. Add the new config constants.
- [ ] Step 4: `pipeline/.venv/bin/pytest pipeline/tests/test_catalog.py -v` passes. Real build: `python -m pipeline.metadata.build_catalog_db` writes `catalog.sqlite` (expect 3,129,812 rows, 600 MB to 1 GB, under 10 minutes); `python -c "from pipeline.metadata.catalog import Catalog; print(Catalog().search('abbey road')[:3])"` returns Beatles releases.
- [ ] Step 5: Commit `feat(pipeline): sqlite catalog with full-text search`.

### Task 2: gold and wild set format, loader, and video frame sampler

**Files:**
- Create: `pipeline/gold/__init__.py`, `pipeline/gold/labels.py`, `pipeline/gold/sample_frames.py`, `pipeline/tests/test_labels.py`, `pipeline/tests/test_sample_frames.py`, `pipeline/tests/fixtures/split_vectors.json` (generated by conftest: five sha1 strings with their expected split)

**Interfaces (produces):**
```python
# pipeline/gold/labels.py
KINDS = ("positive", "negative", "unknown")
TAGS = ("tilt", "glare", "dark", "colored-light", "hand", "shrink-wrap", "text-only", "rack", "blur", "partial", "crate", "flash")   # "partial": any cut-off; "crate": the bottom hidden by the sleeve in front, the crate-digging case
LICENSES = ("own", "cc0", "cc-by", "cc-by-sa", "pexels", "unsplash", "other-permitted")
@dataclass
class Label: file: str; sha1: str; kind: str; mbid: str | None; tags: list[str]; source_url: str | None; license: str; split: str; notes: str; labeled_at: str
def split_for(set_name: str, sha1: str) -> str      # "gold" -> "eval"; "wild" -> "sim" if int(sha1[:8], 16) % 2 == 0 else "eval"
def load_labels(set_dir: Path) -> list[Label]      # reads set_dir/labels.jsonl, keeps the LAST line per sha1, validates kinds/tags/licenses, resolves file paths relative to set_dir
def append_label(set_dir: Path, label: Label) -> None   # append one JSON line, fsync

# pipeline/gold/sample_frames.py   CLI: python -m pipeline.gold.sample_frames SET [--fps 1.0] [--max-frames 8]
def sample_frames(video: Path, out_dir: Path, fps: float, max_frames: int) -> list[Path]   # ffmpeg -i video -vf fps=FPS -frames:v N out_dir/<video-stem>-%02d.jpg ; returns written paths
def process_inbox(set_dir: Path, fps: float, max_frames: int) -> int   # every .mp4/.mov in set_dir/inbox -> frames in set_dir/inbox (videos moved to set_dir/inbox/done/) ; returns frame count
```
JSONL record keys are exactly the `Label` field names. `labeled_at` is ISO 8601 UTC.

- [ ] Step 1: Failing tests: `test_split_vectors` (loads the fixture vector and asserts `split_for` on each), `test_load_keeps_latest_label` (two lines for one sha1, second wins), `test_load_rejects_unknown_tag`, `test_sample_frames_writes_n_frames` (conftest generates a 3-second synthetic video with ffmpeg if `ffmpeg` is on PATH, else the test skips with a clear reason).
- [ ] Step 2: Run, FAIL.
- [ ] Step 3: Implement.
- [ ] Step 4: Tests pass. `which ffmpeg` is present (if not, `brew install ffmpeg` is the one allowed install in this task; record it in the report).
- [ ] Step 5: Commit `feat(pipeline): gold and wild label format with video frame sampling`.

### Task 3: the labeler (annotator/ repurposed, local-only)

**Files:**
- Delete: `annotator/lib/supabaseClient.ts`, `annotator/app/api/upload/route.ts`, `annotator/app/api/save/route.ts`
- Create: `annotator/lib/labels.ts`, `annotator/lib/data.ts` (paths under `RS_DATA_DIR`), `annotator/lib/catalog.ts` (better-sqlite3 read-only connection), `annotator/app/api/search/route.ts`, `annotator/app/api/ingest/route.ts`, `annotator/app/api/label/route.ts`, `annotator/app/api/queue/route.ts`, `annotator/app/api/image/[set]/[sha1]/route.ts`, `annotator/lib/labels.test.ts`
- Modify: `annotator/app/page.tsx` (new UI), `annotator/package.json` (add `better-sqlite3`, `zod`, `vitest`; remove `@supabase/supabase-js`), `annotator/README.md`, `.env.example` (add `RS_DATA_DIR`)

**Interfaces (produces):**
```ts
// lib/labels.ts  (mirror of pipeline/gold/labels.py)
export const KINDS, TAGS, LICENSES as const  // identical lists
export const LabelSchema = z.object({ file, sha1, kind, mbid: z.string().uuid().nullable(), tags: z.array(z.enum(TAGS)), source_url: z.string().url().nullable(), license: z.enum(LICENSES), split: z.enum(["eval","sim"]), notes: z.string(), labeled_at: z.string() })
export function splitFor(set: "gold" | "wild", sha1: string): "eval" | "sim"   // same rule as Python
// API
GET  /api/search?q=...&limit=20          -> { results: [{ mbid, title, artists, format, year?, thumb: "https://coverartarchive.org/release/<mbid>/front-250" }] }
POST /api/ingest   multipart { set, file } | json { set, url, source_url, license }   -> { sha1, split, file }   // saves <set>/photos/<sha1>.jpg (re-encodes to JPEG, max side 1600), refuses a wild image without source_url + license, returns existing record if sha1 already present
GET  /api/queue?set=gold                  -> { next: { file, sha1 } | null, remaining }   // unlabeled images in <set>/inbox/ and photos/ without a label line
POST /api/label    json Label             -> { ok: true }   // validates with LabelSchema, split recomputed server-side, appends to <set>/labels.jsonl
GET  /api/image/<set>/<sha1>              -> the JPEG (so the page can show local files)
```
Page flow: pick set (gold or wild); paste (Ctrl+V image or URL), drop files or URLs, or press N for the next inbox item; image fills the left pane; right pane has the search box (debounced, results with thumbnails, arrow keys to move, Enter to pick), a kind toggle (positive, negative, unknown) with hotkeys P N U, tag checkboxes with hotkeys 1 to 9 and 0, license select and source field (prefilled from the drop when the browser provides it), notes, and Save (Enter). After Save the next item loads. A status strip shows counts per set and per split.

- [ ] Step 1: Failing vitest cases: `splitFor` against the same five-vector fixture Task 2 generated (copy the JSON into the test, or read it from `../pipeline/tests/fixtures/split_vectors.json`), `LabelSchema` rejects a wild label without license, accepts a gold label with `license: "own"`.
- [ ] Step 2: Run, FAIL. Then implement routes and page; `npm run lint && npx tsc --noEmit && npx vitest run` clean.
- [ ] Step 3: Manual check with `RS_DATA_DIR` pointing at a tmp folder: paste a cover image from the clipboard, search "abbey road", pick a release, save; the JSONL line validates with Python `load_labels`; a second label for the same image produces a second line and `load_labels` returns the newer one.
- [ ] Step 4: `annotator/README.md` rewritten for the labeler: how to run, hotkeys, where files go, the license rule. Remove the Supabase sections and the stale `../pipeline/detector/export_yolo_dataset.py` path (that script now lives in `pipeline/legacy/`). Also put double quotes around the `RS_IMAGE_ROOT` value in the root `.env.example` so the file loads with dotenv parsers and with `source` alike.
- [ ] Step 5: Commit `feat(annotator): local labeler for gold and wild photo sets`.

### Task 4: retrieval core (embedders, index, identify, eval)

This merges the recovery plan's Tasks 4, 5 and 7 with the report's changes: seven backbones, one vector per image, three query views, lenient album-level scoring, per-slice recall, negatives.

**Files:**
- Create: `pipeline/retrieval/embedders.py`, `pipeline/retrieval/embed.py`, `pipeline/retrieval/identify.py`, `pipeline/retrieval/eval.py`, `pipeline/tests/test_embedders.py`, `pipeline/tests/test_embed.py`, `pipeline/tests/test_index_identify.py`, `pipeline/tests/test_eval.py`
- Modify: `pipeline/retrieval/build_index.py` (rewrite), `pipeline/requirements.txt` (+ `transformers==4.55.4`, `imagehash`, `webdataset`, `pyarrow`, `faiss-cpu==1.12.0`, `huggingface_hub`)
- Delete: `pipeline/retrieval/embed_batch.py`, `pipeline/retrieval/query.py`

**Interfaces (produces):**
```python
# embedders.py
EMBEDDER_NAMES = ("clip-vit-b-32", "dinov3-vits16", "dinov3-vitsplus16", "dinov3-convnext-tiny", "dinov3-vitb16", "siglip2-b16-256", "pe-core-s16")
# sources, verify each on its model card before use: open_clip ViT-B-32/openai; facebook/dinov3-vits16-pretrain-lvd1689m, facebook/dinov3-vits16plus-pretrain-lvd1689m, facebook/dinov3-convnext-tiny-pretrain-lvd1689m, facebook/dinov3-vitb16-pretrain-lvd1689m (gated, HF_TOKEN); google/siglip2-base-patch16-256; PE-Core S/16 via timm or facebook/PE-Core-S16-384
class Embedder(Protocol):
    name: str; dim: int; input_px: int
    def preprocess(self, img: Image.Image) -> torch.Tensor
    def embed(self, images: list[Image.Image]) -> np.ndarray       # (n, dim) float32, L2-normalized; CLS token for ViTs (option pool="cls+mean"), pooled features for ConvNeXt
def load_embedder(name: str, device: str | None = None, input_px: int | None = None) -> Embedder
def query_views(img: Image.Image) -> list[Image.Image]            # [center square crop, 10% tighter crop, gray-padded so the object fills 70%]; the report's three scales

# embed.py   CLI: python -m pipeline.retrieval.embed --model NAME (--prefixes 00 01 | --all | --uuids FILE) [--source drive|shards] [--batch 64] [--workers 6] [--px 224]
def embed_images(paths: list[Path], embedder: Embedder, batch: int, workers: int) -> tuple[list[str], np.ndarray]   # keyed by file stem (mbid); unreadable files logged to <out>/errors.log and skipped
def embed_prefix(prefix: str, embedder: Embedder, out_dir: Path, source: str, batch: int, workers: int) -> int        # writes out_dir/<prefix>.npz once (uuids U36, embs float16) and <prefix>.done
def pending_prefixes(out_dir: Path, source: str) -> list[str]

# build_index.py   CLI: python -m pipeline.retrieval.build_index --model NAME [--M 32] [--ef-construction 200]
def build_index(shards: list[Path], dim: int, out_dir: Path, M: int, ef_construction: int) -> int   # verifies dims; writes index.hnsw + labels.npy (U36)

# identify.py   CLI: python -m pipeline.retrieval.identify photo.jpg --model NAME [--k 5] [--views 1|3]
@dataclass
class Match: mbid: str; title: str; artists: tuple[str, ...]; score: float
class Identifier:
    def __init__(self, embedder: Embedder, index_dir: Path, catalog: Catalog, clusters: dict[str, int] | None = None, ef: int = 128): ...
    def identify(self, img: Image.Image, k: int = 5, views: int = 3) -> list[Match]
        # embed each view, knn k*10 per view, max score per cluster (cluster = clusters[mbid] if given else album_key), top k

# eval.py   CLI: python -m pipeline.retrieval.eval --model NAME --set gold|wild [--distractors 50000] [--views 1|3] [--px 224] [--seed 0] [--dry-run]
@dataclass
class EvalItem: label: Label; targets: set[str]   # target mbids: the labeled mbid plus every mbid sharing its album_key, plus its cluster members when clusters exist
def load_eval_set(set_dir: Path, catalog: Catalog, clusters: dict[str, int] | None) -> list[EvalItem]   # eval split only; positives need mbid; warns on unresolvable
def is_hit(result_mbid: str, item: EvalItem) -> bool
def build_mini_gallery(items: list[EvalItem], distractors: int, seed: int) -> list[str]   # all target mbids + seeded sample from existing_files.json
@dataclass
class EvalReport:
    r1: float; r5: float; per_tag: dict[str, tuple[float, int]]; neg_max: float; pos_min_top1: float; micro_ap: float; rows: list[dict]
    def to_markdown(self) -> str
def run_eval(model: str, set_name: str, distractors: int, views: int, px: int, seed: int, work_dir: Path) -> EvalReport   # writes docs/results/<date>-eval-<model>-<set>.md and a .json beside it
```
Per-tag recall is reported with its count so a slice of three photos is visibly weak. `micro_ap` follows SSCD: positives and negatives ranked by top-1 score, average precision of "is a positive and hit".

- [ ] Step 1: Failing tests: `test_each_embedder_unit_vectors` (parametrized, `@pytest.mark.slow`, two fixture images, shape and norm; skips gated models when `HF_TOKEN` is unset with a clear message), `test_query_views_three_sizes`, `test_embed_prefix_skips_corrupt_and_resumes`, `test_build_index_and_identify_roundtrip` (synthetic vectors, duplicates collapse by album_key), `test_is_hit_rules`, `test_eval_report_metrics_and_slices` (hand-built rows), `test_load_eval_set_uses_eval_split_only`.
- [ ] Step 2: Run, FAIL. Step 3: Implement. Step 4: all pass; smoke: build an index from the 2025-11 shards (`clip-vit-b-32-aug13`, allow_pickle, cast uuids) and identify a CAA `front-500` for a uuid in prefix 00.
- [ ] Step 5: `git rm` the two old scripts. Commit `feat(pipeline): retrieval core with seven backbones, query views and slice eval`.

### Task 5: the gallery pass (Step 1): shards, hashes, duplicate clusters

**Files:**
- Create: `pipeline/gallery/__init__.py`, `pipeline/gallery/shard_and_hash.py`, `pipeline/gallery/cluster_duplicates.py`, `pipeline/tests/test_gallery.py`

**Interfaces (produces):**
```python
# shard_and_hash.py   CLI: python -m pipeline.gallery.shard_and_hash [--prefixes ...|--all] [--workers 6]
def process_prefix(prefix: str, out_dir: Path, workers: int) -> tuple[int, int]   # reads IMAGE_ROOT/<prefix>/*.jpg sequentially; writes out_dir/<prefix>.tar (WebDataset: <mbid>.jpg at 256 px short side, q90) and out_dir/<prefix>.phash.npy (structured: mbid U36, phash uint64); touches <prefix>.done; returns (ok, skipped)
# cluster_duplicates.py   CLI: python -m pipeline.gallery.cluster_duplicates [--max-distance 8]
def cluster(phash_files: list[Path], max_distance: int) -> dict[str, int]   # exact-hash union, then Hamming <= max_distance via 4-way multi-index hashing; writes out_dir/clusters.parquet (mbid, cluster_id, cluster_size) and prints N, cluster count, size histogram
```
Both call `config.require_image_root()` first. Time the first prefix and print an ETA for the rest; expect hours at USB-3 speeds. Write to `config.GALLERY256_DIR` on the internal disk, never to the drive.

- [ ] Step 1: Failing tests on a tmp image root with three prefixes, one corrupt file, two identical covers under different mbids and one near-duplicate: shard contains the good files at 256 px, errors logged, `.done` resumes, clusters group the identical pair and the near-duplicate at distance <= 8 and not an unrelated image.
- [ ] Step 2: Run, FAIL. Step 3: Implement. Step 4: tests pass. Real run after H1: `--prefixes 00` first, report time and sizes in `docs/results/<date>-gallery-pass.md`, then `--all` under `nohup caffeinate -i`, then `cluster_duplicates`. Record N (deduplicated), cluster count, and the size histogram in the same results file.
- [ ] Step 5: Commit `feat(pipeline): gallery pass with 256px shards, phash and duplicate clusters`.

### Task 6: new covers since the dump (D11)

**Files:**
- Create: `pipeline/metadata/fetch_new_covers.py`, `pipeline/tests/test_fetch_new_covers.py`
- Modify: `pipeline/metadata/build_catalog_db.py` (already ingests `delta/releases/*.jsonl` per Task 1; confirm the `release_group` column fills)

**Interfaces (produces):**
```python
# fetch_new_covers.py   CLI: python -m pipeline.metadata.fetch_new_covers --since 2025-08-13 [--until DATE] [--limit N] [--dry-run]
def list_new_items(since: str, until: str | None, cursor_file: Path) -> Iterator[dict]   # Internet Archive scrape API, q='collection:coverartarchive AND (addeddate:[SINCE TO UNTIL] OR updatedate:[SINCE TO UNTIL])', fields identifier,addeddate,updatedate; cursor saved after each page; yields {mbid, added, updated}
def front_image_id(mbid: str) -> str | None                      # GET https://coverartarchive.org/release/<mbid>; same priority as parallel_fetch_cover_art_urls
def download_front(mbid: str, cover_id: str, out_dir: Path) -> Path   # <out_dir>/<mbid[:2]>/<mbid>.jpg (500 px); skip if present
def fetch_release(mbid: str, session, last_call: float) -> dict  # MB WS /ws/2/release/<mbid>?inc=artist-credits+release-groups&fmt=json ; sleeps to keep >= 1.0 s between calls; User-Agent "RecordScan/0.1 (gsajith@umich.edu)"; backoff 503 -> 5 s doubling to 60 s
def slim(release_json: dict) -> dict                              # same keys as slim_file.py plus "release_group" and "cover_id"
def main(...) -> None                                             # writes delta/covers/<prefix>/<mbid>.jpg, delta/releases/<date>.jsonl, delta/artists/<date>.jsonl, delta/state.json (cursor, counts); resumable; --dry-run prints the item count only
```
Expect a backlog in the hundreds of thousands; at one MusicBrainz call per second that is days of background running, which is fine. Print a throughput line every 1,000 items.

- [ ] Step 1: Failing tests with recorded fixture JSON (generated by conftest, not fetched): `test_scrape_cursor_resume` (two fake pages, cursor file written between them), `test_front_image_priority`, `test_slim_keeps_release_group`, `test_rate_limit_sleeps` (monkeypatched clock, asserts >= 1.0 s spacing), `test_backoff_on_503`.
- [ ] Step 2: Run, FAIL. Step 3: Implement. Step 4: tests pass. Live: `--dry-run` prints the real backlog count (record it in `docs/results/<date>-catalog-delta.md`); then `--limit 500` end to end; then start the full run under `nohup`. Rebuild the catalog with `build_catalog_db` once the first batch lands and confirm `release_group` is populated for delta rows.
- [ ] Step 5: Commit `feat(pipeline): fetch covers and metadata added since the dump from the cover art archive`.

### Task 7: run the frozen sweep (Step 2) and the crop study (Step 3)

Needs H1, H2 frames labeled through the labeler (at least the smoke-run size: 40 positives and 20 negatives; the full 300 later), H5.

**Files:**
- Create: `pipeline/retrieval/sweep.py`, `pipeline/retrieval/crop_study.py`, `docs/results/<date>-frozen-sweep.md`, `docs/results/<date>-crop-study.md`

**Interfaces (produces):**
```python
# sweep.py   CLI: python -m pipeline.retrieval.sweep --set gold [--models all|names] [--distractors 50000] [--px 224 448]
def run_sweep(models, set_name, distractors, pxs, views) -> Path   # one eval per (model, px) with views=3; a summary table (model, params, dim, px, r1, r5, micro_ap, images/s on the M1) plus per-tag tables; applies the report's gates: shortlist = top 2 teachers of any size and top 2 students <= 30M params; prints whether any student clears r1 >= 0.90 and r5 >= 0.97 (the skip-training exit)
# crop_study.py   CLI: python -m pipeline.retrieval.crop_study --model NAME --set gold
def run_crop_study(model, set_name) -> Path   # treatments: plain center crop, 3 views, OpenCV contour rectify (largest 4-point contour >= 20% of frame, fallback to plain), and when running on macOS with Vision available via pyobjc: VNDetectRectanglesRequest rectify; per-photo miss cause from tags (tilt/partial -> framing, glare -> glare, text-only, unknown-not-in-catalog, near-duplicate confusion); gate: framing+skew share of remaining misses < 25% closes the detector track. Also a visibility curve: every gold positive re-queried with its bottom 10, 20, 30 and 40 percent masked by a neutral band and by a strip cut from another cover (the crate case), reporting r1 at 90, 80, 70 and 60 percent visible, so the point where the frozen model breaks is a number before any training
```
- [ ] Step 1: Tests for the gate logic and the summary table on hand-built eval JSON (no models).
- [ ] Step 2: Implement; run the smoke sweep when 40 labeled positives exist, the full sweep at 300. Each run writes its results file with the command line at the top.
- [ ] Step 3: Stop and hand the shortlist and the crop verdict to Gautham. Phase B's task details get written from these numbers.
- [ ] Step 4: Commit `feat(pipeline): frozen backbone sweep and crop study runners` plus the results files.

### Task 8: export spike (Step 4)

Runs on the shortlist from Task 7, or on all three DINOv3 small models and PE-Core S if Task 7 is not done yet; conversion risk is worth finding early.

**Files:**
- Create: `pipeline/export/__init__.py`, `pipeline/export/to_coreml.py`, `pipeline/export/to_executorch.py`, `pipeline/export/parity.py`, `pipeline/tests/test_parity.py`, `docs/results/<date>-export-spike.md`
- Modify: `pipeline/requirements-tools.txt` (+ `coremltools==9.*`, `executorch` pinned to the current 1.x, `onnx`)

**Interfaces (produces):**
```python
# to_coreml.py   CLI: python -m pipeline.export.to_coreml --model NAME --px 224 [--fp16]   -> EXPORTS_DIR/<model>-<px>.mlpackage ; prints the Mac compute-unit report (coremltools MLModel with compute_units=ALL, 50 timed predictions, median and p95 ms)
# to_executorch.py   CLI: python -m pipeline.export.to_executorch --model NAME --px 224 --backend xnnpack|qnn   -> EXPORTS_DIR/<model>-<px>-<backend>.pte
# parity.py
def parity(model: str, exported: Path, images: list[Path]) -> float   # mean cosine between PyTorch and exported embeddings; the bar is >= 0.99
```
Device timing: iOS via Xcode's Core ML performance report on a connected iPhone (open the `.mlpackage`, Performance tab, run on device; record median, p95, compute-unit placement per layer, first-load time); Android via the ExecuTorch benchmark runner over `adb` on an available phone. Record every number and the phone model in the results file. Gate per the report: a candidate over 30 ms p95 on the oldest target phone, failing conversion, or under 0.99 parity drops out.

- [ ] Step 1: Failing `test_parity_on_identity_model` (a tiny torch module exported with coremltools; parity 1.0).
- [ ] Step 2: Implement the three scripts; export each candidate; run parity on 20 gallery images; fill the results file; note RoPE or unsupported-op failures verbatim.
- [ ] Step 3: Commit `feat(pipeline): core ml and executorch export with parity check` plus results.

### Task 9: docs for Phase A

- [ ] Update `README.md` (layout from Part C, how to run each instrument, where results live) and `CLAUDE.md` (data layout, the no-training-on-eval rule, HF token, rate limits). Commit `docs: phase a instruments and data layout`.

---

## Phases B to D: outline, to be detailed after Task 7 and Task 8 report

Each later task is written at full detail only once the measurements it depends on exist. The gates are the report's; thresholds are the report author's judgment and may be revised with a recorded ruling.

- **Task 10, Step 5: full-gallery embedding and hybrid v1.** Rent a GPU, embed the deduplicated gallery with the two teacher candidates from the shards, rerun eval at full scale, pick the teacher, build the faiss HNSW index, serve from `pipeline/retrieval/serve.py` (FastAPI `POST /identify`, JPEG in for v1), and point `web/` at it with the GPT route as fallback (this absorbs the recovery plan's Tasks 6, 12 and 13). Gate: full-gallery r1 drop over 5 points sends Step 8 to mine hard negatives.
- **Task 11, Step 6: simulator and calibration.** `pipeline/sim/`: a kornia pipeline with two layers of variation. The fitted layer sets the common case: tilt from rectangle corners, exposure and color cast from per-channel fits against the matched clean cover, blur and glare area, all measured on the wild `sim` split, plus the real light fields from the H3 bank applied multiplicatively. The randomized layer covers the tail nobody enumerated: a generative model of illumination (random color temperature 2,000 to 10,000 K, exposure plus or minus 2 stops, random smooth multiplicative gradients and blobs, random saturation and grayscale) sampled wider than anything observed, which is standard domain randomization and costs little in the common case. Occlusion is its own op: hide the bottom 10 to 45 percent with the top strip of another cover or a plastic edge (the crate case), hide a corner with a hand cutout, with `partial`-style side cuts at lower probability; the asymmetric pairing teaches the model to produce the full cover's vector from 60 to 70 percent of it. A background bank from the `sim` split. Two gap tests: a domain classifier (synthetic vs real crops; AUC near 0.5 or the simulator is missing something visible, and its saliency says what) and the recall-gap check (synthetic within 5 points of real, same slice ranking, including the visibility curve from Task 7). Also measures battery and thermal at 5 to 10 fps on the test phones alongside Step 4's numbers.
- **Task 12, mobile shell (D12).** Camera, square guide, on-device embedding of the shortlisted student, POST of the vector or JPEG to the server, result and preview playback. Device timing moves from Xcode reports to the real app.
- **Task 13, Step 7: projection head on cached frozen features** (M1). Gate: gain >= 5 points sends to Task 14; under 2 skips it.
- **Task 14, Step 8: teacher fine-tune, SSCD-style** (rented GPU). Gate: beats Task 13 by >= 3 points, no slice loses more than 3.
- **Task 15, Step 9: student distillation** against precomputed teacher targets. Ship the smallest student within 2 points and under 30 ms p95.
- **Task 16, Step 10: index compression study** (faiss IVF-PQ64, PQ32, OPQ, binary + rescore). Gate: PQ64 within 2 points of exact search ships on-device as an asset pack; otherwise stay hybrid.
- **Task 17, Step 11: lock calibration and app integration.** 99% lock precision, three agreeing frames with scores fused across frames (a crate shot gets several partial views as the phone moves, and the fused score beats any single frame), OCR fallback for low-margin and crate results (the visible top of a sleeve usually carries the artist and title), ship bar per the report.
- **Monthly job:** Task 6 rerun plus catalog rebuild plus incremental embedding of `delta/` with the shipped teacher; annual dump re-slim adding `release_group`.

---

## Self-review notes

- Spec coverage: report Steps 0 to 4 map to Tasks 2, 3 (Step 0), 5 (Step 1), 4 and 7 (Step 2), 7 (Step 3), 8 (Step 4). The catalog update (Gautham's question B) is Task 6 and D11. The labeler (his labeling-app idea) is Task 3 with D8. Simulator fidelity (question C) is Task 11's calibration design plus H3. Candid photos (question A) are D9, H4 and the wild set in Tasks 2 and 3. Phone hardware is covered in Task 8 (compute-unit placement, first-load, memory) and Task 11 (battery, thermal).
- Names reused across tasks are defined once: `Release`, `Catalog`, `normalize`, `album_key`, `Label`, `split_for`, `load_labels`, `append_label`, `Embedder`, `load_embedder`, `query_views`, `embed_images`, `embed_prefix`, `build_index`, `Identifier`, `Match`, `EvalItem`, `is_hit`, `EvalReport`, `run_eval`, `process_prefix`, `cluster`, `parity`.
- Review focus items 1 to 5 are owned by Tasks 2, 2 and 3, 1, 5, 6.
- Proportion: Phase A is specified to signature and test level; Phases B to D are deliberately outlines because their inputs do not exist yet.
