# Record Scan recovery and rebuild plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn three half-finished Record Scan repos into one working pipeline and app: point a phone at a vinyl record, identify it against a 3.1M-cover MusicBrainz gallery using local CLIP-style embeddings and an HNSW index, and play a 30-second Deezer preview.

**Architecture:** One git repo `record-scan` holds the Next.js phone app (`app/`) and the Python pipeline (`pipeline/`). All large data lives outside git in `~/Projects/record-scan-data` and on the external drive. The finished pipeline stages (metadata slimming, cover download) are kept as they are. The live stage (embed, index, identify, serve) is rewritten around one `config.py`, a pluggable embedder, one vector per image, and an evaluation harness that must pass on real phone photos before the multi-hour full embedding run. The app calls a FastAPI retrieval service and falls back to the existing GPT-5-mini route when the service is unavailable.

**Tech stack:** Python 3.12, PyTorch 2.8 on MPS, open_clip, hnswlib, FastAPI, pytest. Next.js 15, React 19, TypeScript, react-webcam, Deezer public API, Vercel. Supabase only for the deferred detector track.

**Spec:** Parts A to D of this document. No separate spec exists. This plan was reverse-engineered from the code and data on 2026-10-24.

## Global constraints

- Never `git add` anything under `~/Projects/record-scan-data`, any `*.json` over 1 MB, `*.npz`, `*.pt`, `*.index`, or `node_modules`. The root `.gitignore` from Task 2 enforces this. Check `git status` before every commit.
- Agents delete nothing from disk except regenerable build artifacts this plan names. Everything else moves into `_archive-*` folders. Deletions are Gautham's call (Part F, H6).
- Commit messages are Conventional Commits (`type(scope): description`, body lines under 100 characters). A global commit-msg hook rejects anything else. Task 1 adds commitlint to the repo.
- Python: one virtualenv at `pipeline/.venv`, Python 3.12, dependencies pinned in `pipeline/requirements.txt`. TypeScript: no `any`.
- Any script that reads cover images calls `require_image_root()` first and exits with a clear message if the drive is not mounted. Never `os.makedirs` a path under `/Volumes`.
- The phone-photo evaluation (Task 9) must clear the bar in D3 before the full embedding run (Task 10) starts.

## Review focus

1. External drive unmounted: every pipeline entry point exits with a clear message before touching the disk. Test in Task 3.
2. Corrupt or truncated JPEG in the gallery: the embedder logs and skips the file and the prefix shard still completes. Test in Task 4.
3. Photo with no record in it (wall, hand, blank sleeve): `identify` returns a top score below the lock threshold and the app keeps scanning. Negatives in the gold set (Task 9), threshold applied in Task 13.
4. Many releases share one cover image (reissues, country variants): results and eval metrics collapse to album level (normalized title plus primary artist), not release UUID. Test in Task 5.
5. Deezer returns no match or rate-limits: the app still shows the identified album and artist. Playback is optional. Checked in Task 13.

---

## Part A: what exists today (inventory, 2026-10-24)

### A1. Why there are "two repos"

There are three git repos plus one data folder, created in two bursts of work.

| When | What | Where | State |
|---|---|---|---|
| 2025-08-15 (one day) | v0 app. Next.js plus react-webcam, sends the frame to GPT-5-mini (Responses API with a zod schema), searches Deezer by album name, embeds the Deezer widget. | `~/Projects/record-scan-app` = github.com/gsajith/record-scan, deployed at record-scan.vercel.app. 13 commits, clean, pushed. | Works end to end. Costs tokens per scan. The only part that runs today. |
| 2025-08-16 to 2025-08-26 | Data pipeline. Downloaded the MusicBrainz JSON dump (18.6 GB tar.xz), slimmed the 3.13M releases that have cover art, fetched cover IDs from Cover Art Archive, downloaded 3,128,310 covers at 500 px to the external drive, split them into 256 two-character prefix folders. | `~/Projects/record-scan/` is the dump download folder, not a git repo. `~/Projects/record-scan/mbdump/` is a git repo with 4 commits in a flat layout and no remote. | Finished. These scripts never run again unless the slim schema changes. |
| 2025-11-10 to 2025-11-12 | Retrieval and detector. Restructured `mbdump` into `src/{fetch-covers,record-retrieval,train-model}` (never committed), wrote open_clip embed, index and query scripts, embedded 2 of 256 prefixes as a test, built an annotation web app for a YOLO cover detector. | Same `mbdump` repo, uncommitted, plus `mbdump/src/train-model/album-annotator` = github.com/gsajith/album-annotator (private, 7 commits, pushed, deployed at album-annotator.vercel.app). | Stopped mid-stage. Index never built. Query never worked after the restructure. |

So `record-scan-app` is not dead. It is v0, and its GPT call is the thing the pipeline is meant to replace. `record-scan/mbdump` is the real ML work and the part in a messy state. The name collision (the `record-scan` folder versus the `record-scan` GitHub repo) is an accident of downloading the dump into a folder with the project's name.

### A2. Data locations

| Data | Location | Size | Notes |
|---|---|---|---|
| Raw MusicBrainz release dump | `~/Projects/record-scan/musicbrains_json_dump.tar.xz` plus COPYING, README, TIMESTAMP, SCHEMA_SEQUENCE, REPLICATION_SEQUENCE, JSON_DUMPS_SCHEMA_NUMBER | 18.6 GB | Dump dated 2025-08-13. The extracted `release` file (about 250 GB) is gone. Only needed to re-slim, for example to add release-group IDs (D5). |
| Slimmed releases | `~/Projects/record-scan/mbdump/src/fetch-covers/slim_releases.json` | 627 MB | 3,129,812 releases that have Cover Art Archive front art. Fields: title, id, artist_ids, asin, disambiguation, format, aliases. |
| Slimmed releases with cover ids | `.../fetch-covers/slim_releases_with_covers.json` | 714 MB | 3,128,762 have a `cover_id`. |
| uuid to release | `.../fetch-covers/uuid_to_metadata.json` | 840 MB | Dict form of the file above. Regenerable in a minute. |
| artist id to name and aliases | `.../fetch-covers/release_slim_artists.json` | 70 MB | 673,401 artists. |
| Downloaded image list | `.../fetch-covers/existing_files.json` | 119 MB | 3,128,310 UUIDs. The only index of what is on the drive. |
| Failure lists | `failed_cover_fetches.json` (688), `failed_cover_id_fetches.json` (7,746) | small | |
| Cover images | `/Volumes/Seagate Portabl/record scan Pictures/record_scan_cover_images_split/<uuid[:2]>/<uuid>.jpg` | about 250 GB, 3.13M files | External drive. Not mounted on 2026-10-24. Older scripts reference the pre-split path `.../Pictures/record_scan_cover_images`. |
| Embeddings (test run) | `.../record-retrieval/embeddings/{00,01}.npz` and `manifest.json` | 300 MB | 24,270 images times 13.6 augmentations = 329,960 vectors, 512-d float16, open_clip ViT-B/32. |
| Detector base weights | `.../record-retrieval/yolov8n.pt` | 6 MB | COCO pretrained, never trained on covers. |
| Annotations | Supabase project named in `album-annotator/.env.local` | unknown | Gautham confirmed on 2026-10-24 that the project is paused or expired (its host no longer resolves). The annotation rows and uploaded photos are lost unless the Supabase dashboard still offers a restore. `album_dataset/` is empty, so nothing was ever exported locally. |
| Stale model caches | `~/.cache/clip` 1.2 GB, `~/.cache/torch/hub/checkpoints` 1.2 GB, HF `models--openai--clip-vit-base-patch32` 1.1 GB, HF `models--timm--vit_base_patch32_clip_224.openai` 577 MB | about 4 GB | From abandoned August experiments with the OpenAI `clip` package, DINOv2 and HF transformers CLIP. |
| Old virtualenv | `~/Projects/record-scan/mbdump/venv` | 1.4 GB | Absolute paths baked in. Recreate rather than move. |

Format breakdown of the 3.13M releases: Digital Media 1.64M, CD 1.01M, 12" Vinyl 198k, 7" Vinyl 83k, Cassette 33k. Releases with any vinyl format: 302,620.

### A3. Pipeline status

| Stage | Script(s) | Status | Evidence |
|---|---|---|---|
| 1. Slim the dump | `slim_file.py` | Done | slim_releases.json, 3,129,812 rows |
| 2. Cover IDs | `parallel_fetch_cover_art_urls.py` (`fetch_cover_art_urls.py` is the slow serial draft) | Done | 3,128,762 cover ids, 7,746 failures |
| 3. Download covers | `fetch_cover_art.py`, `move_images_to_subdirectories.py` | Done | existing_files.json has 3,128,310, 688 failures |
| 4. uuid map | `generate_uuid_to_metadata.py` | Done | uuid_to_metadata.json |
| 5. Embed | `embed_batch.py` | 0.8 percent | Prefixes 00 and 01 only. `prefixes = ['00', '01']` is hardcoded in `main()`. |
| 6. Index | `build_index.py` | Never run | No `.index` file anywhere on disk |
| 7. Query | `query.py` | Broken | Paths point at pre-restructure locations. Uses the `clip` package while the gallery used `open_clip`. |
| 8. Detector data | album-annotator plus `export_yolo_dataset.py` | Lost | Supabase project paused or expired, dataset folder empty, no trained weights. Restarting this track means annotating from zero. |
| 9. Retrieval service | none | Not started | |
| 10. App using the pipeline | none (v0 uses GPT) | Not started | |

### A4. Deployments and services

- record-scan.vercel.app builds from github.com/gsajith/record-scan (the v0 app). Needs `OPENAI_API_KEY`.
- album-annotator.vercel.app builds from github.com/gsajith/album-annotator. Needs the two Supabase vars. Its backend is paused or deleted.
- No `.vercel` link folders exist locally. Both projects were linked through the Vercel GitHub integration.

## Part B: problems found

### B1. Data-safety hazards (fix before any commit)

- `mbdump/.gitignore` is `/*.json`, `release`, `/venv`. After the restructure the JSON moved into `src/fetch-covers/`, so `git add -A` would stage about 2.3 GB of JSON, 300 MB of `.npz`, `yolov8n.pt`, and record `album-annotator` as an embedded repo.
- `fetch_cover_art.py` and `count_files.py` call `os.makedirs(OUTPUT_DIR)` on a `/Volumes/...` path. With the drive unmounted this silently creates the folder on the boot disk and would download 250 GB there.
- `.DS_Store` is tracked at mbdump HEAD.

### B2. Bugs in the live stage

- `embed_batch.py` rewrites `manifest.json` after every image and recompresses the whole `.npz` shard after every 32 vectors. Both are quadratic per prefix and are the main reason a prefix takes so long. The model loads at import time, so `build_index.py` and `verify_embeddings.py` load CLIP just to import a path constant. The test override `prefixes = ['00', '01']` is still in `main()`.
- `query.py` sets `INDEX_PATH = "cover_hnsw.index"` but the index would be written under `embeddings/`. `UUID_TO_META_PATH = "../uuid_to_metadata.json"` but the file is in `../fetch-covers/`. It embeds with the OpenAI `clip` package while the gallery used `open_clip` (same weights, different preprocessing code; keep one). It leaves a `print(len(approx))` debug line and calls `cv2.imshow`.
- `resume_failed_prefixes.py` looks for shards in the current directory, not under `embeddings/`.
- `export_yolo_dataset.py` calls `load_dotenv("album-annotator/.env.local")` relative to the current directory. `album_cover.yaml` holds an absolute path.
- v0 app, `src/app/api/deezer/album/route.ts`: the URL template literal starts with a tab character and the album name is not URL-encoded. `page.tsx` has unused `Image` and `useCallback` imports, effect dependency warnings, and matches artists by exact lowercase string.

### B3. Design flaw: index-time augmentation does not scale

NOTES.md budgets about 3 GB of embeddings and a 10 to 15 GB index, assuming one vector per image. The shards actually hold 13.6 vectors per image. At full scale that is 42.5M vectors: 43 GB on disk and roughly 100 GB of RAM for an hnswlib graph with M=48. It will not fit in 64 GB. One vector per image is 3.13M vectors, about 3.2 GB on disk and 7.5 GB in RAM, and the embedding run is 13 times shorter. Tolerance to rotation, glare and skew should come from the query side (a framing guide plus test-time augmentation of the query photo) and be measured, not assumed. D2 makes one vector per image the default.

### B4. Dead code

Not carried into the new layout. Originals remain in the archive folder.

- `mbdump/src/fetch-covers/fetch_cover_art_urls.py` (serial draft of the parallel script)
- `mbdump/dev/count_files.py`, `dev/count_existing_files.py`, `dev/test_cuda.py` (one-off checks, and the Mac has no CUDA)
- Tracked at mbdump HEAD but already removed from the working tree: `generate_2_embeddings.py`, `generate_100_embeddings.py`, `generate_all_embddings.py` (OpenAI `clip`, HF CLIP and DINOv2 experiments superseded by `embed_batch.py`), `.DS_Store`
- `mbdump/src/record-retrieval/detect_album_proxy.py` (COCO "book" and "tv" proxy detector experiment)
- `mbdump/src/record-retrieval/resume_failed_prefixes.py` and `verify_embeddings.py` (folded into the new embed and index scripts)
- `album-annotator/lib/datasetPaths.ts` (unused since the "Remove upload page" commit)
- `record-scan-app/public/{file,globe,next,vercel,window}.svg` and the create-next-app README. The same five SVGs sit in album-annotator.

Kept but moved: `dev/line1.json` and `dev/line1_ideal.json` (schema examples), `visualize_embeddings_umap.py` (dev tool, dependencies become optional extras), `NOTES.md` and the two READMEs (historical docs).

Elsewhere on disk: nothing else related to this project was found under `~/Projects`, `~/Documents` or `~/Downloads` besides the caches in A2. `~/Downloads/archive` is an unrelated face dataset.

## Part C: target layout

```
~/Projects/record-scan/            git: github.com/gsajith/record-scan   (today's record-scan-app, renamed)
  package.json                     tooling only (husky and commitlint)
  commitlint.config.mjs  .husky/  .github/workflows/pr-title.yml
  CLAUDE.md  README.md  .gitignore  .env.example
  app/                             Next.js phone app (today's repo root contents)
  pipeline/
    requirements.txt  requirements-tools.txt  .venv/
    config.py                      RS_DATA_DIR, RS_IMAGE_ROOT, derived dirs, fail-fast checks
    metadata/                      slim_file.py, parallel_fetch_cover_art_urls.py, fetch_cover_art.py,
                                   move_images_to_subdirectories.py, generate_uuid_to_metadata.py,
                                   catalog.py (new), examples/line1.json, examples/line1_ideal.json
    retrieval/                     embedders.py, embed.py, build_index.py, identify.py, eval.py, serve.py
    detector/                      export_yolo_dataset.py                   (deferred track)
    tools/                         visualize_embeddings_umap.py
    tests/
  docs/
    superpowers/plans/2026-10-24-record-scan-recovery.md   (this file)
    history/                       mbdump-readme-2025-11.md, pipeline-notes-2025-11.md, detector-notes-2025-11.md

~/Projects/record-scan-data/       NOT git  (today's ~/Projects/record-scan, renamed)
  musicbrainz-dump-2025-08-13/     the tar.xz and its six sidecar files
  metadata/                        the six JSON files from fetch-covers
  embeddings/clip-vit-b-32-aug13/  00.npz, 01.npz, manifest.json  (the 2025-11 augmented test shards)
  embeddings/<model>/              new shards and index (Tasks 10 and 11)
  models/                          yolov8n.pt
  gold/                            photos/ and labels.csv  (Part F, H2)
  _archive-mbdump-2025-11/         the old mbdump git repo, untouched (contains venv; delete when ready)

~/Projects/album-annotator/        git: github.com/gsajith/album-annotator  (moved out of the pipeline tree, deferred)
```

## Part D: decisions

Recommendation first. The plan assumes the recommendation unless Gautham says otherwise (H4).

- **D1 One repo.** Merge app and pipeline into `record-scan`. The annotator stays its own repo and moves out of the pipeline tree. Alternative: keep three repos and only fix paths. Recommendation: merge, because the split is what caused the confusion.
- **D2 One vector per image, no index-time augmentation.** See B3. Query-side test-time augmentation is a flag on `identify()`. Alternative: keep 3 or 4 augmentations, still about 25 GB of RAM. Recommendation: one.
- **D3 The eval picks the model, not a default.** Candidates: open_clip ViT-B/32 (current), open_clip ViT-B/16, DINOv2 ViT-S/14. Instance retrieval of near-identical artwork usually favors DINOv2 or B/16 over B/32. Bar to pass before the full run: at least 90 percent top-5 and 75 percent top-1 album-level accuracy on the gold photos against 50k distractors. If no model passes, stop and report rather than embed 3.1M images.
- **D4 Framing guide instead of a detector, for now.** The app shows a square guide, the user fills it with the sleeve, and `identify` uses the center square. The YOLO detector track is deferred until the eval shows crop error is the dominant failure. Its training data is gone with the expired Supabase project, so restarting it costs a fresh annotation session of 100 to 300 photos. Only pay that if the eval says so.
- **D5 Scope: all 3.13M releases, with album-level dedup at result time.** Vinyl-only (303k) would miss albums whose only cover is on the CD release. Release-group dedup needs a re-slim from the tar.xz, which is why the tar.xz is kept. Deferred (T15).
- **D6 Serving: a Python FastAPI service on the Mac** holding the model and the index (about 8 GB RAM), reached from the Vercel app through a `RETRIEVAL_URL` env var, with phone testing over Tailscale or cloudflared. The GPT route stays as a fallback. Later options: a 16 GB VPS, or on-device with a product-quantized index (about 64 bytes per vector, roughly 200 MB) and a Core ML CLIP (T16).
- **D7 Agents delete nothing.** Part F, H6 lists what Gautham can delete and the space it frees.

## Part E: orchestration

Phase 1 needs no drive and no human input. Phase 2 needs the drive and the gold photos. Phase 3 is long-running. Phase 4 is deferred.

```
Phase 1  T1 -> T2 -+-> T3 -> T4 (embed)      -+
                   |    \--> T5 (index, id)   +-> T8 (README, CLAUDE.md)
                   +-> T6 (app cleanup)       |
                   +-> T7 (eval harness)     -+
Phase 2  [H1 drive, H2 gold photos] -> T9 eval runs -> decision gate (D3)
Phase 3  T10 full embed (hours) -> T11 index -> T12 serve -> T13 app -> [H5b deploy]
Phase 4  deferred: T14 detector (needs H3), T15 release-group dedup, T16 on-device
```

Dispatch: one fresh agent per task and a fresh reviewer per task (superpowers:subagent-driven-development). After T3 lands, T4, T5, T6 and T7 touch disjoint directories and can run as parallel workers. Merge in the order T5, T4, T7, T6. T10 runs under `nohup caffeinate -i` with the agent polling the log rather than holding the process. If dispatching through Orca, load `orca skills get orchestration` and use the diagram above as the task DAG, with T9 as a decision gate.

## Part F: human tasks (only Gautham can do these)

- **H1** Mount the Seagate drive. Confirm `/Volumes/Seagate Portabl/record scan Pictures/record_scan_cover_images_split/00` exists. Needed from T9 on.
- **H2** Gold set. Photograph 25 to 40 of your own records with your phone the way a user would: standing, room light, slight angle, some glare. Add 5 photos with no record in them. Save as `~/Projects/record-scan-data/gold/photos/<slug>.jpg` and fill `~/Projects/record-scan-data/gold/labels.csv` with the columns `file,artist,title`, leaving artist and title blank for negatives. These photos are hold-outs. Do not use them for anything else.
- **H3** (deferred track, only if T14 is ever approved) Try a restore in the Supabase dashboard. If it is gone, create a new project, put its URL and anon key in `~/Projects/album-annotator/.env.local` and the Vercel project, recreate the `annotations` table and the public `album-images` bucket as described in `docs/history/detector-notes-2025-11.md`, then annotate 100 to 300 photos.
- **H4** Confirm or override D1 to D6.
- **H5a** Right after Task 2 lands: in Vercel, set the record-scan project's root directory to `app/`. Until then the next deploy of record-scan.vercel.app fails.
- **H5b** When T12 is running: add `RETRIEVAL_URL` to the Vercel project and expose the service over Tailscale or cloudflared.
- **H6** Deletions you may want, none done by the plan: `_archive-mbdump-2025-11/venv` (1.4 GB), `record-scan/app/.next` (regenerable), `~/.cache/clip` (1.2 GB), `~/.cache/torch/hub/checkpoints` (1.2 GB, only if T9 does not pick DINOv2), HF `models--openai--clip-vit-base-patch32` (1.1 GB) and `models--timm--vit_base_patch32_clip_224.openai` (577 MB), and eventually the 18.6 GB tar.xz once D5 is settled, or move it to the Seagate drive.

---

## Phase 1: consolidate and make the pipeline runnable (no drive needed)

### Task 1: commitlint in the app repo

Per the global rule, a repo of mine with no `commitlint.config.*` gets it before other work, as its own `chore:` commit and PR.

**Files:**
- Create: `commitlint.config.mjs`, `.husky/commit-msg`, `.github/workflows/pr-title.yml` (copied from `~/Projects/gsajith_blind_box`)
- Modify: `package.json` (add `"prepare": "husky"` and devDependencies `@commitlint/cli@20.5.3`, `@commitlint/config-conventional@20.5.3`, `husky@9.1.7`)

- [ ] Step 1: `cd ~/Projects/record-scan-app && git status --porcelain` prints nothing and `git rev-parse --short HEAD` prints `3127423`.
- [ ] Step 2: Copy the three files. Run `npm i -D @commitlint/cli@20.5.3 @commitlint/config-conventional@20.5.3 husky@9.1.7`, add the prepare script, run `npm run prepare`.
- [ ] Step 3: Verify: `echo "bad message" | npx commitlint` exits non-zero and `echo "chore: x" | npx commitlint` exits 0.
- [ ] Step 4: Branch `chore/commitlint`, commit `chore: add commitlint and husky commit-msg hook`, push, `gh pr create`, merge, pull main.

### Task 2: restructure into the monorepo and move data out of git

Everything is a move or a copy. Nothing is deleted. Do this with the drive unmounted; nothing here touches it.

**Files:** see Part C. Key edits: root `.gitignore`, root `package.json` split, `.env.example`.

- [ ] Step 1: Preconditions. `git -C ~/Projects/record-scan-app status --porcelain` is empty except the untracked `docs/` folder holding this plan. `git -C ~/Projects/record-scan/mbdump/src/train-model/album-annotator status --porcelain` is empty. `pgrep -fl "record-scan"` shows no dev server or python process from these folders.
- [ ] Step 2: Renames.
  ```bash
  mv ~/Projects/record-scan ~/Projects/record-scan-data
  mv ~/Projects/record-scan-app ~/Projects/record-scan
  mv ~/Projects/record-scan-data/mbdump/src/train-model/album-annotator ~/Projects/album-annotator
  ```
- [ ] Step 3: In `~/Projects/record-scan`: `mkdir app`. `git mv` these into `app/`: `src`, `public`, `tsconfig.json`, `next.config.ts`, `eslint.config.mjs`, `package-lock.json`, `README.md`, `.gitignore`. Plain `mv` for the untracked `.env`, `node_modules`, `.next`, `next-env.d.ts`. Split `package.json`: the root keeps `name`, `private`, `scripts.prepare` and the three commitlint and husky devDependencies; `app/package.json` gets everything else. Run `npm install` at the root and in `app/`. Create `.env.example` at the root listing `OPENAI_API_KEY`, `RETRIEVAL_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` with placeholder values, and delete `app/.env.example`.
- [ ] Step 4: Root `.gitignore`, exact content:
  ```
  node_modules/
  .next/
  .env*
  !.env.example
  .venv/
  __pycache__/
  *.pyc
  .DS_Store
  *.npz
  *.pt
  *.index
  *.tsbuildinfo
  next-env.d.ts
  pipeline/**/*.json
  !pipeline/metadata/examples/*.json
  ```
- [ ] Step 5: Pipeline scripts, copied with `cp` from `~/Projects/record-scan-data/mbdump`:
  - `src/fetch-covers/{slim_file,parallel_fetch_cover_art_urls,fetch_cover_art,move_images_to_subdirectories,generate_uuid_to_metadata}.py` to `pipeline/metadata/`
  - `dev/line1.json`, `dev/line1_ideal.json` to `pipeline/metadata/examples/`
  - `src/record-retrieval/{embed_batch,build_index,query}.py` to `pipeline/retrieval/`
  - `src/record-retrieval/visualize_embeddings_umap.py` to `pipeline/tools/`
  - `src/train-model/export_yolo_dataset.py` to `pipeline/detector/` (`album_cover.yaml` is not copied; Task 3 generates it)
  - `README.md` to `docs/history/mbdump-readme-2025-11.md`, `src/record-retrieval/NOTES.md` to `docs/history/pipeline-notes-2025-11.md`, `src/train-model/README.md` to `docs/history/detector-notes-2025-11.md`

  Not copied (B4): `fetch_cover_art_urls.py`, `dev/*.py`, `detect_album_proxy.py`, `resume_failed_prefixes.py`, `verify_embeddings.py`.
- [ ] Step 6: Data moves inside `~/Projects/record-scan-data`. Same volume, so each is instant.
  ```bash
  cd ~/Projects/record-scan-data
  mkdir -p musicbrainz-dump-2025-08-13 metadata embeddings/clip-vit-b-32-aug13 models gold/photos
  mv musicbrains_json_dump.tar.xz COPYING README TIMESTAMP SCHEMA_SEQUENCE REPLICATION_SEQUENCE JSON_DUMPS_SCHEMA_NUMBER musicbrainz-dump-2025-08-13/
  mv mbdump/src/fetch-covers/*.json metadata/
  mv mbdump/src/record-retrieval/embeddings/* embeddings/clip-vit-b-32-aug13/
  mv mbdump/src/record-retrieval/yolov8n.pt models/
  mv mbdump _archive-mbdump-2025-11
  ```
- [ ] Step 7: Verify. `git add -A` then `git diff --cached --name-only | xargs -I{} du -k {} | sort -n | tail -5` shows nothing over 1 MB except `app/package-lock.json`. `du -sh ~/Projects/record-scan-data/{metadata,embeddings,models,musicbrainz-dump-2025-08-13}` matches A2 (about 2.3 GB, 300 MB, 6 MB, 18.6 GB). `cd app && npm run lint && npx tsc --noEmit` pass.
- [ ] Step 8: Commit `chore: restructure into app/ and pipeline/ monorepo` with a body listing the renames and the data folder. Push. Tell Gautham to do H5a now.

### Task 3: pipeline config, environment and fail-fast checks

**Files:**
- Create: `pipeline/config.py`, `pipeline/requirements.txt`, `pipeline/requirements-tools.txt`, `pipeline/__init__.py` plus `__init__.py` in each subpackage, `pipeline/tests/conftest.py`, `pipeline/tests/test_config.py`
- Modify: `pipeline/metadata/*.py`, `pipeline/detector/export_yolo_dataset.py`, `pipeline/tools/visualize_embeddings_umap.py` (paths only)

**Interfaces (produces):**
```python
# pipeline/config.py
DATA_DIR: Path        # env RS_DATA_DIR, default ~/Projects/record-scan-data
IMAGE_ROOT: Path      # env RS_IMAGE_ROOT, default "/Volumes/Seagate Portabl/record scan Pictures/record_scan_cover_images_split"
META_DIR, EMB_DIR, MODELS_DIR, GOLD_DIR, DETECTOR_DIR: Path   # DATA_DIR / metadata | embeddings | models | gold | detector
def require_image_root() -> Path   # raises SystemExit("Cover images not found at <path>. Mount the drive or set RS_IMAGE_ROOT.") unless IMAGE_ROOT / "00" is a directory
def image_path(uuid: str) -> Path  # IMAGE_ROOT / uuid[:2] / f"{uuid}.jpg"
def device() -> str                # "mps" if torch.backends.mps.is_available() else "cpu"
```

`requirements.txt`, pinned to what the old venv had: `torch==2.8.0 torchvision==0.23.0 open_clip_torch==3.2.0 timm==1.0.19 hnswlib==0.8.0 numpy==2.2.6 pillow==11.3.0 opencv-python-headless==4.12.0.88 tqdm==4.67.1 aiohttp==3.12.15 aiofiles==24.1.0 requests==2.32.4 fastapi uvicorn python-multipart pytest`. `requirements-tools.txt`: `umap-learn plotly dash supabase python-dotenv ultralytics`.

- [ ] Step 1: Failing tests in `test_config.py`: `test_env_overrides_data_dir` (monkeypatch `RS_DATA_DIR`, reload `config`, assert `META_DIR == tmp / "metadata"`), `test_require_image_root_exits_when_missing` (monkeypatch `RS_IMAGE_ROOT` to an empty tmp dir, `pytest.raises(SystemExit)`), `test_image_path_uses_two_char_prefix`.
- [ ] Step 2: `python3.12 -m venv pipeline/.venv && pipeline/.venv/bin/pip install -r pipeline/requirements.txt`. Run the tests. Expect FAIL with ImportError.
- [ ] Step 3: Implement `config.py`. Replace every hardcoded path in `pipeline/metadata/*.py` with `META_DIR` and `IMAGE_ROOT`. In `export_yolo_dataset.py` use `load_dotenv(Path(__file__).parents[2] / ".env")`, write the dataset under `DETECTOR_DIR / "album_dataset"`, and write `album_cover.yaml` next to it with the resolved absolute path. In `visualize_embeddings_umap.py` import `EMB_DIR` and `image_path` from `config` instead of `embed_batch`. Remove the `os.makedirs(OUTPUT_DIR)` in `fetch_cover_art.py` and `move_images_to_subdirectories.py` and call `require_image_root()` instead.
- [ ] Step 4: `cd pipeline && .venv/bin/pytest tests/test_config.py -v` passes. `grep -rn "/Volumes\|/Users/" pipeline --include=*.py` matches only `config.py`.
- [ ] Step 5: Commit `refactor(pipeline): central config with fail-fast image root`.

### Task 4: embedders and the embed script (one vector per image, resumable by prefix)

**Files:**
- Create: `pipeline/retrieval/embedders.py`, `pipeline/retrieval/embed.py` (replaces `embed_batch.py`), `pipeline/tests/test_embedders.py`, `pipeline/tests/test_embed.py`, `pipeline/tests/fixtures/` (three tiny generated JPEGs and one truncated one, generated by `conftest.py`)
- Delete: `pipeline/retrieval/embed_batch.py`

**Interfaces (produces):**
```python
# embedders.py
EMBEDDER_NAMES = ("clip-vit-b-32", "clip-vit-b-16", "dinov2-vits14")
class Embedder(Protocol):
    name: str
    dim: int
    def preprocess(self, img: Image.Image) -> torch.Tensor
    def embed(self, images: list[Image.Image]) -> np.ndarray   # (n, dim) float32, rows L2-normalized
def load_embedder(name: str, device: str | None = None) -> Embedder
def tta_views(img: Image.Image) -> list[Image.Image]  # [original, rotate -8, rotate +8, 0.9 center crop, brightness 1.15]; used by identify, never by embed

# embed.py   CLI: python -m pipeline.retrieval.embed --model clip-vit-b-32 (--prefixes 00 01 | --all) [--batch 64] [--workers 6]
def embed_prefix(prefix: str, embedder: Embedder, out_dir: Path, batch: int, workers: int) -> int
    # writes out_dir/<prefix>.npz once at the end (uuids: array of str, embs: float16) and touches out_dir/<prefix>.done; returns the vector count
def pending_prefixes(out_dir: Path) -> list[str]
    # two-character hex prefixes that are directories under IMAGE_ROOT minus those with a .done file
```

The shard folder is `EMB_DIR / embedder.name`. Images load through a `torch.utils.data.DataLoader` with `workers` processes using the embedder's `preprocess`. An unreadable file yields `None`, is appended to `out_dir/<prefix>.errors`, and does not abort the prefix. The model loads in `main()`, never at import. Dims: clip-vit-b-32 512, clip-vit-b-16 512, dinov2-vits14 384 via `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` with ImageNet normalization, resize 224 and center crop. Progress is a tqdm bar per prefix showing images per second, and after the first prefix finishes the script prints an ETA for all pending prefixes.

- [ ] Step 1: Failing tests. `test_each_embedder_returns_unit_vectors` (parametrized over `EMBEDDER_NAMES`, two fixture images, assert shape `(2, dim)` and norms close to 1; marked `@pytest.mark.slow` because it downloads weights). `test_embed_prefix_writes_one_shard_and_skips_corrupt` (tmp `RS_IMAGE_ROOT` with prefix `ab` holding 3 good and 1 truncated JPEG, a stub embedder returning ones; assert the npz has 3 uuids, `.errors` has 1 line, `.done` exists). `test_pending_prefixes_skips_done`.
- [ ] Step 2: Run. FAIL with module not found.
- [ ] Step 3: Implement `embedders.py` and `embed.py`.
- [ ] Step 4: `pytest tests/test_embed.py -v` passes. `pytest -m slow tests/test_embedders.py -v` passes.
- [ ] Step 5: `git rm pipeline/retrieval/embed_batch.py`. Commit `feat(pipeline): pluggable embedders and prefix-resumable embed script`.

### Task 5: index build and identify

**Files:**
- Create: `pipeline/metadata/catalog.py`, `pipeline/retrieval/identify.py` (replaces `query.py`), `pipeline/tests/test_catalog.py`, `pipeline/tests/test_index_identify.py`
- Modify: `pipeline/retrieval/build_index.py` (rewrite)
- Delete: `pipeline/retrieval/query.py`

**Interfaces (produces):**
```python
# catalog.py   metadata lookups, loaded once
@dataclass
class Release: uuid: str; title: str; artists: list[str]; format: str | None
def normalize(s: str) -> str   # casefold, strip accents, drop punctuation, collapse spaces, drop a leading "the "
class Catalog:
    def __init__(self, meta_dir: Path = META_DIR): ...   # loads uuid_to_metadata.json and release_slim_artists.json
    def release(self, uuid: str) -> Release
    def album_key(self, uuid: str) -> str                 # normalize(title) + "|" + normalize(first artist)
    def find(self, artist: str, title: str) -> list[str]  # uuids whose album_key equals the normalized inputs; used by eval

# build_index.py   CLI: python -m pipeline.retrieval.build_index --model clip-vit-b-32 [--M 32] [--ef-construction 200]
def build_index(shards: list[Path], dim: int, out_dir: Path, M: int, ef_construction: int) -> int
    # verifies every shard's dim and equal uuid/emb lengths first; writes out_dir/index.hnsw and out_dir/labels.npy (dtype "U36"); returns the vector count

# identify.py   CLI: python -m pipeline.retrieval.identify photo.jpg --model clip-vit-b-32 [--k 5] [--tta]
@dataclass
class Match: uuid: str; title: str; artists: list[str]; score: float   # score = 1 - cosine distance
class Identifier:
    def __init__(self, model: str, index_dir: Path, catalog: Catalog, ef: int = 128): ...
    def identify(self, img: Image.Image, k: int = 5, tta: bool = False, center_square: bool = True) -> list[Match]
        # center square crop, then TTA views averaged if tta, embed, knn with k * 10, collapse by album_key keeping the best score, return top k
```

- [ ] Step 1: Failing tests. `test_normalize` (`"The Beatles"` becomes `"beatles"`, accents stripped, `"Abbey Road (Remastered)"` keeps its words). `test_find_by_artist_title` on a 3-entry fake metadata folder. `test_build_index_and_identify_roundtrip`: 200 random unit vectors as a fake shard with uuids, three of them duplicated under other uuids that share one fake album_key, a stub embedder that returns vector 7 for any image; assert top-1 is uuid 7 and the duplicates collapse to one `Match`. `test_build_index_rejects_dim_mismatch`.
- [ ] Step 2: Run. FAIL.
- [ ] Step 3: Implement. hnswlib `space="cosine"`, `M=32`, `ef_construction=200`, `ef=128` at query time. Labels go in `labels.npy` to avoid a 3M-entry JSON.
- [ ] Step 4: `pytest tests/test_catalog.py tests/test_index_identify.py -v` passes. Smoke test on the real 2025-11 shards, no drive needed: `python -m pipeline.retrieval.build_index --model clip-vit-b-32-aug13` builds from `EMB_DIR/clip-vit-b-32-aug13/*.npz` (330k vectors, under a minute). Then `identify` on a cover fetched from `https://coverartarchive.org/release/<uuid>/front-500` for any uuid in shard `00` returns that album at top-1.
- [ ] Step 5: `git rm pipeline/retrieval/query.py`. Commit `feat(pipeline): hnsw index build and album-level identify`.

### Task 6: app cleanup (v0 keeps working)

**Files:**
- Create: `app/src/lib/deezer.ts`
- Modify: `app/src/app/page.tsx`, `app/src/app/api/deezer/album/route.ts`, `app/src/app/layout.tsx` (description), `app/README.md` (replace the boilerplate with how to run and the env vars)
- Delete: `app/public/{file,globe,next,vercel,window}.svg`

**Interfaces (produces):**
```ts
// app/src/lib/deezer.ts (server only)
export interface DeezerAlbum { id: number; title: string; artist: string; cover: string; preview: string | null } // preview is the first track's 30 s mp3 from /album/{id}/tracks
export function normalizeName(s: string): string   // lowercase, strip accents and punctuation, drop a leading "the "
export async function findAlbum(title: string, artists: string[]): Promise<DeezerAlbum | null>
  // encodeURIComponent on the query; a result matches when normalizeName(result.artist.name) equals any normalized artist; with no artists given, the first result wins
```

- [ ] Step 1: Rewrite the Deezer route to call `findAlbum` and return `DeezerAlbum | null`. Keep the path `/api/deezer/album` and add an `artists` query param (comma separated) so the deployed app keeps working. Remove the tab and encode the query.
- [ ] Step 2: `page.tsx`: drop the unused imports, type the result as `DeezerAlbum`, run the Deezer search from the capture handler instead of an effect, keep the capture button and the widget iframe for now.
- [ ] Step 3: `cd app && npm run lint && npx tsc --noEmit` clean. `npm run dev`, scan one record, confirm the Deezer widget still appears.
- [ ] Step 4: Commit `refactor(app): typed deezer lookup and lint cleanup`.

### Task 7: evaluation harness (code only; runs in Task 9)

**Files:**
- Create: `pipeline/retrieval/eval.py`, `pipeline/tests/test_eval.py`

**Interfaces (produces):**
```python
# eval.py   CLI: python -m pipeline.retrieval.eval --model clip-vit-b-32 [--distractors 50000] [--tta] [--seed 0] [--dry-run]
@dataclass
class GoldItem: file: Path; artist: str; title: str; uuids: list[str]   # uuids is empty for negatives
def load_gold(gold_dir: Path, catalog: Catalog) -> list[GoldItem]       # reads labels.csv; warns and drops positives whose (artist, title) resolve to no uuid; --dry-run prints this table and exits
def build_mini_gallery(gold: list[GoldItem], distractors: int, seed: int) -> list[str]   # every gold uuid plus a seeded random sample from existing_files.json
def run_eval(model: str, gold: list[GoldItem], gallery: list[str], tta: bool, work_dir: Path) -> EvalReport
@dataclass
class EvalReport:
    top1: float; top5: float; neg_max_score: float; pos_min_top1_score: float; rows: list[dict]
    # top1 and top5 are album-level: a hit is any returned Match whose album_key equals the album_key of a gold uuid
    # rows per photo: file, rank of first hit or None, best score, top-1 title
    def to_markdown(self) -> str
```

`run_eval` embeds the mini gallery with the Task 4 batching into `work_dir/<model>/`, builds a throwaway index with the Task 5 builder, runs `identify` on every photo, and writes `work_dir/report-<model>[-tta].md`.

- [ ] Step 1: Failing tests. `test_load_gold_resolves_and_warns` (fake catalog, 2 positives, 1 negative, 1 unresolvable). `test_report_metrics` (hand-built rows give known top1 and top5 values).
- [ ] Step 2: Run. FAIL.
- [ ] Step 3: Implement.
- [ ] Step 4: `pytest tests/test_eval.py -v` passes.
- [ ] Step 5: Commit `feat(pipeline): phone-photo evaluation harness`.

### Task 8: docs for the next person, agent or human

**Files:**
- Create: `CLAUDE.md`. Rewrite: `README.md`.

- [ ] Step 1: `README.md`: what the project is in one paragraph, the layout from Part C, the data locations from A2, how to run each stage with the exact commands from Tasks 3 to 7 and 10 to 13, and the status table from A3 updated to what has landed.
- [ ] Step 2: `CLAUDE.md`, short: the `RS_DATA_DIR` and `RS_IMAGE_ROOT` env vars and defaults, never commit data, the drive must be mounted for embed and eval, run `pipeline/.venv/bin/pytest` and the app lint and tsc before commits, link to this plan.
- [ ] Step 3: Commit `docs: project map, data locations and run instructions`.

## Phase 2: prove retrieval works on real phone photos (needs H1 and H2)

### Task 9: run the eval and pick the model

- [ ] Step 1: `python -c "from pipeline.config import require_image_root; require_image_root()"` passes. `ls ~/Projects/record-scan-data/gold/photos | wc -l` is at least 30. `python -m pipeline.retrieval.eval --dry-run` resolves at least 25 positives. Fix typos in `labels.csv`, not in code.
- [ ] Step 2: Run each of the three models with and without `--tta` (6 runs, 50k distractors). Each run embeds about 50k images, which takes minutes on MPS. Record images per second per model from the logs.
- [ ] Step 3: Write `docs/eval-<date>.md` with the six reports, the throughput numbers, a proposed lock threshold (between `neg_max_score` and `pos_min_top1_score`), and a recommendation against the D3 bar.
- [ ] Step 4: Stop and hand the decision to Gautham. Do not start Task 10 until the model and settings are confirmed.

## Phase 3: full scale, service, app

### Task 10: embed all 3.13M covers

- [ ] Step 1: Time one prefix: `python -m pipeline.retrieval.embed --model <chosen> --prefixes 02`. Report images per second and the ETA for the remaining prefixes.
- [ ] Step 2: `nohup caffeinate -i pipeline/.venv/bin/python -m pipeline.retrieval.embed --model <chosen> --all > ~/Projects/record-scan-data/embeddings/<chosen>/embed.log 2>&1 &`. Poll the log. On a crash, rerun the same command; resume is per prefix.
- [ ] Step 3: Done when 256 `.done` files exist and the total vector count across shards equals 3,128,310 minus the total lines in `*.errors`, within 0.1 percent. Commit nothing (it is data) but add the run's numbers to the README status table.

### Task 11: build the full index

- [ ] Step 1: `/usr/bin/time -l pipeline/.venv/bin/python -m pipeline.retrieval.build_index --model <chosen>`. Expect roughly 8 GB peak RSS for 512-d, less for 384-d.
- [ ] Step 2: Re-run the Task 9 gold photos through `identify` against the full index. Accuracy must not drop more than 5 points from the mini-gallery result.
- [ ] Step 3: Record index size, build time and median query latency in the README.

### Task 12: retrieval service

**Files:**
- Create: `pipeline/retrieval/serve.py`, `pipeline/tests/test_serve.py`

**Interfaces (produces):**
```
POST /identify   multipart field "image" (JPEG), query k=5&tta=0   ->  200 {"matches": [{"uuid", "title", "artists", "score"}], "ms": int}; 400 on a non-image body
GET  /health     ->  {"model": str, "vectors": int}
```
FastAPI app. `Identifier` and `Catalog` are built once at startup from the `RS_MODEL` env var. Run with `uvicorn pipeline.retrieval.serve:app --host 0.0.0.0 --port 8008`.

- [ ] Step 1: Failing test with `TestClient` and a stub `Identifier` injected through a dependency override: post a fixture JPEG and assert the JSON shape; post text and assert 400.
- [ ] Step 2: Implement. Tests pass. Run it for real: `curl -F image=@<gold photo> localhost:8008/identify` returns the right album.
- [ ] Step 3: Commit `feat(pipeline): fastapi identify service`.

### Task 13: the app scans continuously and plays the preview

**Files:**
- Create: `app/src/app/api/identify/route.ts`, `app/src/lib/scanner.ts`, `app/src/lib/scanner.test.ts`
- Modify: `app/src/app/page.tsx`, `app/package.json` (add vitest)

**Interfaces (produces):**
```ts
// api/identify/route.ts   POST {image: dataUrl} forwards the JPEG to `${RETRIEVAL_URL}/identify`
//   returns {matches: [{title, artists, score}], source: "retrieval"}
//   on fetch failure or missing RETRIEVAL_URL it runs the existing OpenAI route's logic and returns {matches: [{title, artists, score: 1 - error}], source: "openai"}
// lib/scanner.ts
export const LOCK_THRESHOLD: number   // from the Task 9 report
export const MIN_CONSECUTIVE = 2
export function shouldLock(scores: number[]): boolean   // true when the last MIN_CONSECUTIVE scores are all >= LOCK_THRESHOLD
```

Behavior: a "Start" tap (a user gesture) starts the camera and unlocks one `<audio>` element. A square framing guide overlays the viewfinder. Every 800 ms the page grabs the center square at 512 px and posts it. When `shouldLock` is true the page stops scanning, shows cover, title and artist, calls `/api/deezer/album`, sets `audio.src` to the preview and plays. A "Scan again" button resumes. If Deezer has no match or no preview, the identification still shows without audio (Review focus 5).

- [ ] Step 1: Failing vitest cases for `shouldLock`: below threshold, one frame above, two frames above.
- [ ] Step 2: Implement the route, `scanner.ts` and the page changes.
- [ ] Step 3: `cd app && npm run lint && npx tsc --noEmit && npx vitest run` clean. Manual test on the phone over Tailscale or cloudflared with the Task 12 service running: a record locks within about 3 seconds and audio plays; a blank wall never locks; with `RETRIEVAL_URL` unset the GPT fallback still identifies.
- [ ] Step 4: Commit `feat(app): continuous scanning with local retrieval and preview playback`. Then H5b.

## Phase 4: deferred tracks (do not start without a new decision)

- **T14 Detector** (needs H3 and a fresh annotation set). Export annotations, train YOLOv8n on them, add a `crop="detector"` mode to `identify`, re-run the eval. Adopt only if it beats the framing guide.
- **T15 Release-group dedup.** Re-slim from the tar.xz adding `release-group.id`. Collapse the gallery to one cover per release group, which cuts the index several-fold and removes most same-cover ambiguity.
- **T16 On-device.** Product-quantize the index (faiss IVF-PQ), export the embedder to Core ML, measure latency on the phone. Removes the server from D6.

---

## Self-review notes

- Spec coverage: every row of A3 maps to a task. Stages 1 to 4 are kept and re-pathed in T3. Stage 5 is T4 and T10. Stage 6 is T5 and T11. Stage 7 is T5. Stage 8 is T14. Stage 9 is T12. Stage 10 is T13. The B1 hazards are fixed in T2 and T3, the B2 bugs in T3 to T6, B3 in T4 and D2, B4 in T2 (not copied) and T6.
- Names used across tasks are defined once each: `Embedder`, `load_embedder`, `tta_views`, `embed_prefix`, `pending_prefixes`, `Catalog`, `Release`, `album_key`, `normalize`, `build_index`, `Identifier.identify`, `Match`, `GoldItem`, `EvalReport`, `DeezerAlbum`, `findAlbum`, `normalizeName`, `shouldLock`.
- Review focus items 1 to 5 each have an owning task: T3, T4, T9 with T13, T5, T13.
