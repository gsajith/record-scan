# Record Scan identifier plan

> **How this plan runs:** tasks are done in order with checkbox (`- [ ]`) steps. From Task 2 on, the controller session (Opus 5.5) implements tasks itself; if subagents are ever used, dispatch them with model `opus`. Progress is logged in `.superpowers/sdd/2026-10-24-identifier-plan/progress.md`.

**Goal:** Build the record identifier the research report specifies: one vector per deduplicated cover, a query encoder trained on simulated phone photos, a phone-sized student, served hybrid first and on-device if a compressed index proves itself. This plan supersedes Tasks 4 to 13 of `docs/superpowers/plans/2026-10-24-record-scan-recovery.md`; Tasks 1 to 3 of that plan are done and its Parts A to F remain the project inventory.

**Architecture:** Phase A builds the measuring instruments (labeler, gold and wild sets, catalog search, retrieval core, gallery pass, catalog update job, export spike) and runs the frozen-backbone sweep. Nothing in Phase A trains a model. Phases B to D (hybrid v1, simulator, head, fine-tune, distillation, index compression, lock calibration) are gated on Phase A's numbers and are outlined here; their tasks get written at full detail after Step 4's measurements exist, because the model and the dimensions they depend on are unknown until then.

**Tech stack:** Python 3.12, PyTorch 2.8 (MPS), Hugging Face transformers and timm for backbones, open_clip for the CLIP baseline, hnswlib and faiss-cpu, WebDataset shards, SQLite FTS5, imagehash, ffmpeg. Next.js 16 (the `annotator/` app, repurposed as the labeler) with zod and Node 24's built-in `node:sqlite` (chosen over better-sqlite3 in Task 3: no native build, and npm 11 would need its install script approved). coremltools 9 and ExecuTorch 1.x for the export spike.

**Spec:** `docs/research/reports/On device album cover retrieval 2026.md` (the twelve gated steps and the evidence behind them) plus the research notes beside it. Decisions D8 to D13 below extend the recovery plan's D1 to D7. Where this plan and the report differ, this plan wins; it records why.

**Contents:** Global constraints, Review focus, Decisions, What only Gautham can do (tasks and checkpoints), Layout, Phase A (Tasks 1 to 9), Phases B to D (outline), Data you can delete and when, Self-review notes.

**Status:** Tasks 1 and 2 done. Task 6 code is in; its full cover fetch runs until about 2026-10-27, then a catalog rebuild finishes it. Task 3 is built; its checkpoint, an hour of real labeling (#13), is next. Task 4 done: seven backbones, identify and eval; Gautham signed off the eval rules (#17) with albums as release groups.

## Global constraints

- Everything from the recovery plan's Global constraints still applies: never `git add` data, agents delete nothing but regenerable artifacts, Conventional Commits, one venv at `pipeline/.venv`, synced from `pipeline/requirements.lock` (how-to atop `requirements.txt`), `require_image_root()` before touching the drive, never `os.makedirs` under `/Volumes`, TypeScript without `any`.
- Data layout under `RS_DATA_DIR` (default `~/Projects/record-scan-data`): `metadata/` (JSON, `catalog.sqlite`), `gold/` and `wild/` (each with `photos/`, `inbox/`, `labels.jsonl`, and `sources.jsonl` recording each photo's inbox frame or web address and, for wild, its fingerprint), `gallery256/` (shards, hashes, clusters), `musicbrainz-dump-<date>/` (downloaded dumps), `embeddings/<model>/`, `exports/`, `lightbank/`. Config exposes each as a constant.
- Pipeline modules read `config.X` at call time, never `from pipeline.config import X` at import. Tests generate fixtures into `tmp_path`; no JSON or image fixtures are committed except tiny generated ones under `pipeline/tests/fixtures/` created by `conftest.py`.
- One label record format, defined once in `pipeline/gold/labels.py` and mirrored in `annotator/lib/labels.ts`, with a shared test vector so the two never drift.
- Gold photos and the `eval` half of the wild set never enter any training, simulator fitting, or background bank. The split is a deterministic hash, assigned once, shown read-only.
- External services: Cover Art Archive thumbnails may be hot-linked by the labeler for display only. Requests to coverartarchive.org and the MusicBrainz web service send a `User-Agent` naming the project and a contact address, back off on 429 and 5xx, and the web service gets at most one request per second. Catalog updates use MetaBrainz dumps rather than per-release web service calls (D11).
- Every measured number lands in `docs/results/<date>-<topic>.md` with the command that produced it, the git SHA, and a hash of the eval config, so the report's decision gates are checked against written evidence. A number without provenance counts as no number.
- Eval integrity (from Dan Luu's "There's no point at which turning your brain off will work", 2026): the hit rule, the eval and sim split, the distractor sampler and its seeds are locked in their own reviewed commit before any model work. A later change to any of them needs a recorded ruling in the ledger and its own commit; it never lands in a task that also touches model, simulator or training code. Reviewers of model tasks check the diff for edits to `eval.py`, `labels.py`, seeds and hit logic and report any as Important.
- Sealed holdout: Gautham keeps about a third of his records (videos and frames) outside `RS_DATA_DIR`, in a folder no agent is told about, labeled with the same labeler against a second data dir. It is scored once per phase, by Gautham, with the same `eval.py`. Agents never see it.
- Surprising numbers are bugs until shown otherwise: if a frozen backbone scores far below its published benchmark, or a student beats its teacher, the task stops at that result, the controller checks preprocessing and data before anything else, and no training is dispatched to "fix" it.
- Every eval run writes a miss gallery, `docs/results/<run>/misses.html`: each missed photo beside the top five proposed covers with scores, plus the twenty lowest-scoring hits. Gates are decided from the gallery and the table together, never from the table alone.

## Review focus

1. Two labels for one image: the labeler appends a new line and the loader keeps the latest line per `sha1`; a re-label never silently keeps the old answer. Test in Task 2.
2. Split drift: the Python and TypeScript `split_for(sha1)` return the same answer for the same input; the same five literal sha1 values are asserted in both suites. Tests in Tasks 2 and 3.
3. Catalog search misses on accents and case: "bjork" finds Björk and "the beatles" finds "Beatles, The"; FTS5 is built with `unicode61 remove_diacritics 2`. Test in Task 1.
4. A corrupt JPEG or an unmounted drive mid-run: the gallery pass logs and skips the file, finishes the prefix, and resumes per prefix on rerun; it never creates folders under `/Volumes`. Test in Task 5.
5. Lost or clobbered covers in a catalog update: the cover fetcher never overwrites or deletes a file on the drive, keeps an old cover while it is still upstream, backs off on 503, and resumes from its lookup log after a crash without repeating a request. Test in Task 6.
6. Metric gaming: a model task that changes the hit rule, the split, seeds or test assertions so that a number improves without the model improving. Reviewers check for it in every task from Task 4 on; the eval-integrity constraint forbids it in the same commit.

---

## Decisions

- **D8 Labeler is local-only.** The `annotator/` app runs on the Mac, writes files under `RS_DATA_DIR`, and has no cloud backend. It listens on 127.0.0.1 only and refuses cross-site API calls; no Tailscale or other remote access (Gautham, 2026-10-25). Supabase code is removed. (Gautham, 2026-10-24.)
- **D9 Candid internet photos are allowed for evaluation and simulator fitting, with a license filter and a one-time split.** Sources: Flickr Creative Commons, Wikimedia Commons, Pexels, Unsplash, Creative Commons YouTube frames. No Instagram. Half (by hash) feeds simulator statistics and backgrounds; half is evaluation. Revised 2026-10-26 (Gautham): the labeler doesn't ask for or check a license or source; it records the image address when a drag provides one and stores the license as `other-permitted`. Because licenses aren't tracked, nothing from the wild set is used to train weights. A wild photo that looks like one already added (perceptual hash) is refused, so one picture can't land in both halves.
- **D10 Gallery working copy at 256 px short side, JPEG quality 90, in WebDataset tar shards keyed by MBID.** One pass over the drive produces it together with a 64-bit pHash per cover. Later steps read the shards, not the drive.
- **D11 Catalog updates come from a fresh MusicBrainz dump, re-slimmed, plus the Cover Art Archive dump.** (Revised by Gautham, 2026-10-25.) The first version planned to list new items through Internet Archive search, but every Cover Art Archive item there is `noindex`: the scrape API finds 230 of millions. So an update downloads the latest JSON `release.tar.xz` and re-slims it with `release_group` kept, and diffs it against the catalog. The 162 MB `mbdump-cover-art-archive.tar.bz2` from the full export tells which old covers are still upstream and which old releases gained a front. coverartarchive.org gives the best front for each release with no cover left, as in 2025. No MusicBrainz web service calls, so a run takes hours, not days. Old covers stay while they exist upstream and new fronts are added beside them as `<mbid>-<cover_id>.jpg` on the drive; covers deleted upstream leave the catalog but stay on disk. MetaBrainz publishes dumps twice a week, so the cadence is whatever is useful.
- **D13 Oldest target phone (open, Gautham decides before Task 8 reports).** The 30 ms p95 latency bar and the memory budget are defined against the oldest phone the app must run well on, one iPhone and one Android. Until named, the export spike reports numbers for every phone available and the gate cannot be applied.
- **D12 Mobile shell framework: proposed Expo (React Native) with VisionCamera frame processors and `react-native-executorch` for one codebase on both platforms; native Swift and Kotlin if the export spike shows the React Native path cannot hit the latency bar.** Not needed in Phase A: iOS device timing comes from Xcode's Core ML performance report on a connected iPhone, which needs no app. Gautham confirms or overrides before Phase B's Task 12.
- **D14 Two use cases, both first-class (Gautham, 2026-10-27).** (1) Held up: the user holds a record so its cover fills most of the screen. This must be close to perfect and carries the r1 >= 0.90 bar. (2) Crate flipping: the user films while flipping through a crate, and the app finds the front record and plays its preview in real time, staying quiet between records. The app finds the cover itself in both; a square guide is at most an optional hint. This revises the recovery plan's D4 (framing guide instead of a detector): whether the app crops, rectifies or detects the cover is decided by Task 7's crop study on real crate frames. Reports show the two cases side by side through the `crate` tag.

## What only Gautham can do

### Tasks

- **H1** Mount the Seagate drive before Tasks 5, 8 (distractors) and anything in Phase B.
- **H2** Gold set, revised 2026-10-27 for D14 and a 12 to 15 record collection: your own records plus a record store session (ask the staff first), 80 to 150 different records in all. For each record, a ten-second held-up clip (angle, distance, light, with and without your hand). Separately, 30 to 60 second videos of flipping through a crate at normal speed, at home and in the store, sampled at 2 frames per second; label each frame with the front record and tag it `crate`. Plus 100 to 200 negative frames (walls, shelves of spines, books, CDs, screens) and a few records missing from Cover Art Archive. Targets count different records, not frames: 40 for the smoke sweep, 150 with at least 300 positive frames for the full sweep. Drop the videos into `~/Projects/record-scan-data/gold/inbox/`; Task 2's sampler turns them into frames and Task 3's labeler labels them.
- **H3** Light bank and geometry target. (a) A glossy white 12-inch square (a blank glossy sleeve, or glossy photo paper mounted on board; matte paper shows shadows and color cast but no specular glare, and real sleeves are mostly glossy laminate). Photograph it 100 to 200 times in every place records live for you: shelf, desk lamp, window light, dark room, colored bulb, phone flash, a record store. Shoot half the frames with the card inside a clear plastic outer sleeve, since store copies are often wrapped. (b) One geometry target: a printed cover with ArUco markers in the four corners, shot 50 times at the tilts and distances you would really hold a phone, which gives ground-truth corners for Step 3. (c) Optional and cheap: 10 to 20 real catalog covers printed on glossy paper at 12 inches, a "printed" set that can be photographed in any environment with a known answer; it is scored as its own set, never mixed into gold. Drop everything into `~/Projects/record-scan-data/lightbank/inbox/`.
- **H4** Wild set: collect 300 to 1,000 licensed candid photos of people holding or flipping through records (see D9) by pasting or dropping them into the labeler (see D9 for what it records).
- **H5** Accept the DINOv3 license on Hugging Face (`facebook/dinov3-*` are gated), run `huggingface-cli login` on the Mac, and put the token in the pipeline `.env` as `HF_TOKEN`. Also read Apple's MobileCLIP2 model terms and say yes or no to including it in the sweep.
- **H6** Confirm or override D12 before Phase B.
- **H7** Before Phase B: an account on one GPU provider (RunPod or Modal) with a small balance; the whole plan budgets under $100.
- **H8** Sealed holdout, revised 2026-10-27: 40 to 50 records from a different store or a different trip than gold, shot the same way (held-up clips and flipping videos), so it also tests a place the gold set never saw. Keep the videos and frames in a folder outside `RS_DATA_DIR` that no agent is told about, laid out as `<folder>/gold/inbox/`, with `<folder>/metadata/catalog.sqlite` linked to the real catalog so the labeler starts. Label them by running the labeler with `RS_DATA_DIR` pointed there. Score them yourself once per phase.

### Checkpoints

Review by agents gates code quality. These five gates are judgment and feel, which the essay above argues cannot be delegated. A task that reaches its checkpoint is not complete until Gautham has done the item and the ledger records it.

| After | Gautham does | Why an agent cannot |
|---|---|---|
| Task 3 | Uses the labeler for an hour on real inbox items before it is called done; friction notes go back as a fix round | the reviewer can check routes, not whether fifty labels in a sitting is bearable |
| Task 4 | Reads `pipeline/retrieval/eval.py` and `pipeline/gold/labels.py` himself (the hit rule, the split, the slices) and signs off in the ledger | everything downstream rests on this code; an eval that quietly flatters the model invalidates every later number |
| Task 7 | Picks the teacher and student shortlist from the miss gallery and the table together, and names D13 | the gallery shows what kind of photo fails; the table cannot |
| Task 11 | Looks at fifty synthetic photos beside fifty real ones of the same covers and says whether they could be mistaken for each other | the domain classifier measures a gap; a person recognizes what is wrong with the fake |
| Task 17 | Uses the app in a record store on the sealed holdout records | the ship bar is a feeling of trust, measured last by the numbers |

Also before Phase B spends money: ten minutes reading the DINOv3 license text and Table 14 of the DINOv3 paper, the two sources the whole backbone choice leans on, because the research report flagged its own extraction of one of them as unverified.

## Layout

```
record-scan/
  pipeline/
    config.py                      + CATALOG_DB, WILD_DIR, GALLERY256_DIR, EXPORTS_DIR, LIGHTBANK_DIR, RESULTS_DIR
    metadata/
      catalog.py                   SQLite-backed Catalog: release(), search(), album_key(), find()
      slim_file.py                 JSON dump -> releases-<date>.jsonl + artists-<date>.json (Task 6)
      build_catalog_db.py          slim + covers list -> catalog.sqlite (FTS5, covers table)
      fetch_new_covers.py          new dump + CAA dump -> covers on the drive + covers-<date>.jsonl (Task 6)
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

### Task 1: catalog in SQLite with full-text search (done, f270bb2)

**Files:**
- Create: `pipeline/metadata/catalog.py`, `pipeline/metadata/build_catalog_db.py`, `pipeline/tests/test_catalog.py`
- Modify: `pipeline/config.py` (new constants listed under Layout), `pipeline/requirements.txt` (no new deps; sqlite3 is stdlib)

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
Load `slim_releases_with_covers.json` and `release_slim_artists.json` whole (about 6 GB of RAM, fine on the Mac), insert in one transaction with `executemany`, then build the FTS table. Also ingest `delta/releases/*.jsonl` when present (Task 6's output, which carries `release_group`). Superseded by Task 6: `build(releases_jsonl, artists_json, covers_jsonl, out)` now reads one dated slim plus its covers list, and the delta folder is gone.

- [x] Step 1: Failing tests: `test_normalize` ("The Beatles" -> "beatles"; "Beatles, The" -> "beatles"; "Björk" -> "bjork"; "Abbey Road (Remastered)" keeps its words), `test_build_and_search_tmp_db` (three fake releases written to `tmp_path` JSON, build, `search("bjork")` returns the Björk release, `search("beatl")` prefix-matches, `release(mbid)` round-trips), `test_find_exact`, `test_missing_db_exits`.
- [x] Step 2: Run, FAIL.
- [x] Step 3: Implement. Add the new config constants.
- [x] Step 4: `pipeline/.venv/bin/pytest pipeline/tests/test_catalog.py -v` passes. Real build: `python -m pipeline.metadata.build_catalog_db` writes `catalog.sqlite` (expect 3,129,812 rows, 600 MB to 1 GB, under 10 minutes); `python -c "from pipeline.metadata.catalog import Catalog; print(Catalog().search('abbey road')[:3])"` returns Beatles releases.
- [x] Step 5: Commit `feat(pipeline): sqlite catalog with full-text search`.

### Task 2: gold and wild set format, loader, and video frame sampler (done)

**Files:**
- Create: `pipeline/gold/__init__.py`, `pipeline/gold/labels.py`, `pipeline/gold/sample_frames.py`, `pipeline/tests/test_labels.py`, `pipeline/tests/test_sample_frames.py`

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

- [x] Step 1: Failing tests: `test_split_vectors` (five literal sha1 values with their expected split, the same five asserted in `annotator/lib/labels.test.ts`), `test_load_keeps_latest_label` (two lines for one sha1, second wins), `test_load_rejects_unknown_tag`, `test_sample_frames_writes_n_frames` (conftest generates a 3-second synthetic video with ffmpeg if `ffmpeg` is on PATH, else the test skips with a clear reason).
- [x] Step 2: Run, FAIL.
- [x] Step 3: Implement.
- [x] Step 4: Tests pass. `which ffmpeg` is present (if not, `brew install ffmpeg` is the one allowed install in this task; record it in the report).
- [x] Step 5: Commit `feat(pipeline): gold and wild label format with video frame sampling`.

### Task 3: the labeler (annotator/ repurposed, local-only)

**Files:**
- Delete: `annotator/lib/supabaseClient.ts`, `annotator/app/api/upload/route.ts`, `annotator/app/api/save/route.ts`
- Create: `annotator/lib/labels.ts`, `annotator/lib/data.ts` (paths under `RS_DATA_DIR`), `annotator/lib/catalog.ts` (`node:sqlite` read-only connection), `annotator/app/api/search/route.ts`, `annotator/app/api/ingest/route.ts`, `annotator/app/api/label/route.ts`, `annotator/app/api/queue/route.ts`, `annotator/app/api/image/[set]/[sha1]/route.ts`, `annotator/lib/labels.test.ts`
- Modify: `annotator/app/page.tsx` (new UI), `annotator/package.json` (add `zod`, `sharp`, `vitest`; remove `@supabase/supabase-js`), `annotator/README.md`, `.env.example` (add `RS_DATA_DIR`)

**Interfaces (produces):**
```ts
// lib/labels.ts  (mirror of pipeline/gold/labels.py)
export const KINDS, TAGS, LICENSES as const  // identical lists
export const LabelSchema = z.object({ file, sha1, kind, mbid: z.string().uuid().nullable(), tags: z.array(z.enum(TAGS)), source_url: z.string().url().nullable(), license: z.enum(LICENSES), split: z.enum(["eval","sim"]), notes: z.string(), labeled_at: z.string() })
export function splitFor(set: "gold" | "wild", sha1: string): "eval" | "sim"   // same rule as Python
// API
GET  /api/search?q=...&limit=20          -> { results: [{ mbid, title, artists, format, year?, thumb: "https://coverartarchive.org/release/<mbid>/front-250" }] }
POST /api/ingest   multipart { set, file, source_url? } | json { set: "wild", url }   -> { sha1, split, file }   // saves <set>/photos/<sha1>.jpg (re-encodes to JPEG, max side 1600), returns the existing record if the sha1 is already present, refuses (409) a wild near-duplicate
POST /api/queue   json { set, skip }      -> { next: { file, sha1, video, sourceUrl } | null, remaining, counts }   // photos/ without a label line, then <set>/inbox/ frames (moved to inbox/done/); video lets the page keep the album within one recording; counts per set and split
POST /api/label    json Label             -> { ok: true }   // validates with LabelSchema, split recomputed server-side, appends to <set>/labels.jsonl
GET  /api/image/<set>/<sha1>              -> the JPEG (so the page can show local files)
```
Page flow: pick set (gold or wild); paste (Ctrl+V image or URL), drop files or URLs; the next unlabeled photo or inbox frame loads on open and after each save, and S skips one; image fills the left pane; right pane has the search box (debounced, results with thumbnails, arrow keys to move, Enter to pick), a kind toggle (positive, negative, unknown) with hotkeys P N U, tag checkboxes with hotkeys 1 to 9, 0, - and = (twelve tags), / to focus search, notes, and Save (Enter). After Save the next item loads; the picked album, the kind and the video-wide `crate` and `rack` tags carry over between frames of the same video and reset for a new one; other tags reset on every frame (`annotator/lib/carry.ts`). A status strip shows counts per set and per split.

- [x] Step 1: Failing vitest cases: `splitFor` against the same five literal sha1 values as `SPLIT_VECTORS` in `pipeline/tests/test_labels.py`, `LabelSchema` rejects a wild label without license, accepts a gold label with `license: "own"`.
- [x] Step 2: Run, FAIL. Then implement routes and page; `npm run lint && npx tsc --noEmit && npx vitest run` clean.
- [x] Step 3: Manual check with `RS_DATA_DIR` pointing at a tmp folder: paste a cover image from the clipboard, search "abbey road", pick a release, save; the JSONL line validates with Python `load_labels`; a second label for the same image produces a second line and `load_labels` returns the newer one.
- [x] Step 4: `annotator/README.md` rewritten for the labeler: how to run, hotkeys, where files go, the license rule. Remove the Supabase sections and the stale `../pipeline/detector/export_yolo_dataset.py` path (that script now lives in `pipeline/legacy/`). Also put double quotes around the `RS_IMAGE_ROOT` value in the root `.env.example` so the file loads with dotenv parsers and with `source` alike.
- [x] Step 5: Commit `feat(annotator): local labeler for gold and wild photo sets`.

Progress: built in issues #10 (PR #36), #11 (PR #38) and #12 (PR #40). Steps 1 to 5 landed as those three PRs, not one commit. Since D9's 2026-10-26 revision, no test checks a wild license: wild labels store `other-permitted`. Gautham validated each PR by hand, including HEIC and the wild paste, drag and duplicate checks. The checkpoint (an hour of real labeling, #13) is still open.

### Task 4: retrieval core (embedders, index, identify, eval)

This merges the recovery plan's Tasks 4, 5 and 7 with the report's changes: seven backbones, one vector per image, three query views, lenient album-level scoring, per-slice recall, negatives.

**Files:**
- Create: `pipeline/retrieval/embedders.py`, `pipeline/retrieval/embed.py`, `pipeline/retrieval/identify.py`, `pipeline/retrieval/eval.py`, `pipeline/tests/test_embedders.py`, `pipeline/tests/test_embed.py`, `pipeline/tests/test_index_identify.py`, `pipeline/tests/test_eval.py`
- Modify: `pipeline/retrieval/build_index.py` (rewrite), `pipeline/requirements.txt` (+ `transformers==5.19.0` (4.55.4 predates DINOv3), `imagehash`, `webdataset`, `pyarrow`, `faiss-cpu==1.12.0`, `huggingface_hub`)
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
def embed_images(paths: list[Path], embedder: Embedder, batch: int, workers: int, errors_log: Path) -> tuple[list[str], np.ndarray]   # keyed by file stem (mbid); unreadable files logged to errors_log and skipped
def embed_prefix(prefix: str, embedder: Embedder, out_dir: Path, batch: int, workers: int) -> int        # writes out_dir/<prefix>.npz once (uuids U36, embs float16, signature), <prefix>.errors.log and <prefix>.done holding the signature; source param added with shards (Task 5)
def pending_prefixes(out_dir: Path, signature: str) -> list[str]  # prefixes not done by this signature; source param added with shards (Task 5)

# build_index.py   CLI: python -m pipeline.retrieval.build_index --model NAME [--M 32] [--ef-construction 200]
def build_index(shards: list[Path], dim: int, out_dir: Path, M: int, ef_construction: int) -> int   # verifies dims and one shared signature; writes index.hnsw + labels.npy (U36) + signature.txt

# identify.py   CLI: python -m pipeline.retrieval.identify photo.jpg --model NAME [--k 5] [--views 1|3]
@dataclass
class Match: mbid: str; title: str; artists: tuple[str, ...]; score: float
class Identifier:
    def __init__(self, embedder: Embedder, index_dir: Path, catalog: Catalog, clusters: dict[str, int] | None = None, ef: int = 128): ...
    def identify(self, img: Image.Image, k: int = 5, views: int = 3) -> list[Match]
        # embed each view, knn k*10 per view, max score per cluster (cluster = clusters[mbid] if given else Catalog.album_of, the release group), top k

# eval.py   CLI: python -m pipeline.retrieval.eval --model NAME --set gold|wild [--distractors 50000] [--views 1|3] [--px 224] [--seed 0] [--dry-run]
@dataclass
class EvalItem: label: Label; targets: set[str]   # target mbids: the labeled mbid plus every mbid sharing its album_key, plus its cluster members when clusters exist
def load_eval_set(set_dir: Path, catalog: Catalog, clusters: dict[str, int] | None) -> list[EvalItem]   # eval split only; positives need mbid; warns on unresolvable
def is_hit(result_mbid: str, item: EvalItem) -> bool
def build_mini_gallery(items: list[EvalItem], catalog: Catalog, distractors: int, seed: int) -> list[Cover]   # every cover of every target release + a seeded permutation draw over the catalog's covers table, skipping targets (existing_files.json is gone since Task 6)
@dataclass
class EvalReport:
    r1: float; r5: float; per_tag: dict[str, tuple[float, int]]; neg_max: float | None; pos_min_top1: float | None; micro_ap: float; rows: list[Row]
    @classmethod from_rows(rows) -> EvalReport; def to_markdown(self) -> str
def run_eval(model: str, set_name: str, distractors: int, views: int, px: int | None, seed: int, work_dir: Path) -> EvalReport   # writes docs/results/<date>-eval-<model>-<set>/report.md, report.json (with git SHA, config hash, command) and misses.html (each miss beside its top-5 covers with scores, plus the 20 lowest-scoring hits; thumbnails hot-linked from coverartarchive.org front-250)
```
Per-tag recall is reported with its count so a slice of three photos is visibly weak. `micro_ap` follows SSCD: positives and negatives ranked by top-1 score, average precision of "is a positive and hit".

- [x] Step 1: Failing tests: `test_each_embedder_unit_vectors` (parametrized, `@pytest.mark.slow`, two fixture images, shape and norm; skips gated models when `HF_TOKEN` is unset with a clear message), `test_query_views_three_sizes`, `test_embed_prefix_skips_corrupt_and_resumes`, `test_build_index_and_identify_roundtrip` (synthetic vectors, releases of one release group collapse), `test_is_hit_rules`, `test_eval_report_metrics_and_slices` (hand-built rows), `test_load_eval_set_uses_eval_split_only`.
- [x] Step 2: Run, FAIL. Step 3: Implement. Step 4: all pass; smoke: build an index from the 2025-11 shards (`clip-vit-b-32-aug13`, allow_pickle, cast uuids) and identify a CAA `front-500` for a uuid in prefix 00.
- [x] Step 5: `git rm` the two old scripts. Commit `feat(pipeline): retrieval core with seven backbones, query views and slice eval`.

Progress: issue #15 landed the CLIP path: `embedders.py` (CLIP only), `query_views`, `embed.py`, the `build_index.py` rewrite and `identify.py`, with their tests. The smoke run returned each of three prefix-00 covers first, and a phone photo of The Dark Side of the Moon came back first; numbers in `docs/results/2026-10-25-clip-baseline-smoke.md`. #18 added the other six backbones (smoke numbers in `docs/results/2026-10-27-backbone-smoke.md`), and #16 the eval, so Task 4 is complete. Steps 1 to 5 landed as those three PRs, not one commit. Notes:
- The two old scripts were already removed in d3c1455, so Step 5 has nothing left to `git rm`.
- CLIP loads as `ViT-B-32-quickgelu`, which matches how OpenAI trained the weights. The 2025-11 shards were built with plain `ViT-B-32` (GELU), so QuickGELU queries reach only 0.92 to 0.96 against their own covers. Never mix the two in one index.
- The six new backbones take their preprocessing from their model cards or reference code. DINOv3 resizes the whole image to a square with no crop; SigLIP2 does the same at 256 px; PE-Core does the same at 384 px with bilinear resizing, as Meta's reference code and training did (timm's config would crop instead). DINOv3 runs at any multiple of 16 px through `load_embedder(input_px=...)`; the others run at their native size only.
- ViTs embed with the final-norm CLS token and ConvNeXt with its normed global average pool. The plan's `pool="cls+mean"` option has no caller yet; the frozen sweep adds it if it wants to compare.
- Every shard and index records its embedder's signature (name, input px and a hash of the preprocessing). build_index refuses mixed shards and identify refuses an index from another signature, so a preprocessing change can't silently reuse old vectors; eval keys its gallery cache by signature. A change to weights or pooling doesn't alter the signature: give it a new model name. Shards from before signatures (the prefix 00 and 01 smoke runs) need re-embedding before use.
- torch 2.8's MPS backend has two bugs these models hit: PE-Core's attention pool aborts the process, and the fused `layer_norm` is wrong on flat areas of covers (DINOv3 ConvNeXt came out at cosine 0.995 to CPU). `embedders.py` works around both; all seven match CPU at 1.000000 on 200 covers. `test_each_embedder_matches_cpu_on_mps` (slow) checks it on flat and textured images; run it after any torch upgrade.
- timm moved to 1.0.30: 1.0.19 cannot download from huggingface_hub 2.x. CLIP vectors are bit-identical across the upgrade.
- `query_views` applies EXIF orientation first, so a portrait phone photo is not embedded sideways.
- `identify` widens its search past k*10 neighbours when one big group, such as the 38 Abbey Road releases, fills them and fewer than k albums come back.
- `embed.py` reads the drive only and has no `--source shards`, `--uuids` or `--px` yet. Tasks 5 and 7 add them along with their first callers. If the drive drops mid-prefix, it exits without marking the prefix done.
- Deferred from the PR #37 review, both for when Task 5's shards remove the drive bottleneck: run `preprocess` in the decode workers instead of the main thread, and cache `album_of` lookups in `Identifier` for eval runs.
- `identify` gives each album one slot, where an album is a MusicBrainz release group (`Catalog.album_of`, the rule the eval also uses since Gautham's #17 ruling).
- HNSW at the default `ef=128` returned the right top-1 album for 99.8% of 500 noisy queries on the 2025-11 index, with no gain up to `ef=1024`. Eval at 50k distractors can use exact search if that 0.2% matters.

Progress: issue #16 landed `eval.py`, the locked eval rules, in a commit of its own. Gautham signed them off in #17 on 2026-10-27 with the release-group change below. The rules:
- A hit is the labeled release, any release in its MusicBrainz release group (`Catalog.same_album`), or, once Task 5's clusters exist, any release in the same duplicate cluster as any of those album releases. The cluster rule is that broad on purpose: a lookalike of a reissue shows the same cover. A release with no release group stands alone.
- Ruling (Gautham, 2026-10-27, #17): albums are release groups, not normalized title plus first artist. Title matching merged different albums (Weezer's self-titled albums: 7 release groups under one key) and split one album ("Abbey Road (2019 Mix)" and the super deluxe edition sit in Abbey Road's 54-release group under other keys). `Catalog.album_of` holds the rule for both eval and `identify`. The catalog build indexes `release_group`; until the next rebuild, `same_album` scans, about 4 s per labeled positive.
- Only `split == "eval"` labels are scored. A positive whose release has left the catalog stays in as a certain miss, since the app could not find it either; `report.json` counts these as `positives_missing`.
- `negative` and `unknown` photos both count as photos with no right answer: they set `neg_max` and rank in micro AP.
- Per-tag numbers are r1 over positives with that tag, with the count. `pos_min_top1` is the lowest top-1 score among positives hit at rank 1. Micro AP breaks score ties against the model: wrong answers rank first.
- The gallery is every cover of every target release plus `--distractors` covers from a `numpy.random.default_rng(seed)` permutation of the catalog's covers table, skipping target releases. A catalog rebuild changes the draw, so `report.json` records a gallery hash over the covers actually embedded; runs are comparable only when both the config hash and the gallery hash match. A complete gallery's hash equals the one `--dry-run` prints. The config hash covers the set, distractors, views, px, seed and the labels file, but not the model. The draw is per cover, so a release with extra fronts is a little likelier to be drawn. `gallery_embedded` in `report.json` counts the covers that were readable; a drive that drops mid-embed stops the run instead of caching a partial gallery. Results go to `docs/results/<date>-eval-<model>-<set>-<config hash>/`, so a same-day run with other settings never overwrites another. `--px 224` and no `--px` hash differently for CLIP although they run the same; pass `--px` only to change the size.
- Retrieval goes through `Identifier` (HNSW, `ef=128`), the same path the app will use. Gallery embeddings are cached per model, px and gallery hash under `EMB_DIR/eval/`.
- Clusters are not wired into the CLI yet; Task 5 passes them once `clusters.parquet` exists. When it does, `identify` must group by release group and cluster together (connected components of both), as the hit rule does; today it groups by cluster alone when clusters are given, so two releases of one album in different clusters would take two slots. Test that interaction there.
- `misses.html` shows the misses, the 20 lowest-scoring hits and the 20 highest-scoring negatives. Thumbnails are each release's current front from the Cover Art Archive, which can differ from the extra front that matched.

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
List covers from the catalog's `covers` table (Task 6), not by globbing the drive: since Task 6 the drive also holds `<mbid>-<cover_id>.jpg` extra fronts and covers deleted upstream that the catalog dropped. Key shard entries by file stem so an mbid's extra fronts stay distinct. Both call `config.require_image_root()` first. Time the first prefix and print an ETA for the rest; expect hours at USB-3 speeds. Write to `config.GALLERY256_DIR` on the internal disk, never to the drive.

- [ ] Step 1: Failing tests on a tmp image root with three prefixes, one corrupt file, two identical covers under different mbids and one near-duplicate: shard contains the good files at 256 px, errors logged, `.done` resumes, clusters group the identical pair and the near-duplicate at distance <= 8 and not an unrelated image.
- [ ] Step 2: Run, FAIL. Step 3: Implement. Step 4: tests pass. Real run after H1: `--prefixes 00` first, report time and sizes in `docs/results/<date>-gallery-pass.md`, then `--all` under `nohup caffeinate -i`, then `cluster_duplicates`. Record N (deduplicated), cluster count, and the size histogram in the same results file.
- [ ] Step 5: Commit `feat(pipeline): gallery pass with 256px shards, phash and duplicate clusters`.

### Task 6: new covers since the dump (D11, done)

**Files:**
- Create: `pipeline/metadata/fetch_new_covers.py`, `pipeline/tests/test_fetch_new_covers.py`, `pipeline/tests/test_slim_file.py`
- Modify: `pipeline/metadata/slim_file.py` (a tested streaming module, keeps `release_group`), `pipeline/metadata/build_catalog_db.py` and `catalog.py` (JSONL input, `covers` table, `Catalog.covers()`), `pipeline/config.py` (drop `DELTA_DIR`)

**Interfaces (produces):**
```python
# slim_file.py   CLI: python -m pipeline.metadata.slim_file DUMP/release.tar.xz --date YYYY-MM-DD
def slim_record(obj: dict) -> dict | None   # {"release": 2025 slim keys + release_group, "artists": {...}}; None without cover art
def slim_dump(dump: Path, releases_out: Path, artists_out: Path) -> int   # streams mbdump/release through tar; META_DIR/releases-<date>.jsonl, artists-<date>.json
# catalog.py
@dataclass(frozen=True)
class Cover: mbid: str; cover_id: str; file: str; primary: bool   # file relative to IMAGE_ROOT
Catalog.covers(mbid) -> list[Cover]                                 # primary first
# fetch_new_covers.py   CLI: python -m pipeline.metadata.fetch_new_covers --date D --since PREV [--limit N] [--workers 16] [--dry-run]
def plan_covers(previous, releases, cover_release, new_fronts, fronts) -> Plan   # keep / download / lookup / dropped, no network; a new front replaces a non-front primary
def best_front(caa_json: dict) -> str | None                             # 2025 priority: approved front, pending front, any approved; needs a 500 px thumbnail
def get_with_backoff(session, url, sleep, tries=7) -> Response | None    # None on 404; 5 s doubling to 60 s on 429, 5xx, dropped connections
def run(date, since, session, limit, dry_run, workers, sleep) -> Counts  # reads META_DIR/covers-<since>.jsonl (a fixed snapshot); writes images to the drive, fetched-<date>.jsonl (resume log), covers-<date>.jsonl
# legacy/seed_covers_2025.py   one-off: covers-2025-08-13.jsonl from the 2025 slim, leaving out the 453 failed downloads missing from the drive
```
Commands, run from the repo root with the drive mounted:
```
D=~/Projects/record-scan-data/musicbrainz-dump-2026-10-24
curl -C - -o $D/release.tar.xz https://data.metabrainz.org/pub/musicbrainz/data/json-dumps/20261024-001001/release.tar.xz
curl -o $D/mbdump-cover-art-archive.tar.bz2 https://data.metabrainz.org/pub/musicbrainz/data/fullexport/20261024-102842/mbdump-cover-art-archive.tar.bz2
tar -xjf $D/mbdump-cover-art-archive.tar.bz2 -C $D mbdump
python -m pipeline.legacy.seed_covers_2025                                        # first update only
python -m pipeline.metadata.slim_file $D/release.tar.xz --date 2026-10-24        # about 40 min
python -m pipeline.metadata.fetch_new_covers --date 2026-10-24 --since 2025-08-13 --dry-run
python -m pipeline.metadata.fetch_new_covers --date 2026-10-24 --since 2025-08-13 --limit 500
python -m pipeline.metadata.build_catalog_db
nohup caffeinate -i python -m pipeline.metadata.fetch_new_covers --date 2026-10-24 --since 2025-08-13 > $D/fetch.log 2>&1 &
python -m pipeline.metadata.build_catalog_db                                      # again when the full run ends
```
The next update passes the new dump's date as `--date` and this one's as `--since`, and so starts from this update's `covers-<date>.jsonl`. Rebuilding the catalog mid-run is safe: the fetcher never reads it.

- [x] Step 1: Failing tests on generated fixtures: `slim_record` keeps the release group; `best_front` priority; `plan_covers` keeps old covers still upstream, adds new fronts, drops deleted ones, promotes an extra when the primary is gone, and looks up new releases; `get` backs off on 503, 429 and dropped connections; `run` resumes after a `--limit` run and after a crash mid-write without repeating a request, never overwrites or deletes a 2025 file, stops when the drive goes away, and `--dry-run` makes no requests; `covers_after` keeps one primary per release; `slim_dump` skips a malformed line; the catalog build takes release groups and covers.
- [x] Step 2: Run, FAIL. Step 3: Implement. Step 4: tests pass.
- [x] Step 5: Live: `--dry-run` backlog in `docs/results/2026-10-25-catalog-update.md`; `--limit 500` end to end; catalog rebuilt with `release_group` filled; full run started under `nohup`.
- [x] Step 6: Commit `feat(pipeline): update covers and metadata from a fresh musicbrainz dump`.

### Task 7: run the frozen sweep (Step 2) and the crop study (Step 3)

Needs H1, H2 frames labeled through the labeler (at least the smoke-run size: 40 different records and 20 negatives; the full 150 records later), H5. Every table states how many different records stand behind it, and shows held-up and `crate` frames side by side (D14); per-frame r1 stays the locked metric.

**Files:**
- Create: `pipeline/retrieval/sweep.py`, `pipeline/retrieval/crop_study.py`, `docs/results/<date>-frozen-sweep.md`, `docs/results/<date>-crop-study.md`

**Interfaces (produces):**
```python
# sweep.py   CLI: python -m pipeline.retrieval.sweep --set gold [--models all|names] [--distractors 50000] [--px 224 448]
def run_sweep(models, set_name, distractors, pxs, views) -> Path   # one eval per (model, px) with views=3; a summary table (model, params, dim, px, r1, r5, micro_ap, images/s on the M1) plus per-tag tables; applies the report's gates: shortlist = top 2 teachers of any size and top 2 students <= 30M params; prints whether any student clears r1 >= 0.90 and r5 >= 0.97 (the skip-training exit)
# crop_study.py   CLI: python -m pipeline.retrieval.crop_study --model NAME --set gold
def run_crop_study(model, set_name) -> Path   # run on held-up and real crate frames separately (D14); treatments: full frame, plain center crop, 3 views, OpenCV contour rectify (largest 4-point contour >= 20% of frame, fallback to plain), and when running on macOS with Vision available via pyobjc: VNDetectRectanglesRequest rectify; per-photo miss cause from tags (tilt/partial -> framing, glare -> glare, text-only, unknown-not-in-catalog, near-duplicate confusion); gate: framing+skew share of remaining misses < 25% closes the detector track. Also a visibility curve: every gold positive re-queried with its bottom 10, 20, 30 and 40 percent masked by a neutral band and by a strip cut from another cover (the crate case), reporting r1 at 90, 80, 70 and 60 percent visible, so the point where the frozen model breaks is a number before any training
```
- [ ] Step 1: Tests for the gate logic and the summary table on hand-built eval JSON (no models).
- [ ] Step 2: Implement; run the smoke sweep when 40 different records are labeled, the full sweep at 150. Each run writes its results file with the command line at the top.
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

- [ ] Update `README.md` (the Layout section, how to run each instrument, where results live) and `CLAUDE.md` (data layout, the no-training-on-eval rule, HF token, rate limits). Commit `docs: phase a instruments and data layout`.

## Phases B to D: outline, to be detailed after Task 7 and Task 8 report

Each later task is written at full detail only once the measurements it depends on exist. The gates are the report's; thresholds are the report author's judgment and may be revised with a recorded ruling.

- **Task 10, Step 5: full-gallery embedding and hybrid v1.** Rent a GPU, embed the deduplicated gallery with the two teacher candidates from the shards, rerun eval at full scale, pick the teacher, build the faiss HNSW index, serve from `pipeline/retrieval/serve.py` (FastAPI `POST /identify`, JPEG in for v1), and point `web/` at it with the GPT route as fallback (this absorbs the recovery plan's Tasks 6, 12 and 13). Gate: full-gallery r1 drop over 5 points sends Step 8 to mine hard negatives.
- **Task 11, Step 6: simulator and calibration.** `pipeline/sim/`: a kornia pipeline with two layers of variation. The fitted layer sets the common case: tilt from rectangle corners, exposure and color cast from per-channel fits against the matched clean cover, blur and glare area, all measured on the wild `sim` split, plus the real light fields from the H3 bank applied multiplicatively. The randomized layer covers the tail nobody enumerated: a generative model of illumination (random color temperature 2,000 to 10,000 K, exposure plus or minus 2 stops, random smooth multiplicative gradients and blobs, random saturation and grayscale) sampled wider than anything observed, which is standard domain randomization and costs little in the common case. Occlusion is its own op: hide the bottom 10 to 45 percent with the top strip of another cover or a plastic edge (the crate case), hide a corner with a hand cutout, with `partial`-style side cuts at lower probability; the asymmetric pairing teaches the model to produce the full cover's vector from 60 to 70 percent of it. A background bank from the `sim` split. Two gap tests: a domain classifier (synthetic vs real crops; AUC near 0.5 or the simulator is missing something visible, and its saliency says what) and the recall-gap check (synthetic within 5 points of real, same slice ranking, including the visibility curve from Task 7). Also measures battery and thermal at 5 to 10 fps on the test phones alongside Step 4's numbers.
- **Task 12, mobile shell (D12).** Camera, square guide, on-device embedding of the shortlisted student, POST of the vector or JPEG to the server, result and preview playback. Device timing moves from Xcode reports to the real app.
- **Task 13, Step 7: projection head on cached frozen features** (M1). Gate: gain >= 5 points sends to Task 14; under 2 skips it.
- **Task 14, Step 8: teacher fine-tune, SSCD-style** (rented GPU). Gate: beats Task 13 by >= 3 points, no slice loses more than 3.
- **Task 15, Step 9: student distillation** against precomputed teacher targets. Ship the smallest student within 2 points and under 30 ms p95.
- **Task 16, Step 10: index compression study** (faiss IVF-PQ64, PQ32, OPQ, binary + rescore). Gate: PQ64 within 2 points of exact search ships on-device as an asset pack; otherwise stay hybrid.
- **Task 17, Step 11: lock calibration and app integration.** 99% lock precision for both D14 modes: a held-up record locks fast; while flipping, the app plays only once the same front record agrees over several frames, stays quiet between records, and switches when the front record changes. Three agreeing frames with scores fused across frames (a crate shot gets several partial views as the phone moves, and the fused score beats any single frame), OCR fallback for low-margin and crate results (the visible top of a sleeve usually carries the artist and title), ship bar per the report.
- **Catalog update job:** Task 6 rerun on a newer dump plus catalog rebuild plus incremental embedding of the new `covers-<date>.jsonl` entries with the shipped teacher.

---

## Data you can delete, and when

Nothing here is deleted by an agent. Each row names the task whose completion makes the item dead; after that task's review is clean, the controller lists the row in its report and Gautham deletes it.

| Item under `~/Projects/record-scan-data` | Size | Removable after | Why |
|---|---|---|---|
| `metadata/uuid_to_metadata.json` | 840 MB | Task 1 (catalog.sqlite built, tests pass) | `Catalog` replaces it; regenerable from `slim_releases_with_covers.json` in a minute anyway |
| `_archive-album-annotator-2025-11/` | about 28 MB after cleanup | Task 3 (labeler merged) | the GitHub repo gsajith/album-annotator is archived and holds the same history |
| `_archive-mbdump-2025-11/` | about 1.1 GB after cleanup, almost all `.git` | Task 6 (catalog update merged) | the scripts are in `pipeline/legacy/`; only the August 2025 commit history remains here |
| `musicbrainz-dump-2025-08-13/` | 17 GB | Task 6 (catalog rebuilt from the 2026-10-24 slim) | superseded by the 2026-10-24 dump |
| `metadata/slim_releases.json`, `slim_releases_with_covers.json`, `release_slim_artists.json`, `existing_files.json`, `failed_cover_fetches.json`, `failed_cover_id_fetches.json` | 1.6 GB | Task 6 (full run finished and catalog rebuilt) | the 2025 slim and fetch state; `covers-2025-08-13.jsonl` carries every 2025 cover forward and already leaves out the failed downloads |
| `musicbrainz-dump-2026-10-24/release.tar.xz` | 22 GB | Task 6 (full run finished) | `releases-2026-10-24.jsonl` holds everything kept from it; download the next dump for the next update |
| `~/.cache/clip` (1.2 GB), `~/.cache/torch/hub/checkpoints` (1.2 GB), HF `models--openai--clip-vit-base-patch32` (1.1 GB) | 3.5 GB | now | the OpenAI `clip` package, DINOv2 hub weights and transformers CLIP are not used by the new retrieval core |
| HF `models--timm--vit_base_patch32_clip_224.openai` | 577 MB | keep | open_clip loads the `clip-vit-b-32` baseline from this cache |

Already removed on 2026-10-24 at Gautham's request: `_archive-mbdump-2025-11/venv` (1.4 GB) and `_archive-album-annotator-2025-11/node_modules` (419 MB). On 2026-10-25, after issue #15 merged: `embeddings/clip-vit-b-32-aug13/` (1.1 GB with the smoke index), the November 2025 augmented shards built with the GELU config.

---

## Self-review notes

- Spec coverage: report Steps 0 to 4 map to Tasks 2, 3 (Step 0), 5 (Step 1), 4 and 7 (Step 2), 7 (Step 3), 8 (Step 4). The catalog update (Gautham's question B) is Task 6 and D11. The labeler (his labeling-app idea) is Task 3 with D8. Simulator fidelity (question C) is Task 11's calibration design plus H3. Candid photos (question A) are D9, H4 and the wild set in Tasks 2 and 3. Phone hardware is covered in Task 8 (compute-unit placement, first-load, memory) and Task 11 (battery, thermal).
- Names reused across tasks are defined once: `Release`, `Catalog`, `normalize`, `album_key`, `Label`, `split_for`, `load_labels`, `append_label`, `Embedder`, `load_embedder`, `query_views`, `embed_images`, `embed_prefix`, `build_index`, `Identifier`, `Match`, `EvalItem`, `is_hit`, `EvalReport`, `run_eval`, `process_prefix`, `cluster`, `parity`.
- Review focus items 1 to 5 are owned by Tasks 2, 2 and 3, 1, 5, 6.
- Proportion: Phase A is specified to signature and test level; Phases B to D are deliberately outlines because their inputs do not exist yet.

