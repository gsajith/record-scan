# Labeler

A local app for labeling photos of records for the identifier's gold and wild sets. It runs
on this Mac only, reads the catalog, and writes plain files under `RS_DATA_DIR`. Nothing goes
to the cloud (decision D8 in `docs/superpowers/plans/2026-10-24-identifier-plan.md`).

## Run it

Needs Node 24 or later.

```sh
cd annotator
npm install
npm run dev        # http://localhost:3000
```

It reads the data folder from `RS_DATA_DIR`, defaulting to `~/Projects/record-scan-data`, and
needs the catalog built first (`python -m pipeline.metadata.build_catalog_db`). To label the
sealed holdout, point it at that folder instead:

```sh
RS_DATA_DIR=~/Projects/record-scan-sealed npm run dev
```

The server only listens on 127.0.0.1 and refuses requests from other sites or host names.

## Labeling

Pick **Gold** or **Wild** at the top. The next unlabeled photo loads by itself, and so does
the one after each save.

- **Gold** is your own photos of your own records, always licensed `own`. Drop record videos
  into `gold/inbox/`, run `python -m pipeline.gold.sample_frames gold` to turn them into
  frames, then label the frames here. You can also paste, drop or choose a photo.
- **Wild** is licensed photos from the internet (decision D9). Paste or drop an image, or an
  image link. Every wild photo needs the page it came from and a license. Dragging an image
  out of a web page fills in its address as the source. A photo that looks like one already in
  the wild set is refused, so one picture can't land in both the sim and eval halves.

The album and kind carry over between frames of the same video, so after the first frame you
add tags and press Enter. They reset for a new video or a pasted photo.

| Key | Does |
|---|---|
| P, N, U | Kind: positive, negative, unknown |
| 1 to 9, 0, -, = | Toggle a tag, in the order shown |
| / | Search the catalog |
| ↑ ↓, Enter | Move through results, pick one |
| Esc | Leave the search box |
| Enter | Save |
| S | Skip this photo for now |

No shortcut fires while you type in a field or hold ⌘.

## Where files go

Inside `RS_DATA_DIR`, for each of `gold/` and `wild/`:

| Path | Holds |
|---|---|
| `inbox/` | Videos and frames waiting to be labeled |
| `inbox/done/`, `inbox/unreadable/` | Frames already taken into `photos/`, and files that weren't images |
| `photos/<sha1>.jpg` | Every photo, upright, at most 1600 px, named by the sha1 of the file as added |
| `labels.jsonl` | One label per line; a re-label adds a line and the newest wins |
| `sources.jsonl` | Where each photo came from: its video frame, or a wild photo's page, license and fingerprint |

`pipeline/gold/labels.py` reads `labels.jsonl`, and `lib/labels.ts` mirrors its rules. Gold
photos are always in the eval split. Wild photos are split by hash, once, and the server
decides the split, never the page.

## License rule

A wild photo is kept only with its source page and one of the licenses in the list. Nothing
from the wild set trains weights unless its license is CC0 or equivalent and Gautham says so
(D9).

## Tests

```sh
npm test           # vitest
npx tsc --noEmit
npm run lint
```
