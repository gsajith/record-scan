# Legacy pipeline scripts

These scripts built the cover image set and the first metadata index. They finished their
work and are kept here for reference only. They are not tested or maintained, and their
imports and paths may no longer match the rest of `pipeline/`.

## August 2025 data pipeline

The run started with `pipeline/metadata/slim_file.py`, which slimmed the MusicBrainz dump
and still lives in `metadata/` for the annual re-slim. The four scripts below ran after it.

- `parallel_fetch_cover_art_urls.py` queried the Cover Art Archive for each release and
  recorded its front cover id.
- `fetch_cover_art.py` downloaded the covers at 500px, 3,128,310 in total.
- `move_images_to_subdirectories.py` split the flat image folder into `<uuid[:2]>/` prefix
  folders, the layout the image root still uses.
- `generate_uuid_to_metadata.py` built `uuid_to_metadata.json` from the slimmed releases.

## November 2025 detector track

- `export_yolo_dataset.py` exported annotations from Supabase into a YOLOv8 dataset.

## Why they are retired

- `fetch_new_covers.py` from the identifier plan's Task 6 replaces both fetchers.
- `catalog.sqlite` from the identifier plan's Task 1 replaces `uuid_to_metadata.json`.
- The detector track is deferred and its Supabase backend is gone.

The originals also exist in `~/Projects/record-scan-data/_archive-mbdump-2025-11/` and in
git history.
