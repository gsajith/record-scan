"""One-off: write META_DIR/covers-2025-08-13.jsonl, the cover list the first catalog update starts from.

    python -m pipeline.legacy.seed_covers_2025

Each 2025 release with a cover id becomes one primary cover at <mbid[:2]>/<mbid>.jpg. Releases in
failed_cover_fetches.json are checked on the drive and left out when the file is missing, so
fetch_new_covers looks them up again instead of listing a file that was never downloaded.
"""

import json
from dataclasses import asdict

from pipeline import config
from pipeline.metadata.catalog import Cover


def main() -> None:
    image_root = config.require_image_root()
    failed = json.loads((config.META_DIR / "failed_cover_fetches.json").read_text())
    missing = {m for m in failed if not config.image_path(m).exists()}
    releases = json.loads((config.META_DIR / "slim_releases_with_covers.json").read_text())
    out = config.META_DIR / "covers-2025-08-13.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for r in releases:
            if r.get("cover_id") is not None and r["id"] not in missing:
                c = Cover(r["id"], str(r["cover_id"]), f"{r['id'][:2]}/{r['id']}.jpg", True)
                f.write(json.dumps(asdict(c)) + "\n")
                n += 1
    print(f"{n:,} covers -> {out}; left out {len(missing):,} of {len(failed):,} failed 2025 downloads "
          f"missing from {image_root}")


if __name__ == "__main__":
    main()
