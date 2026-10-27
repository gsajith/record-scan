"""Build catalog.sqlite from a slimmed dump (slim_file) and its covers list (fetch_new_covers).

    python -m pipeline.metadata.build_catalog_db [--date YYYY-MM-DD] [--out PATH]

Reads META_DIR/releases-<date>.jsonl, artists-<date>.json and covers-<date>.jsonl; --date
defaults to the newest releases file.
"""

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

from pipeline import config
from pipeline.metadata.catalog import normalize

SCHEMA = """
CREATE TABLE artists (id TEXT PRIMARY KEY, name TEXT NOT NULL, aliases_json TEXT);
CREATE TABLE releases (
    mbid TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    primary_artist TEXT,
    artist_ids_json TEXT NOT NULL,
    format TEXT,
    cover_id TEXT,
    release_group TEXT,
    album_key TEXT NOT NULL
);
CREATE TABLE covers (mbid TEXT NOT NULL, cover_id TEXT NOT NULL, file TEXT NOT NULL, is_primary INTEGER NOT NULL);
CREATE VIRTUAL TABLE release_fts USING fts5(
    mbid UNINDEXED, title, artists, tokenize="unicode61 remove_diacritics 2"
);
"""


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build(releases_jsonl: Path, artists_json: Path, covers_jsonl: Path, out: Path) -> int:
    """Write the catalog to out + ".tmp", then move it over out. Returns the release count."""
    artists: dict[str, dict] = json.loads(artists_json.read_text())
    covers = list(_jsonl(covers_jsonl))
    primary_cover = {c["mbid"]: c["cover_id"] for c in covers if c["primary"]}
    releases = list(_jsonl(releases_jsonl))

    release_rows, fts_rows = [], []
    for rowid, r in enumerate(releases, start=1):
        ids = r.get("artist_ids") or []
        names = [artists[a]["name"] for a in ids if a in artists]
        primary = artists[ids[0]]["name"] if ids and ids[0] in artists else None
        cover = primary_cover.get(r["id"])
        release_rows.append((
            rowid, r["id"], r["title"], primary, json.dumps(ids), r.get("format"),
            cover, r.get("release_group"),
            f"{normalize(r['title'])}|{normalize(primary or '')}",
        ))
        # Index the normalized forms so search() and the index agree on "the", "&", apostrophes
        fts_rows.append((rowid, r["id"], normalize(r["title"]), " ".join(normalize(n) for n in names)))

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    try:
        db.execute("PRAGMA journal_mode = OFF")
        db.execute("PRAGMA synchronous = OFF")
        db.executescript(SCHEMA)
        with db:
            db.executemany(
                "INSERT INTO artists VALUES (?, ?, ?)",
                ((i, a["name"], json.dumps(a["aliases"]) if a.get("aliases") else None) for i, a in artists.items()),
            )
            db.executemany("INSERT INTO releases (rowid, mbid, title, primary_artist, artist_ids_json, format,"
                           " cover_id, release_group, album_key) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", release_rows)
            db.execute("CREATE INDEX releases_album_key ON releases (album_key)")
            db.execute("CREATE INDEX releases_release_group ON releases (release_group)")
            db.executemany("INSERT INTO covers VALUES (?, ?, ?, ?)",
                           ((c["mbid"], c["cover_id"], c["file"], int(c["primary"])) for c in covers))
            db.execute("CREATE INDEX covers_mbid ON covers (mbid)")
            db.executemany("INSERT INTO release_fts (rowid, mbid, title, artists) VALUES (?, ?, ?, ?)", fts_rows)
            db.execute("INSERT INTO release_fts (release_fts) VALUES ('optimize')")
    finally:
        db.close()
    os.replace(tmp, out)
    return len(release_rows)


def latest_dump_date(meta_dir: Path) -> str:
    dates = sorted(p.stem.removeprefix("releases-") for p in meta_dir.glob("releases-*.jsonl"))
    if not dates:
        raise SystemExit(f"No releases-<date>.jsonl in {meta_dir}. Run pipeline.metadata.slim_file first.")
    return dates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", help="dump date, YYYY-MM-DD (default: newest)")
    parser.add_argument("--out", type=Path, default=config.CATALOG_DB)
    args = parser.parse_args()
    date = args.date or latest_dump_date(config.META_DIR)
    if not (config.META_DIR / f"covers-{date}.jsonl").exists():
        raise SystemExit(f"No covers-{date}.jsonl in {config.META_DIR}. Run pipeline.metadata.fetch_new_covers first.")
    start = time.monotonic()
    n = build(
        config.META_DIR / f"releases-{date}.jsonl",
        config.META_DIR / f"artists-{date}.json",
        config.META_DIR / f"covers-{date}.jsonl",
        args.out,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"{n:,} releases from the {date} dump -> {args.out} ({size_mb:,.0f} MB) in {time.monotonic() - start:.0f} s")


if __name__ == "__main__":
    main()
