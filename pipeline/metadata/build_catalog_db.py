"""Build catalog.sqlite from the slim dump JSON plus any delta JSONL (Task 6's catalog updates).

    python -m pipeline.metadata.build_catalog_db [--out PATH] [--delta DIR]
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
CREATE VIRTUAL TABLE release_fts USING fts5(
    mbid UNINDEXED, title, artists, tokenize="unicode61 remove_diacritics 2"
);
"""


def _jsonl(folder: Path):
    for path in sorted(folder.glob("*.jsonl")):  # date-named, so later files win
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def build(releases_json: Path, artists_json: Path, out: Path, delta_dir: Path | None) -> int:
    """Write the catalog to out + ".tmp", then move it over out. Returns the release count."""
    artists: dict[str, dict] = json.loads(artists_json.read_text())
    # Keyed by mbid so a delta row replaces the dump row it updates. The delta fetcher writes
    # whole rows, so a field missing from a delta row is missing upstream too.
    releases = {r["id"]: r for r in json.loads(releases_json.read_text())}
    if delta_dir is not None:
        if (delta_dir / "artists").is_dir():
            artists.update({a["id"]: a for a in _jsonl(delta_dir / "artists")})
        if (delta_dir / "releases").is_dir():
            releases.update({r["id"]: r for r in _jsonl(delta_dir / "releases")})

    release_rows, fts_rows = [], []
    for rowid, r in enumerate(releases.values(), start=1):
        ids = r.get("artist_ids") or []
        names = [artists[a]["name"] for a in ids if a in artists]
        primary = artists[ids[0]]["name"] if ids and ids[0] in artists else None
        cover = r.get("cover_id")
        release_rows.append((
            rowid, r["id"], r["title"], primary, json.dumps(ids), r.get("format"),
            None if cover is None else str(cover), r.get("release_group"),
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
            db.executemany("INSERT INTO release_fts (rowid, mbid, title, artists) VALUES (?, ?, ?, ?)", fts_rows)
            db.execute("INSERT INTO release_fts (release_fts) VALUES ('optimize')")
    finally:
        db.close()
    os.replace(tmp, out)
    return len(release_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=config.CATALOG_DB)
    parser.add_argument("--delta", type=Path, default=config.DELTA_DIR, help="skipped when absent")
    args = parser.parse_args()
    start = time.monotonic()
    n = build(
        config.META_DIR / "slim_releases_with_covers.json",
        config.META_DIR / "release_slim_artists.json",
        args.out,
        args.delta,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"{n:,} releases -> {args.out} ({size_mb:,.0f} MB) in {time.monotonic() - start:.0f} s")


if __name__ == "__main__":
    main()
