"""Read-only access to catalog.sqlite, built by `python -m pipeline.metadata.build_catalog_db`."""

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pipeline import config


@dataclass(frozen=True)
class Release:
    mbid: str
    title: str
    artists: tuple[str, ...]  # resolved names in artist_ids order, unknown ids skipped
    format: str | None
    cover_id: str | None
    release_group: str | None


_APOSTROPHES = "'’ʼ`"
_TRAILING_THE = re.compile(r",\s*the$")


def normalize(s: str) -> str:
    """Matching form of a title or artist name: "Beatles, The" and "The Beatles" both give "beatles".

    Apostrophes are dropped ("don't" -> "dont"); other punctuation and symbols split words
    ("AC/DC" -> "ac dc"), as the FTS5 tokenizer does.
    """
    s = unicodedata.normalize("NFKD", s.casefold())
    s = "".join(c for c in s if not unicodedata.combining(c)).strip()
    s = _TRAILING_THE.sub("", s)  # "beatles, the": moving "the" to the front only to drop it is a no-op
    s = "".join(
        "" if c in _APOSTROPHES else " " if unicodedata.category(c)[0] in "PS" else c for c in s
    )
    s = " ".join(s.split())
    return s.removeprefix("the ")


def _fts_query(query: str) -> str | None:
    """Quote every normalized token so user text cannot carry FTS5 syntax; prefix-match the last."""
    tokens = normalize(query).split()
    if not tokens:
        return None
    return " ".join(f'"{t}"' for t in tokens) + "*"


class Catalog:
    def __init__(self, db_path: Path | None = None):
        path = Path(db_path or config.CATALOG_DB)
        if not path.is_file():
            raise SystemExit(
                f"Catalog not found at {path}. Build it with: python -m pipeline.metadata.build_catalog_db"
            )
        # Read-only so the labeler and eval can share the file
        self._db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)

    def _releases(self, rows: list[tuple]) -> list[Release]:
        ids = {a for row in rows for a in json.loads(row[2])}
        names = dict(
            self._db.execute(
                f"SELECT id, name FROM artists WHERE id IN ({','.join('?' * len(ids))})", list(ids)
            )
        )
        return [
            Release(mbid, title, tuple(names[a] for a in json.loads(ids_json) if a in names), fmt, cover, rg)
            for mbid, title, ids_json, fmt, cover, rg in rows
        ]

    def release(self, mbid: str) -> Release | None:
        rows = self._db.execute(
            "SELECT mbid, title, artist_ids_json, format, cover_id, release_group FROM releases WHERE mbid = ?",
            (mbid,),
        ).fetchall()
        return self._releases(rows)[0] if rows else None

    def search(self, query: str, limit: int = 20) -> list[Release]:
        match = _fts_query(query)
        if match is None:
            return []
        rows = self._db.execute(
            """SELECT r.mbid, r.title, r.artist_ids_json, r.format, r.cover_id, r.release_group
               FROM release_fts JOIN releases r ON r.rowid = release_fts.rowid
               WHERE release_fts MATCH ? ORDER BY bm25(release_fts) LIMIT ?""",
            (match, limit),
        ).fetchall()
        return self._releases(rows)

    def album_key(self, mbid: str) -> str:
        """normalize(title) + "|" + normalize(primary artist). Raises KeyError for an unknown mbid."""
        row = self._db.execute("SELECT album_key FROM releases WHERE mbid = ?", (mbid,)).fetchone()
        if row is None:
            raise KeyError(mbid)
        return row[0]

    def find(self, artist: str, title: str) -> list[str]:
        """Mbids whose normalized title and primary artist equal the given ones."""
        key = f"{normalize(title)}|{normalize(artist)}"
        return [m for (m,) in self._db.execute("SELECT mbid FROM releases WHERE album_key = ?", (key,))]

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM releases").fetchone()[0]
