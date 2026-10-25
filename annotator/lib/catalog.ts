// Read-only search over catalog.sqlite, built by `python -m pipeline.metadata.build_catalog_db`.
// Mirrors pipeline/metadata/catalog.py.

import { existsSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";

const APOSTROPHES = /['’ʼ`]/g;
const TRAILING_THE = /,\s*the$/;

/**
 * Matching form of a title or artist name: "Beatles, The" and "The Beatles" both give "beatles".
 * Apostrophes are dropped ("don't" -> "dont"); other punctuation and symbols split words
 * ("AC/DC" -> "ac dc"), as the FTS5 tokenizer does. toLowerCase stands in for Python's
 * casefold, which differs only for a few letters such as ß.
 */
export function normalize(s: string): string {
  s = s.toLowerCase().normalize("NFKD").replace(/\p{M}/gu, "").trim();
  s = s.replace(TRAILING_THE, "");
  s = s.replace(APOSTROPHES, "").replace(/[\p{P}\p{S}]/gu, " ");
  s = s.split(/\s+/).filter(Boolean).join(" ");
  return s.startsWith("the ") ? s.slice(4) : s;
}

/** Quote every normalized token so user text cannot carry FTS5 syntax; prefix-match the last. */
function ftsQuery(query: string): string | null {
  const tokens = normalize(query).split(" ").filter(Boolean);
  if (tokens.length === 0) return null;
  return tokens.map((t) => `"${t}"`).join(" ") + "*";
}

export type Release = { mbid: string; title: string; artists: string[]; format: string | null };

type ReleaseRow = { mbid: string; title: string; artist_ids_json: string; format: string | null };

export function openCatalog(path: string) {
  if (!existsSync(path)) {
    throw new Error(`Catalog not found at ${path}. Build it with: python -m pipeline.metadata.build_catalog_db`);
  }
  // Read-only so the labeler and the eval can share the file
  const db = new DatabaseSync(path, { readOnly: true });
  const searchRows = db.prepare(
    `SELECT r.mbid, r.title, r.artist_ids_json, r.format
     FROM release_fts JOIN releases r ON r.rowid = release_fts.rowid
     WHERE release_fts MATCH ? ORDER BY bm25(release_fts) LIMIT ?`,
  );

  function artistNames(ids: string[]): Map<string, string> {
    if (ids.length === 0) return new Map();
    const rows = db
      .prepare(`SELECT id, name FROM artists WHERE id IN (${ids.map(() => "?").join(",")})`)
      .all(...ids) as { id: string; name: string }[];
    return new Map(rows.map((r) => [r.id, r.name]));
  }

  return {
    search(query: string, limit = 20): Release[] {
      const match = ftsQuery(query);
      if (match === null) return [];
      const rows = searchRows.all(match, limit) as ReleaseRow[];
      const idsPerRow = rows.map((r) => JSON.parse(r.artist_ids_json) as string[]);
      const names = artistNames([...new Set(idsPerRow.flat())]);
      // Resolved names in artist_ids order, unknown ids skipped
      return rows.map((r, i) => ({
        mbid: r.mbid,
        title: r.title,
        artists: idsPerRow[i].flatMap((id) => names.get(id) ?? []),
        format: r.format,
      }));
    },
  };
}
