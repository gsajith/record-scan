import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { beforeAll, describe, expect, it } from "vitest";
import { normalize, openCatalog } from "./catalog";

describe("normalize", () => {
  // Copied from test_normalize in pipeline/tests/test_catalog.py
  it.each([
    ["The Beatles", "beatles"],
    ["Beatles, The", "beatles"],
    ["Björk", "bjork"],
    ["Abbey Road (Remastered)", "abbey road remastered"],
    ["  Don't   Stop  ", "dont stop"],
    ["AC/DC", "ac dc"],
  ])("turns %j into %j, like the Python catalog", (raw, expected) => {
    expect(normalize(raw)).toBe(expected);
  });
});

const BEATLES = "b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d";
const BJORK = "87c5dedd-371d-4a53-9f7f-80522fb7f3cb";
const ABBEY = "11111111-1111-4111-8111-111111111111";
const DEBUT = "22222222-2222-4222-8222-222222222222";

// The schema from pipeline/metadata/build_catalog_db.py, with two releases
function buildCatalog(): string {
  const path = join(mkdtempSync(join(tmpdir(), "catalog-")), "catalog.sqlite");
  const db = new DatabaseSync(path);
  db.exec(`
    CREATE TABLE artists (id TEXT PRIMARY KEY, name TEXT NOT NULL, aliases_json TEXT);
    CREATE TABLE releases (mbid TEXT PRIMARY KEY, title TEXT NOT NULL, primary_artist TEXT,
      artist_ids_json TEXT NOT NULL, format TEXT, cover_id TEXT, release_group TEXT, album_key TEXT NOT NULL);
    CREATE VIRTUAL TABLE release_fts USING fts5(mbid UNINDEXED, title, artists, tokenize="unicode61 remove_diacritics 2");
    INSERT INTO artists VALUES ('${BEATLES}', 'The Beatles', '[]'), ('${BJORK}', 'Björk', '[]');
    INSERT INTO releases (rowid, mbid, title, primary_artist, artist_ids_json, format, album_key) VALUES
      (1, '${ABBEY}', 'Abbey Road', '${BEATLES}', '["${BEATLES}"]', '12" Vinyl', 'abbey road|beatles'),
      (2, '${DEBUT}', 'Debut', '${BJORK}', '["${BJORK}"]', NULL, 'debut|bjork');
    INSERT INTO release_fts (rowid, mbid, title, artists) VALUES
      (1, '${ABBEY}', 'abbey road', 'beatles'), (2, '${DEBUT}', 'debut', 'bjork');
  `);
  db.close();
  return path;
}

describe("catalog search", () => {
  let catalogPath: string;
  beforeAll(() => {
    catalogPath = buildCatalog();
  });

  it("finds a release from a half-typed query, with its artist names", () => {
    expect(openCatalog(catalogPath).search("beatles abb")).toEqual([
      { mbid: ABBEY, title: "Abbey Road", artists: ["The Beatles"], format: '12" Vinyl' },
    ]);
  });

  it("treats search syntax in the query as plain words", () => {
    const catalog = openCatalog(catalogPath);
    expect(catalog.search('"debut OR')).toEqual([]);
    expect(catalog.search("Björk: Debut*").map((r) => r.mbid)).toEqual([DEBUT]);
  });
});
