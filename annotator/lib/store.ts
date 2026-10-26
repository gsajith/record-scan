// Photos and labels on disk, in the layout pipeline/gold/labels.py reads:
// <set>/photos/<sha1>.jpg and one JSON line per label in <set>/labels.jsonl.
import { execFile } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import {
  appendFileSync,
  closeSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
  writeSync,
} from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { promisify } from "node:util";
import sharp from "sharp";
import { fingerprint, isNearDuplicate } from "./duplicates";
import { checkLabel, SETS, type Label, type Split } from "./labels";

const MAX_SIDE = 1600;

export function isMissing(e: unknown): boolean {
  return (e as NodeJS.ErrnoException).code === "ENOENT";
}

/** Where a photo lives inside its set folder. */
export function photoFile(sha1: string): string {
  return `photos/${sha1}.jpg`;
}

/** The upload is not an image the labeler can decode. Other errors (disk, permissions) pass through. */
export class NotAnImageError extends Error {}

/**
 * Save an image into <setDir>/photos/ as an upright JPEG, at most 1600 px on its long side
 * (sharp also drops EXIF, so no GPS). The name is the sha1 of the uploaded bytes, not the
 * re-encoded ones, so a photo keeps its name and its split across sharp upgrades. An image
 * already saved is left alone, so adding the same file twice gives the same record. Only
 * identical bytes match: a pasted copy (which the browser re-encodes) and the original file
 * get different names.
 */
export async function ingestImage(setDir: string, input: Buffer): Promise<{ sha1: string; file: string }> {
  const sha1 = createHash("sha1").update(input).digest("hex");
  if (!existsSync(join(setDir, photoFile(sha1)))) writePhoto(setDir, sha1, await decodeToJpeg(input));
  return { sha1, file: photoFile(sha1) };
}

function writePhoto(setDir: string, sha1: string, jpeg: Buffer): void {
  const path = join(setDir, photoFile(sha1));
  mkdirSync(join(setDir, "photos"), { recursive: true });
  // Write then rename, so a crash never leaves a half-written photo under its final name
  const partial = `${path}.${randomUUID()}.partial`;
  try {
    writeFileSync(partial, jpeg);
    renameSync(partial, path);
  } catch (e) {
    rmSync(partial, { force: true }); // e.g. a full disk; don't leave the fragment behind
    throw e;
  }
}

/** A wild photo that looks like one already in the set; `match` is that photo's sha1. */
export class NearDuplicateError extends Error {
  constructor(readonly match: string) {
    super(`This looks like a photo already in the wild set (${match.slice(0, 8)}).`);
  }
}

/**
 * Save a wild photo, with the address it came from when the browser gave one. Refuses a
 * near-duplicate of a wild photo already added this way: a pasted copy and the downloaded file
 * of one photo hash differently, and could otherwise land in different splits.
 */
export async function ingestWild(
  setDir: string,
  input: Buffer,
  sourceUrl: string | null,
): Promise<{ sha1: string; file: string }> {
  const sha1 = createHash("sha1").update(input).digest("hex");
  if (existsSync(join(setDir, photoFile(sha1)))) return { sha1, file: photoFile(sha1) };
  const jpeg = await decodeToJpeg(input);
  const phash = await fingerprint(jpeg);
  for (const other of readSources(setDir).values()) {
    if (other.phash && isNearDuplicate(phash, other.phash)) throw new NearDuplicateError(other.sha1);
  }
  writePhoto(setDir, sha1, jpeg);
  appendSource(setDir, { sha1, phash, ...(sourceUrl ? { source_url: sourceUrl } : {}) });
  return { sha1, file: photoFile(sha1) };
}

/**
 * Where each photo came from, one JSON line per fact in <set>/sources.jsonl: the inbox frame it
 * was taken from, or a wild photo's address and fingerprint.
 */
export type Source = { sha1: string; frame?: string; source_url?: string; phash?: string };

export function appendSource(setDir: string, source: Source): void {
  mkdirSync(setDir, { recursive: true });
  appendFileSync(join(setDir, "sources.jsonl"), toJsonLine(source));
}

/** Sources by sha1. Only hints for the page and the duplicate check, so a torn line is skipped. */
export function readSources(setDir: string): Map<string, Source> {
  let text: string;
  try {
    text = readFileSync(join(setDir, "sources.jsonl"), "utf8");
  } catch (e) {
    if (isMissing(e)) return new Map();
    throw e;
  }
  const sources = new Map<string, Source>();
  for (const line of text.split("\n")) {
    try {
      const source = JSON.parse(line) as Source;
      if (typeof source.sha1 === "string") sources.set(source.sha1, { ...sources.get(source.sha1), ...source });
    } catch {
      // skip a torn line
    }
  }
  return sources;
}

async function decodeToJpeg(input: Buffer): Promise<Buffer> {
  try {
    return await toJpeg(input);
  } catch {
    // iPhone HEIC uses HEVC, which sharp's prebuilt binaries leave out for patent reasons.
    // The labeler only runs on the Mac (D8), so let macOS decode those, and only those.
    if (process.platform === "darwin" && (await isHevcHeif(input))) {
      try {
        return await toJpeg(await convertWithSips(input));
      } catch {
        // fall through to the same error as any other unreadable file
      }
    }
    throw new NotAnImageError("not an image the labeler can read");
  }
}

async function isHevcHeif(input: Buffer): Promise<boolean> {
  try {
    // sharp reads the HEIF header even when it cannot decode the pixels
    const { format, compression } = await sharp(input).metadata();
    return format === "heif" && compression === "hevc";
  } catch {
    return false;
  }
}

function toJpeg(input: Buffer): Promise<Buffer> {
  return sharp(input)
    .rotate()
    .resize({ width: MAX_SIDE, height: MAX_SIDE, fit: "inside", withoutEnlargement: true })
    .jpeg({ quality: 90 })
    .toBuffer();
}

/** Convert a HEIC photo to JPEG with macOS's built-in sips tool. */
async function convertWithSips(input: Buffer): Promise<Buffer> {
  const dir = await mkdtemp(join(tmpdir(), "labeler-sips-"));
  try {
    await writeFile(join(dir, "in"), input);
    await promisify(execFile)("/usr/bin/sips", ["-s", "format", "jpeg", join(dir, "in"), "--out", join(dir, "out.jpg")], {
      timeout: 30_000,
    });
    return await readFile(join(dir, "out.jpg"));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

/**
 * One JSON line. Python's splitlines() also breaks on U+0085, U+2028 and U+2029, which
 * JSON.stringify leaves raw, so a note containing one is escaped to keep the line whole.
 */
function toJsonLine(value: unknown): string {
  const json = JSON.stringify(value).replace(/[\u0085\u2028\u2029]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);
  return json + "\n";
}

/** Append one already-checked label. A re-label is a new line; readers keep the latest per sha1. */
export function appendLabel(setDir: string, label: Label): void {
  mkdirSync(setDir, { recursive: true });
  const fd = openSync(join(setDir, "labels.jsonl"), "a");
  try {
    writeSync(fd, toJsonLine(label));
    fsyncSync(fd); // a label is a person's minute of work; don't lose it to a crash
  } finally {
    closeSync(fd);
  }
}

/**
 * The latest label per sha1, like load_labels in Python, with the same checks: a line Python
 * would reject stops here too, naming the line.
 */
export function readLabels(setDir: string): Map<string, Label> {
  // The sealed holdout reuses the same layout under another root, so the folder name decides
  const set = SETS.find((s) => s === basename(setDir));
  if (!set) throw new Error(`${setDir} is not a gold or wild set folder`);
  const path = join(setDir, "labels.jsonl");
  const latest = new Map<string, Label>();
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (e) {
    if (isMissing(e)) return latest; // no labels yet
    throw e;
  }
  text.split("\n").forEach((line, i) => {
    if (!line.trim()) return;
    let record: unknown;
    try {
      record = JSON.parse(line);
    } catch {
      throw new Error(`${path} line ${i + 1} is not valid JSON`);
    }
    const checked = checkLabel(set, record);
    if ("error" in checked) throw new Error(`${path} line ${i + 1}: ${checked.error}`);
    latest.set(checked.label.sha1, checked.label);
  });
  return latest;
}

export function countLabels(setDir: string): Record<Split, number> {
  const counts = { eval: 0, sim: 0 };
  for (const label of readLabels(setDir).values()) counts[label.split] += 1;
  return counts;
}
