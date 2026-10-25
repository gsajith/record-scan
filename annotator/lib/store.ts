// Photos and labels on disk, in the layout pipeline/gold/labels.py reads:
// <set>/photos/<sha1>.jpg and one JSON line per label in <set>/labels.jsonl.
import { execFile } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { closeSync, existsSync, fsyncSync, mkdirSync, openSync, readFileSync, renameSync, rmSync, writeFileSync, writeSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import sharp from "sharp";
import type { Label, Split } from "./labels";

const MAX_SIDE = 1600;

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
  const file = `photos/${sha1}.jpg`;
  const path = join(setDir, file);
  if (!existsSync(path)) {
    const jpeg = await decodeToJpeg(input);
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
  return { sha1, file };
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

/** Append one already-checked label. A re-label is a new line; readers keep the latest per sha1. */
export function appendLabel(setDir: string, label: Label): void {
  mkdirSync(setDir, { recursive: true });
  const fd = openSync(join(setDir, "labels.jsonl"), "a");
  try {
    writeSync(fd, JSON.stringify(label) + "\n");
    fsyncSync(fd); // a label is a person's minute of work; don't lose it to a crash
  } finally {
    closeSync(fd);
  }
}

/** The latest label per sha1, like load_labels in Python. A broken line stops with its line number. */
export function readLabels(setDir: string): Map<string, Label> {
  const latest = new Map<string, Label>();
  let text: string;
  try {
    text = readFileSync(join(setDir, "labels.jsonl"), "utf8");
  } catch {
    return latest; // no labels yet
  }
  text.split("\n").forEach((line, i) => {
    if (!line.trim()) return;
    try {
      const label = JSON.parse(line) as Label;
      latest.set(label.sha1, label);
    } catch {
      throw new Error(`${join(setDir, "labels.jsonl")} line ${i + 1} is not valid JSON`);
    }
  });
  return latest;
}

export function countLabels(setDir: string): Record<Split, number> {
  const counts = { eval: 0, sim: 0 };
  for (const label of readLabels(setDir).values()) counts[label.split] += 1;
  return counts;
}
