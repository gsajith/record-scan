// Photos and labels on disk, in the layout pipeline/gold/labels.py reads:
// <set>/photos/<sha1>.jpg and one JSON line per label in <set>/labels.jsonl.
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { closeSync, existsSync, fsyncSync, mkdirSync, openSync, writeFileSync, writeSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import sharp from "sharp";
import type { Label } from "./labels";

const MAX_SIDE = 1600;

/**
 * Save an image into <setDir>/photos/ as an upright JPEG, at most 1600 px on its long side
 * (sharp also drops EXIF, so no GPS). The name is the sha1 of the uploaded bytes, not the
 * re-encoded ones, so a photo keeps its name and its split across sharp upgrades. An image
 * already saved is left alone, so pasting the same photo twice gives the same record.
 */
export async function ingestImage(setDir: string, input: Buffer): Promise<{ sha1: string; file: string }> {
  const sha1 = createHash("sha1").update(input).digest("hex");
  const file = `photos/${sha1}.jpg`;
  const path = join(setDir, file);
  if (!existsSync(path)) {
    let jpeg: Buffer;
    try {
      jpeg = await toJpeg(input);
    } catch (e) {
      // iPhone HEIC uses HEVC, which sharp's prebuilt binaries leave out for patent reasons.
      // The labeler only runs on the Mac (D8), so let macOS decode it.
      if (process.platform !== "darwin") throw e;
      jpeg = await toJpeg(await convertWithSips(input));
    }
    mkdirSync(join(setDir, "photos"), { recursive: true });
    writeFileSync(path, jpeg);
  }
  return { sha1, file };
}

function toJpeg(input: Buffer): Promise<Buffer> {
  return sharp(input)
    .rotate()
    .resize({ width: MAX_SIDE, height: MAX_SIDE, fit: "inside", withoutEnlargement: true })
    .jpeg({ quality: 90 })
    .toBuffer();
}

/** Convert any image macOS can read (HEIC included) to JPEG with the built-in sips tool. */
async function convertWithSips(input: Buffer): Promise<Buffer> {
  const dir = await mkdtemp(join(tmpdir(), "labeler-sips-"));
  try {
    await writeFile(join(dir, "in"), input);
    await promisify(execFile)("sips", ["-s", "format", "jpeg", join(dir, "in"), "--out", join(dir, "out.jpg")]);
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
