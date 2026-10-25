// Photos and labels on disk, in the layout pipeline/gold/labels.py reads:
// <set>/photos/<sha1>.jpg and one JSON line per label in <set>/labels.jsonl.
import { createHash } from "node:crypto";
import { closeSync, existsSync, fsyncSync, mkdirSync, openSync, writeFileSync, writeSync } from "node:fs";
import { join } from "node:path";
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
    const jpeg = await sharp(input)
      .rotate()
      .resize({ width: MAX_SIDE, height: MAX_SIDE, fit: "inside", withoutEnlargement: true })
      .jpeg({ quality: 90 })
      .toBuffer();
    mkdirSync(join(setDir, "photos"), { recursive: true });
    writeFileSync(path, jpeg);
  }
  return { sha1, file };
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
