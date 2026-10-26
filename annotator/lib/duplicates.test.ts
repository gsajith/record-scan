import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import sharp from "sharp";
import { describe, expect, it } from "vitest";
import { ingestWild, NearDuplicateError } from "./store";

const source = "https://www.flickr.com/photos/x/1";

function wildDir(): string {
  return join(mkdtempSync(join(tmpdir(), "dupes-")), "wild");
}

// A 400 px photo-like image: an 8x8 grid of gray blocks from a seeded generator
function patterned(seed: number): Promise<Buffer> {
  const pixels = Buffer.alloc(8 * 8);
  for (let i = 0; i < pixels.length; i++) {
    seed = (seed * 1103515245 + 12345) % 2 ** 31;
    pixels[i] = seed % 256;
  }
  return sharp(pixels, { raw: { width: 8, height: 8, channels: 1 } })
    .resize(400, 400, { kernel: "cubic" })
    .png()
    .toBuffer();
}

describe("ingestWild", () => {
  it("refuses a resized, re-saved copy of a wild photo already saved, naming the match", async () => {
    const setDir = wildDir();
    const original = await patterned(1);
    const first = await ingestWild(setDir, original, source);
    const copy = await sharp(original).resize(280).jpeg({ quality: 70 }).toBuffer();

    const error = await ingestWild(setDir, copy, source).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(NearDuplicateError);
    expect((error as NearDuplicateError).match).toBe(first.sha1);
  });

  it("accepts a different photo", async () => {
    const setDir = wildDir();
    await ingestWild(setDir, await patterned(1), source);
    await expect(ingestWild(setDir, await patterned(2), source)).resolves.toMatchObject({ file: expect.any(String) });
  });
});
