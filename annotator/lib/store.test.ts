import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import sharp from "sharp";
import { describe, expect, it } from "vitest";
import type { Label } from "./labels";
import { appendLabel, countLabels, ingestImage, NotAnImageError } from "./store";

function tempSetDir(set: "gold" | "wild"): string {
  return join(mkdtempSync(join(tmpdir(), "labeler-")), set);
}

// A 3000x2000 PNG, larger than the 1600 px cap
function bigPng(): Promise<Buffer> {
  return sharp({ create: { width: 3000, height: 2000, channels: 3, background: "#c33" } }).png().toBuffer();
}

describe("ingestImage", () => {
  it("saves a JPEG at most 1600 px on its long side, named by the sha1 of the uploaded bytes", async () => {
    const setDir = tempSetDir("gold");
    const png = await bigPng();
    const { sha1, file } = await ingestImage(setDir, png);

    const saved = readFileSync(join(setDir, file));
    expect(sha1).toBe(createHash("sha1").update(png).digest("hex"));
    expect(file).toBe(`photos/${sha1}.jpg`);
    expect(await sharp(saved).metadata()).toMatchObject({ format: "jpeg", width: 1600, height: 1067 });
  });

  // iPhone photos are HEIC, which sharp's prebuilt binaries cannot decode; macOS can
  it.skipIf(process.platform !== "darwin")("saves an iPhone HEIC photo as a JPEG", async () => {
    const dir = mkdtempSync(join(tmpdir(), "heic-"));
    writeFileSync(join(dir, "in.png"), await bigPng());
    execFileSync("sips", ["-s", "format", "heic", join(dir, "in.png"), "--out", join(dir, "in.heic")]);
    const heic = readFileSync(join(dir, "in.heic"));

    const setDir = tempSetDir("gold");
    const { sha1, file } = await ingestImage(setDir, heic);

    expect(sha1).toBe(createHash("sha1").update(heic).digest("hex"));
    expect(await sharp(readFileSync(join(setDir, file))).metadata()).toMatchObject({ format: "jpeg", width: 1600 });
  });

  it("refuses a file that is not an image and saves nothing", async () => {
    const setDir = tempSetDir("gold");
    await expect(ingestImage(setDir, Buffer.from("not a photo"))).rejects.toBeInstanceOf(NotAnImageError);
    expect(existsSync(join(setDir, "photos"))).toBe(false);
  });

  it("gives the same record when the same file is added twice", async () => {
    const setDir = tempSetDir("gold");
    const png = await bigPng();
    const first = await ingestImage(setDir, png);
    expect(await ingestImage(setDir, png)).toEqual(first);
    expect(readdirSync(join(setDir, "photos"))).toEqual([`${first.sha1}.jpg`]);
  });
});

describe("appendLabel", () => {
  const sha1 = "86f7e437faa5a7fce15d1ddcb9eaeaea377667b8";
  const label: Label = {
    file: `photos/${sha1}.jpg`, sha1, kind: "negative", mbid: null, tags: [], source_url: null,
    license: "own", split: "eval", notes: "", labeled_at: "2026-10-25T04:00:00Z",
  };

  it("appends a re-label as a newer line, so the latest line wins", () => {
    const setDir = tempSetDir("gold");
    appendLabel(setDir, label);
    appendLabel(setDir, { ...label, tags: ["glare"], labeled_at: "2026-10-25T05:00:00Z" });

    const lines = readFileSync(join(setDir, "labels.jsonl"), "utf8").trimEnd().split("\n");
    expect(lines.map((l) => JSON.parse(l))).toEqual([label, { ...label, tags: ["glare"], labeled_at: "2026-10-25T05:00:00Z" }]);
  });
});

describe("countLabels", () => {
  // From SPLIT_VECTORS: the first lands in eval, the second in sim
  const evalSha1 = "86f7e437faa5a7fce15d1ddcb9eaeaea377667b8";
  const simSha1 = "e9d71f5ee7c92d6dc9e92ffdad17b8bd49418f98";
  const wild = (sha1: string, split: "eval" | "sim", tags: Label["tags"] = []): Label => ({
    file: `photos/${sha1}.jpg`, sha1, kind: "negative", mbid: null, tags, license: "cc0",
    source_url: "https://example.com/photo", split, notes: "", labeled_at: "2026-10-25T04:00:00Z",
  });

  it("counts labeled photos per split, a re-labeled photo once", () => {
    const setDir = tempSetDir("wild");
    appendLabel(setDir, wild(evalSha1, "eval"));
    appendLabel(setDir, wild(simSha1, "sim"));
    appendLabel(setDir, wild(simSha1, "sim", ["glare"]));
    expect(countLabels(setDir)).toEqual({ eval: 1, sim: 1 });
  });

  it("stops at a line the Python loader would reject, naming the line", () => {
    const setDir = tempSetDir("wild");
    appendLabel(setDir, wild(evalSha1, "eval"));
    writeFileSync(join(setDir, "labels.jsonl"), JSON.stringify({ ...wild(simSha1, "sim"), tags: ["sparkles"] }) + "\n", {
      flag: "a",
    });
    expect(() => countLabels(setDir)).toThrow(/line 2.*tags/);
  });

  it("counts nothing for a set with no labels yet", () => {
    expect(countLabels(tempSetDir("gold"))).toEqual({ eval: 0, sim: 0 });
  });
});
