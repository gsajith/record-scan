import { createHash } from "node:crypto";
import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import sharp from "sharp";
import { describe, expect, it } from "vitest";
import type { Label } from "./labels";
import { appendLabel, ingestImage } from "./store";

function tempSetDir(set: "gold" | "wild"): string {
  return join(mkdtempSync(join(tmpdir(), "labeler-")), set);
}

// A 3000x2000 PNG, larger than the 1600 px cap
function bigPng(): Promise<Buffer> {
  return sharp({ create: { width: 3000, height: 2000, channels: 3, background: "#c33" } }).png().toBuffer();
}

describe("ingestImage", () => {
  it("saves a JPEG at most 1600 px on its long side, named by the sha1 of its bytes", async () => {
    const setDir = tempSetDir("gold");
    const { sha1, file } = await ingestImage(setDir, await bigPng());

    const saved = readFileSync(join(setDir, file));
    expect(file).toBe(`photos/${sha1}.jpg`);
    expect(createHash("sha1").update(saved).digest("hex")).toBe(sha1);
    expect(await sharp(saved).metadata()).toMatchObject({ format: "jpeg", width: 1600, height: 1067 });
  });

  it("gives the same record when the same photo is pasted twice", async () => {
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
