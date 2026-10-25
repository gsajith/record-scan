import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import sharp from "sharp";
import { describe, expect, it } from "vitest";
import { nextInQueue } from "./queue";
import { appendLabel, ingestImage } from "./store";

function tempSetDir(): string {
  return join(mkdtempSync(join(tmpdir(), "queue-")), "gold");
}

// A small solid-color JPEG; each color gives a different file and sha1
function jpeg(color: string): Promise<Buffer> {
  return sharp({ create: { width: 64, height: 64, channels: 3, background: color } }).jpeg().toBuffer();
}

function labelAsNegative(setDir: string, sha1: string) {
  appendLabel(setDir, {
    file: `photos/${sha1}.jpg`, sha1, kind: "negative", mbid: null, tags: [], source_url: null,
    license: "own", split: "eval", notes: "", labeled_at: "2026-10-25T04:00:00Z",
  });
}

describe("nextInQueue", () => {
  it("hands out an unlabeled photo, skips labeled ones, and counts what is left", async () => {
    const setDir = tempSetDir();
    const red = await ingestImage(setDir, await jpeg("#f00"));
    const green = await ingestImage(setDir, await jpeg("#0f0"));
    labelAsNegative(setDir, red.sha1);

    expect(await nextInQueue(setDir)).toEqual({ next: green, remaining: 1 });
  });

  it("moves inbox frames into photos in name order, after photos already waiting", async () => {
    const setDir = tempSetDir();
    const waiting = await ingestImage(setDir, await jpeg("#00f"));
    const inbox = join(setDir, "inbox");
    mkdirSync(inbox);
    const frame1 = await jpeg("#ff0");
    writeFileSync(join(inbox, "record-02.jpg"), await jpeg("#0ff"));
    writeFileSync(join(inbox, "record-01.jpg"), frame1);
    writeFileSync(join(inbox, "record.mov"), "a video the frame sampler has not run on yet");

    expect(await nextInQueue(setDir)).toEqual({ next: waiting, remaining: 3 });

    labelAsNegative(setDir, waiting.sha1);
    const sha1 = createHash("sha1").update(frame1).digest("hex");
    expect(await nextInQueue(setDir)).toEqual({ next: { sha1, file: `photos/${sha1}.jpg` }, remaining: 2 });
    expect(existsSync(join(setDir, "photos", `${sha1}.jpg`))).toBe(true);
    expect(readdirSync(join(inbox, "done"))).toEqual(["record-01.jpg"]);
  });

  it("passes over photos skipped this session but still counts them", async () => {
    const setDir = tempSetDir();
    const red = await ingestImage(setDir, await jpeg("#f00"));
    const green = await ingestImage(setDir, await jpeg("#0f0"));

    expect(await nextInQueue(setDir, [red.sha1])).toEqual({ next: green, remaining: 2 });
    expect(await nextInQueue(setDir, [red.sha1, green.sha1])).toEqual({ next: null, remaining: 2 });
  });
});
