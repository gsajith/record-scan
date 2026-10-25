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

    expect(await nextInQueue(setDir)).toEqual({ next: { ...green, video: null }, remaining: 1 });
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

    expect(await nextInQueue(setDir)).toEqual({ next: { ...waiting, video: null }, remaining: 3 });

    labelAsNegative(setDir, waiting.sha1);
    const sha1 = createHash("sha1").update(frame1).digest("hex");
    expect(await nextInQueue(setDir)).toEqual({ next: { sha1, file: `photos/${sha1}.jpg`, video: "record" }, remaining: 2 });
    expect(existsSync(join(setDir, "photos", `${sha1}.jpg`))).toBe(true);
    expect(readdirSync(join(inbox, "done"))).toContain("record-01.jpg");
    expect(readdirSync(inbox)).toEqual(["done", "record-02.jpg", "record.mov"]);
  });

  it("passes over photos skipped this session but still counts them", async () => {
    const setDir = tempSetDir();
    const red = await ingestImage(setDir, await jpeg("#f00"));
    const green = await ingestImage(setDir, await jpeg("#0f0"));

    expect(await nextInQueue(setDir, [red.sha1])).toEqual({ next: { ...green, video: null }, remaining: 2 });
    expect(await nextInQueue(setDir, [red.sha1, green.sha1])).toEqual({ next: null, remaining: 2 });
  });

  it("tells which video an inbox frame came from, so the page knows when the record changes", async () => {
    const setDir = tempSetDir();
    mkdirSync(join(setDir, "inbox"), { recursive: true });
    writeFileSync(join(setDir, "inbox", "abbey road-01.jpg"), await jpeg("#f0f"));
    const pasted = await ingestImage(setDir, await jpeg("#0f0"));

    expect((await nextInQueue(setDir)).next).toEqual({ ...pasted, video: null });
    labelAsNegative(setDir, pasted.sha1);
    expect((await nextInQueue(setDir)).next?.video).toBe("abbey road");
  });

  it("still knows a frame's video in a later request, such as after a page reload", async () => {
    const setDir = tempSetDir();
    mkdirSync(join(setDir, "inbox"), { recursive: true });
    writeFileSync(join(setDir, "inbox", "abbey road-01.jpg"), await jpeg("#f0f"));

    const first = await nextInQueue(setDir);
    expect(await nextInQueue(setDir)).toEqual(first);
    expect(first.next?.video).toBe("abbey road");
  });

  it("copes with two requests taking the same inbox frame at once", async () => {
    const setDir = tempSetDir();
    mkdirSync(join(setDir, "inbox"), { recursive: true });
    writeFileSync(join(setDir, "inbox", "record-01.jpg"), await jpeg("#abc"));

    const [a, b] = await Promise.all([nextInQueue(setDir), nextInQueue(setDir)]);
    expect(b.next).toEqual(a.next);
    expect(a.next?.video).toBe("record");
    expect(readdirSync(join(setDir, "inbox"))).toEqual(["done"]);
  });

  it("moves an unreadable inbox file aside instead of getting stuck on it", async () => {
    const setDir = tempSetDir();
    mkdirSync(join(setDir, "inbox"), { recursive: true });
    writeFileSync(join(setDir, "inbox", "broken-01.jpg"), "not really a jpeg");
    writeFileSync(join(setDir, "inbox", "good-01.jpg"), await jpeg("#123"));

    expect((await nextInQueue(setDir)).next?.video).toBe("good");
    expect(readdirSync(join(setDir, "inbox", "unreadable"))).toEqual(["broken-01.jpg"]);
  });

  it("only treats the frame sampler's -NN suffix as a video frame", async () => {
    const setDir = tempSetDir();
    mkdirSync(join(setDir, "inbox"), { recursive: true });
    writeFileSync(join(setDir, "inbox", "scan-1.jpg"), await jpeg("#321"));
    expect((await nextInQueue(setDir)).next?.video).toBe("scan-1");
  });

  it("ignores a damaged line in the sources file", async () => {
    const setDir = tempSetDir();
    mkdirSync(join(setDir, "inbox", "done"), { recursive: true });
    writeFileSync(join(setDir, "inbox", "done", "sources.jsonl"), '{"sha1": "trunc\n');
    writeFileSync(join(setDir, "inbox", "abbey-01.jpg"), await jpeg("#456"));
    expect((await nextInQueue(setDir)).next?.video).toBe("abbey");
  });
});
