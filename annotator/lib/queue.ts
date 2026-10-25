// What to label next: photos without a label line, then frames waiting in the inbox.
import { appendFileSync, mkdirSync, readdirSync, readFileSync, renameSync, statSync } from "node:fs";
import { join } from "node:path";
import { SHA1_RE } from "./labels";
import { ingestImage, NotAnImageError, photoFile, readLabels } from "./store";

// Frames from pipeline/gold/sample_frames.py are JPEGs; dropped-in photos may be anything here
const IMAGE_FILE = /\.(jpe?g|png|heic|heif)$/i;

/**
 * video: the recording an inbox frame came from ("abbey road" for "abbey road-03.jpg", the
 * sampler's <video-stem>-NN.jpg naming), so the page can tell when the record changes.
 * Null for a photo that was pasted or dropped in.
 */
export type QueueItem = { sha1: string; file: string; video: string | null };

function videoOf(frameName: string): string {
  return frameName.replace(/\.[^.]+$/, "").replace(/-\d+$/, "");
}

// Which inbox frame each photo came from, one { sha1, frame } line per frame taken, so any
// later request (another tab, a page reload) still knows a photo's video
const SOURCES = join("inbox", "done", "sources.jsonl");

function readSources(setDir: string): Map<string, string> {
  try {
    const lines = readFileSync(join(setDir, SOURCES), "utf8").split("\n").filter(Boolean);
    return new Map(lines.map((l) => JSON.parse(l) as { sha1: string; frame: string }).map((s) => [s.sha1, s.frame]));
  } catch (e) {
    if (isMissing(e)) return new Map();
    throw e;
  }
}

function isMissing(e: unknown): boolean {
  return (e as NodeJS.ErrnoException).code === "ENOENT";
}

/** Unlabeled photos, oldest first, so frames keep the order they came in. */
function pendingPhotos(setDir: string, labeled: Map<string, unknown>): string[] {
  let names: string[];
  try {
    names = readdirSync(join(setDir, "photos"));
  } catch {
    return [];
  }
  return names
    .map((n) => n.slice(0, 40))
    .filter((sha1, i) => SHA1_RE.test(sha1) && names[i] === `${sha1}.jpg` && !labeled.has(sha1))
    .map((sha1) => ({ sha1, mtime: statSync(join(setDir, photoFile(sha1))).mtimeMs }))
    .sort((a, b) => a.mtime - b.mtime || a.sha1.localeCompare(b.sha1))
    .map((p) => p.sha1);
}

/** Image files waiting at the top of <set>/inbox/, in name order so a video's frames stay together. */
function inboxImages(setDir: string): string[] {
  try {
    return readdirSync(join(setDir, "inbox"), { withFileTypes: true })
      .filter((e) => e.isFile() && IMAGE_FILE.test(e.name))
      .map((e) => e.name)
      .sort();
  } catch {
    return [];
  }
}

/** Ingest one inbox image into photos/, then move it to inbox/done/ (or inbox/unreadable/). */
async function takeFromInbox(setDir: string, name: string): Promise<void> {
  const inbox = join(setDir, "inbox");
  let sha1: string | null = null;
  let dest = "done";
  try {
    sha1 = (await ingestImage(setDir, readFileSync(join(inbox, name)))).sha1;
  } catch (e) {
    if (isMissing(e)) return; // another request already moved it
    if (!(e instanceof NotAnImageError)) throw e;
    dest = "unreadable"; // move it aside so one bad file can't block the queue
  }
  mkdirSync(join(inbox, dest), { recursive: true });
  if (sha1) appendFileSync(join(setDir, SOURCES), JSON.stringify({ sha1, frame: name }) + "\n");
  try {
    renameSync(join(inbox, name), join(inbox, dest, name));
  } catch (e) {
    if (!isMissing(e)) throw e; // another request moved it while we ingested; the photo is in
  }
}

/**
 * The next photo to label and how many are left, counting it. Photos already in photos/ come
 * first; when none are waiting, the next inbox frame is moved in. Photos in `skip` (passed over
 * this session) are not handed out again but still count as left.
 */
export async function nextInQueue(
  setDir: string,
  skip: string[] = [],
): Promise<{ next: QueueItem | null; remaining: number }> {
  const skipped = new Set(skip);
  const labeled = readLabels(setDir);
  for (;;) {
    const pending = pendingPhotos(setDir, labeled);
    const inbox = inboxImages(setDir);
    const remaining = pending.length + inbox.length;
    const sha1 = pending.find((p) => !skipped.has(p));
    if (sha1) {
      const frame = readSources(setDir).get(sha1);
      return { next: { sha1, file: photoFile(sha1), video: frame ? videoOf(frame) : null }, remaining };
    }
    if (inbox.length === 0) return { next: null, remaining };
    await takeFromInbox(setDir, inbox[0]);
  }
}
