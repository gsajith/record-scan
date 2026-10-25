// What to label next: photos without a label line, then frames waiting in the inbox.
import { mkdirSync, readdirSync, readFileSync, renameSync, statSync } from "node:fs";
import { join } from "node:path";
import { ingestImage, NotAnImageError, readLabels } from "./store";

// Frames from pipeline/gold/sample_frames.py are JPEGs; dropped-in photos may be anything here
const IMAGE_FILE = /\.(jpe?g|png|heic|heif)$/i;

export type QueueItem = { sha1: string; file: string };

/** Unlabeled photos, oldest first, so frames keep the order they came in. */
function pendingPhotos(setDir: string, labeled: Map<string, unknown>): string[] {
  let names: string[];
  try {
    names = readdirSync(join(setDir, "photos"));
  } catch {
    return [];
  }
  return names
    .filter((n) => /^[0-9a-f]{40}\.jpg$/.test(n) && !labeled.has(n.slice(0, 40)))
    .map((n) => ({ sha1: n.slice(0, 40), mtime: statSync(join(setDir, "photos", n)).mtimeMs }))
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
  let dest = "done";
  try {
    await ingestImage(setDir, readFileSync(join(inbox, name)));
  } catch (e) {
    if (!(e instanceof NotAnImageError)) throw e;
    dest = "unreadable"; // move it aside so one bad file can't block the queue
  }
  mkdirSync(join(inbox, dest), { recursive: true });
  renameSync(join(inbox, name), join(inbox, dest, name));
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
    if (sha1) return { next: { sha1, file: `photos/${sha1}.jpg` }, remaining };
    if (inbox.length === 0) return { next: null, remaining };
    await takeFromInbox(setDir, inbox[0]);
  }
}
