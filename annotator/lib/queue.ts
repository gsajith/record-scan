// What to label next: photos without a label line, then frames waiting in the inbox.
import { mkdirSync, readdirSync, readFileSync, renameSync, statSync } from "node:fs";
import { join } from "node:path";
import { SHA1_RE } from "./labels";
import { appendSource, ingestImage, isMissing, NotAnImageError, photoFile, readLabels, readSources } from "./store";

// Frames from pipeline/gold/sample_frames.py are JPEGs; dropped-in photos may be anything here
const IMAGE_FILE = /\.(jpe?g|png|heic|heif)$/i;

/**
 * video: the recording an inbox frame came from ("abbey road" for "abbey road-03.jpg", the
 * sampler's <video-stem>-NN.jpg naming), so the page can tell when the record changes.
 * Null for a photo that was pasted or dropped in. sourceUrl: the address a wild photo came
 * from, when the browser gave one, so labeling it later still records it.
 */
export type QueueItem = { sha1: string; file: string; video: string | null; sourceUrl: string | null };

// The sampler writes exactly two digits; "scan-1.jpg" dropped in by hand is its own photo
function videoOf(frameName: string): string {
  return frameName.replace(/\.[^.]+$/, "").replace(/-\d{2}$/, "");
}

/**
 * The source group in a clip name "<group>~<video>", such as "yt-UC123~abc-s01" for a YouTube
 * channel's video. Every frame of a group lands in the same half (splitFor).
 */
function groupOfFrame(frameName: string): string | null {
  const video = videoOf(frameName);
  const cut = video.indexOf("~");
  return cut > 0 ? video.slice(0, cut) : null;
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
  const group = groupOfFrame(name);
  if (sha1) appendSource(setDir, { sha1, frame: name, ...(group && { group }) });
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
      const { frame, source_url } = readSources(setDir).get(sha1) ?? {};
      const next = { sha1, file: photoFile(sha1), video: frame ? videoOf(frame) : null, sourceUrl: source_url ?? null };
      return { next, remaining };
    }
    if (inbox.length === 0) return { next: null, remaining };
    await takeFromInbox(setDir, inbox[0]);
  }
}
