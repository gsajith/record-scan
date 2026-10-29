// Which tags survive from one frame of a video to the next. The album and kind carry over too
// (the page handles those); everything resets for a new video or a pasted photo.
import type { Kind, Tag } from "./labels";

/** Tags about the whole video: a flipping video is in a crate or on a rack on every frame. */
const VIDEO_TAGS: readonly Tag[] = ["crate", "rack"];

/** Tags for the next frame of the same video. Tags like blur or glare describe one frame and reset. */
export function tagsForNextFrame(tags: Tag[]): Tag[] {
  return tags.filter((t) => VIDEO_TAGS.includes(t));
}

/** What a frame was labeled as, enough to tell whether two frames share an answer. */
export type Answer = { video: string | null; kind: Kind; mbid: string | null };

/**
 * A frame reopened by Undo still holds the answer it carried over from an earlier frame of the
 * same clip, so fixing that earlier frame should fix this one too. A pasted photo has no clip.
 */
export function inheritsAnswer(frame: Answer, before: Answer): boolean {
  return frame.video !== null && frame.video === before.video && frame.kind === before.kind && frame.mbid === before.mbid;
}
