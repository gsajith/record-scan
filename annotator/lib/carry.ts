// Which tags survive from one frame of a video to the next. The album and kind carry over too
// (the page handles those); everything resets for a new video or a pasted photo.
import type { Tag } from "./labels";

/** Tags about the whole video: a flipping video is in a crate or on a rack on every frame. */
const VIDEO_TAGS: readonly Tag[] = ["crate", "rack"];

/** Tags for the next frame of the same video. Tags like blur or glare describe one frame and reset. */
export function tagsForNextFrame(tags: Tag[]): Tag[] {
  return tags.filter((t) => VIDEO_TAGS.includes(t));
}
