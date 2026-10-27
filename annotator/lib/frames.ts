// What an inbox file name says about where a photo came from. pipeline/gold/sample_frames.py
// names frames <video-stem>-NN.jpg; a clip named <group>~<video> carries its source group.

// The sampler writes exactly two digits; "scan-1.jpg" dropped in by hand is its own photo
export function videoOf(frameName: string): string {
  return frameName.replace(/\.[^.]+$/, "").replace(/-\d{2}$/, "");
}

/**
 * The group a photo splits with (splitFor): the part before "~" in its clip name, such as
 * "yt-UC123" in "yt-UC123~abc-s01-01.jpg", else the video itself, so frames of one video never
 * land in both halves even when nobody named the clip.
 */
export function groupOfFrame(frameName: string): string {
  const video = videoOf(frameName);
  const cut = video.indexOf("~");
  return (cut > 0 ? video.slice(0, cut) : video).trim() || video;
}
