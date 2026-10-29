import { describe, expect, it } from "vitest";
import { inheritsAnswer, tagsForNextFrame } from "./carry";

describe("tagsForNextFrame", () => {
  it("keeps tags about the whole video and drops tags about one frame", () => {
    expect(tagsForNextFrame(["crate", "blur", "glare", "rack", "hand", "shrink-wrap"])).toEqual(["crate", "rack"]);
  });

  it("starts empty when nothing describes the video", () => {
    expect(tagsForNextFrame(["tilt", "partial"])).toEqual([]);
  });
});

describe("inheritsAnswer", () => {
  const before = { video: "IMG_1-", kind: "positive" as const, mbid: "old" };

  it("is true for a later frame of the same clip that still has the old answer", () => {
    expect(inheritsAnswer({ video: "IMG_1-", kind: "positive", mbid: "old" }, before)).toBe(true);
  });

  it("is false for another clip, another answer, or a pasted photo", () => {
    expect(inheritsAnswer({ video: "IMG_2-", kind: "positive", mbid: "old" }, before)).toBe(false);
    expect(inheritsAnswer({ video: "IMG_1-", kind: "positive", mbid: "other" }, before)).toBe(false);
    expect(inheritsAnswer({ video: "IMG_1-", kind: "negative", mbid: null }, before)).toBe(false);
    expect(inheritsAnswer({ video: null, kind: "positive", mbid: "old" }, { ...before, video: null })).toBe(false);
  });
});
