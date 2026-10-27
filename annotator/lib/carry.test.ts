import { describe, expect, it } from "vitest";
import { tagsForNextFrame } from "./carry";

describe("tagsForNextFrame", () => {
  it("keeps tags about the whole video and drops tags about one frame", () => {
    expect(tagsForNextFrame(["crate", "blur", "glare", "rack", "hand", "shrink-wrap"])).toEqual(["crate", "rack"]);
  });

  it("starts empty when nothing describes the video", () => {
    expect(tagsForNextFrame(["tilt", "partial"])).toEqual([]);
  });
});
