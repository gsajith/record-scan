import { describe, expect, it } from "vitest";
import { checkLabel, splitFor } from "./labels";

// Copied from SPLIT_VECTORS in pipeline/tests/test_labels.py: both implementations must agree.
const SPLIT_VECTORS: [string, "eval" | "sim"][] = [
  ["86f7e437faa5a7fce15d1ddcb9eaeaea377667b8", "eval"],
  ["e9d71f5ee7c92d6dc9e92ffdad17b8bd49418f98", "sim"],
  ["84a516841ba77a5b4648de2cd0dfcb30ea46dbb4", "sim"],
  ["4a0a19218e082a343a1b17e5333409af9d98f0f5", "eval"],
  ["54fd1711209fb1c0781092374132c66e79e2241b", "eval"],
];

describe("splitFor", () => {
  it.each(SPLIT_VECTORS)("puts wild photo %s in %s, like the Python rule", (sha1, expected) => {
    expect(splitFor("wild", sha1)).toBe(expected);
  });
});

const MBID = "b84ee12a-09ef-421b-82de-0441a926375b";
const [WILD_SHA1] = SPLIT_VECTORS[0];

// A complete wild positive, the same shape as make_label in the Python tests
const wildLabel = {
  file: `photos/${WILD_SHA1}.jpg`,
  sha1: WILD_SHA1,
  kind: "positive",
  mbid: MBID,
  tags: ["glare"],
  source_url: "https://www.flickr.com/photos/x/1",
  license: "cc-by",
  split: "eval",
  notes: "",
  labeled_at: "2026-10-25T04:00:00Z",
};

describe("checkLabel", () => {
  // Every line carries a license field, as Python's loader checks the exact keys
  it("refuses a label missing its license field", () => {
    const noLicense: Partial<typeof wildLabel> = { ...wildLabel };
    delete noLicense.license;
    expect(checkLabel("wild", noLicense)).toMatchObject({ error: expect.stringContaining("license") });
  });

  it("accepts a gold label with license own", () => {
    const gold = { ...wildLabel, source_url: null, license: "own" };
    expect(checkLabel("gold", gold)).toEqual({ label: gold });
  });

  it("refuses a split that does not match the hash rule", () => {
    expect(checkLabel("wild", { ...wildLabel, split: "sim" })).toMatchObject({
      error: expect.stringContaining("split"),
    });
  });

  it("ties the mbid to the kind: a positive needs one, a negative must not have one", () => {
    expect(checkLabel("wild", { ...wildLabel, mbid: null })).toMatchObject({
      error: expect.stringContaining("mbid"),
    });
    expect(checkLabel("wild", { ...wildLabel, kind: "negative" })).toMatchObject({
      error: expect.stringContaining("mbid"),
    });
  });

  it("refuses an uppercase mbid, which the Python loader would reject", () => {
    expect(checkLabel("wild", { ...wildLabel, mbid: MBID.toUpperCase() })).toMatchObject({
      error: expect.stringContaining("mbid"),
    });
  });

  it("accepts a wild photo without a source, like the Python loader", () => {
    const noSource = { ...wildLabel, source_url: null, license: "other-permitted" };
    expect(checkLabel("wild", noSource)).toEqual({ label: noSource });
  });
});
