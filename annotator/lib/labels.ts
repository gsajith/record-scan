// Label records for the gold and wild photo sets. Mirrors pipeline/gold/labels.py, which the
// eval reads: keep the lists, the split rule and the validation rules identical in both.
// This side also checks that sha1 is 40 hex characters and source_url is a URL, which the
// Python loader does not; being stricter here only means it never writes a line Python rejects.
import { z } from "zod";

export const SETS = ["gold", "wild"] as const;
export const KINDS = ["positive", "negative", "unknown"] as const; // unknown: a record, but not in the catalog
// partial: any cut-off; crate: the bottom hidden by the sleeve in front, the crate-digging case
export const TAGS = ["tilt", "glare", "dark", "colored-light", "hand", "shrink-wrap", "text-only", "rack",
  "blur", "partial", "crate", "flash"] as const;
export const LICENSES = ["own", "cc0", "cc-by", "cc-by-sa", "pexels", "unsplash", "other-permitted"] as const;

export type SetName = (typeof SETS)[number];
export type Kind = (typeof KINDS)[number];
export type Tag = (typeof TAGS)[number];
export type License = (typeof LICENSES)[number];
export type Split = "eval" | "sim";

export const SHA1_RE = /^[0-9a-f]{40}$/;
// Lowercase only, like config.UUID_RE in Python
const MBID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

export const LabelSchema = z.strictObject({
  file: z.string(), // relative to the set folder
  sha1: z.string().regex(SHA1_RE),
  kind: z.enum(KINDS),
  mbid: z.string().regex(MBID_RE, "mbid must be a lowercase uuid").nullable(),
  tags: z.array(z.enum(TAGS)),
  source_url: z.url().nullable(),
  license: z.enum(LICENSES),
  split: z.enum(["eval", "sim"]),
  notes: z.string(),
  labeled_at: z.string(), // ISO 8601 UTC
});
export type Label = z.infer<typeof LabelSchema>;

/** Which half a photo belongs to, fixed by its hash. Gold photos are never used for tuning. */
export function splitFor(set: SetName, sha1: string): Split {
  if (set === "gold") return "eval";
  return parseInt(sha1.slice(0, 8), 16) % 2 === 0 ? "sim" : "eval";
}

/** Validate an untrusted label for a set with the same rules as the Python loader. */
export function checkLabel(set: SetName, input: unknown): { label: Label } | { error: string } {
  const parsed = LabelSchema.safeParse(input);
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    return { error: `${issue.path.join(".") || "label"}: ${issue.message}` };
  }
  const label = parsed.data;
  if (label.split !== splitFor(set, label.sha1)) {
    return { error: `split ${label.split} does not match the hash rule for ${label.sha1}` };
  }
  if (label.kind === "positive" && label.mbid === null) return { error: "a positive label needs an mbid" };
  if (label.kind !== "positive" && label.mbid !== null) {
    return { error: `a ${label.kind} label must not carry an mbid` };
  }
  if (set === "wild" && !label.source_url) return { error: "a wild photo needs a source_url (and a license)" };
  return { label };
}
