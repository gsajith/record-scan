// Keyboard shortcuts for the labeler page. The page decides `typing` (focus is in a text field)
// and `modifier` (Command, Control or Option held, so ⌘V still pastes).
import { TAGS, type Kind, type Tag } from "./labels";

export type HotkeyAction =
  | { type: "kind"; kind: Kind }
  | { type: "tag"; tag: Tag }
  | { type: "skip" }
  | { type: "save" }
  | { type: "search" };

/** One key per tag, in TAGS order: the number row, then - and =. */
export const TAG_KEYS = "1234567890-=";

/** One key per kind; the page shows these as hints. */
export const KIND_KEYS: Record<Kind, string> = { positive: "p", negative: "n", unknown: "u" };
const KIND_BY_KEY = Object.fromEntries(Object.entries(KIND_KEYS).map(([kind, key]) => [key, kind as Kind]));

export function hotkeyAction(e: { key: string; typing: boolean; modifier: boolean }): HotkeyAction | null {
  if (e.typing || e.modifier) return null;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (Object.hasOwn(KIND_BY_KEY, key)) return { type: "kind", kind: KIND_BY_KEY[key] };
  const tagIndex = TAG_KEYS.indexOf(key);
  if (key.length === 1 && tagIndex !== -1) return { type: "tag", tag: TAGS[tagIndex] };
  if (key === "s") return { type: "skip" };
  if (key === "Enter") return { type: "save" };
  if (key === "/") return { type: "search" };
  return null;
}
