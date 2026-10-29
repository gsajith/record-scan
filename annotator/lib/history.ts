// Frames saved since the page loaded, so Undo can reopen them. Saving a reopened frame adds a new
// label line, which wins over the old one (Review focus 1), so nothing on disk is rewritten.

/** `back` holds saves, oldest first. `ahead` holds reopened frames stepped past, nearest first. */
export type History<T> = { back: T[]; ahead: T[] };

export function emptyHistory<T>(): History<T> {
  return { back: [], ahead: [] };
}

/**
 * Reopen the newest save. `reopened` is the frame on screen when it came from history: it waits
 * in `ahead` until the frames before it are done. A frame from the queue is still unlabeled, so
 * the queue serves it again later. Null when there is nothing to undo.
 */
export function stepBack<T>(history: History<T>, reopened: T | null): { history: History<T>; entry: T } | null {
  const entry = history.back.at(-1);
  if (entry === undefined) return null;
  const back = { back: history.back.slice(0, -1), ahead: history.ahead };
  return { entry, history: reopened === null ? back : park(back, reopened) };
}

/** Set aside a reopened frame that leaves the screen unsaved, so moving forward comes back to it. */
export function park<T>(history: History<T>, reopened: T): History<T> {
  return { back: history.back, ahead: [reopened, ...history.ahead] };
}

/**
 * Record `entry` as the newest save, and return the next reopened frame to show, or null to go
 * back to the queue. A reopened frame that is skipped is recorded with its old label.
 */
export function stepForward<T>(history: History<T>, entry: T): { history: History<T>; next: T | null } {
  const [next = null, ...ahead] = history.ahead;
  return { next, history: { back: [...history.back, entry], ahead } };
}
