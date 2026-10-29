import { describe, expect, it } from "vitest";
import { emptyHistory, stepBack, stepForward, type History } from "./history";

// Saves the frames in order, the way the page records each save
function saved(...frames: string[]): History<string> {
  return frames.reduce((h, f) => stepForward(h, f).history, emptyHistory<string>());
}

describe("history", () => {
  it("has nothing to undo before the first save", () => {
    expect(stepBack(emptyHistory<string>(), null)).toBeNull();
  });

  it("reopens the newest save, then steps back one more on each undo", () => {
    const first = stepBack(saved("a", "b", "c"), null);
    expect(first?.entry).toBe("c");
    const second = stepBack(first!.history, "c");
    expect(second?.entry).toBe("b");
    expect(stepBack(second!.history, "b")?.entry).toBe("a");
    expect(stepBack(stepBack(second!.history, "b")!.history, "a")).toBeNull();
  });

  it("after saving a reopened frame, moves forward through the frames stepped past, then back to the queue", () => {
    const c = stepBack(saved("a", "b", "c"), null)!;
    const b = stepBack(c.history, "c")!;
    const fixedB = stepForward(b.history, "b2");
    expect(fixedB.next).toBe("c");
    const fixedC = stepForward(fixedB.history, "c2");
    expect(fixedC.next).toBeNull();
    // The corrected saves are what a later undo reopens
    expect(stepBack(fixedC.history, null)?.entry).toBe("c2");
    expect(stepBack(stepBack(fixedC.history, null)!.history, "c2")?.entry).toBe("b2");
  });
});
