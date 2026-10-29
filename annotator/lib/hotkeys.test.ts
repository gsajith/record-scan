import { describe, expect, it } from "vitest";
import { hotkeyAction } from "./hotkeys";

const press = (key: string) => hotkeyAction({ key, typing: false, modifier: false });

describe("hotkeyAction", () => {
  it("maps P, N and U to the kind", () => {
    expect([press("p"), press("N"), press("u")]).toEqual([
      { type: "kind", kind: "positive" },
      { type: "kind", kind: "negative" },
      { type: "kind", kind: "unknown" },
    ]);
  });

  it("maps 1 to 9, 0, - and = to the twelve tags in order", () => {
    expect([press("1"), press("0"), press("-"), press("=")]).toEqual([
      { type: "tag", tag: "tilt" },
      { type: "tag", tag: "partial" },
      { type: "tag", tag: "crate" },
      { type: "tag", tag: "flash" },
    ]);
  });

  it("maps S to skip, Enter to save and / to search", () => {
    expect([press("s"), press("Enter"), press("/")]).toEqual([
      { type: "skip" },
      { type: "save" },
      { type: "search" },
    ]);
  });

  it("maps Command-Z or Control-Z to undo, but leaves a text field its own undo", () => {
    expect(hotkeyAction({ key: "z", typing: false, modifier: true })).toEqual({ type: "undo" });
    expect(hotkeyAction({ key: "z", typing: true, modifier: true })).toBeNull();
    expect(press("z")).toBeNull();
  });

  it("does nothing while typing in a field or with Command, Control or Option held", () => {
    for (const key of ["p", "1", "s", "Enter", "/"]) {
      expect(hotkeyAction({ key, typing: true, modifier: false })).toBeNull();
      expect(hotkeyAction({ key, typing: false, modifier: true })).toBeNull();
    }
    expect(press("x")).toBeNull();
  });
});
