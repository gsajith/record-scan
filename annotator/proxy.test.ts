import { NextRequest } from "next/server";
import { describe, expect, it } from "vitest";
import { proxy } from "./proxy";

function call(host: string, site?: string): number {
  const headers = new Headers({ host });
  if (site) headers.set("sec-fetch-site", site);
  return proxy(new NextRequest(`http://${host}/api/queue`, { method: "POST", headers })).status;
}

describe("proxy", () => {
  it("lets the labeler's own page through on localhost or 127.0.0.1", () => {
    expect(call("127.0.0.1:3000", "same-origin")).toBe(200);
    expect(call("localhost:3000", "same-origin")).toBe(200);
  });

  // DNS rebinding: an outside page whose domain now points at 127.0.0.1 looks same-origin
  it("refuses requests addressed to any other host name", () => {
    expect(call("evil.example:3000", "same-origin")).toBe(403);
  });

  it("refuses requests a browser marks as cross-site", () => {
    expect(call("127.0.0.1:3000", "cross-site")).toBe(403);
  });
});
