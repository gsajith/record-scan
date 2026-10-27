import { NextResponse, type NextRequest } from "next/server";

// The API writes into the gold and wild sets, so only this page on this Mac may call it.
// - Host: the server listens on 127.0.0.1, but a page on another domain that re-points its DNS
//   at 127.0.0.1 (DNS rebinding) would look same-origin; its Host header still names that domain.
// - Sec-Fetch-Site: browsers send it on every request; a form or fetch from another site says
//   "cross-site". Tools without the header (curl, scripts) still work.
const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost"]);

export function proxy(request: NextRequest) {
  const hostname = (request.headers.get("host") ?? "").replace(/:\d+$/, "");
  const site = request.headers.get("sec-fetch-site");
  if (!LOCAL_HOSTS.has(hostname) || (site && site !== "same-origin" && site !== "none")) {
    return NextResponse.json({ error: "The labeler only accepts requests from its own page." }, { status: 403 });
  }
  return NextResponse.next();
}

export const config = { matcher: "/api/:path*" };
