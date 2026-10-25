import { NextResponse, type NextRequest } from "next/server";

// The API writes into the gold and wild sets, so only this page may call it. Browsers send
// Sec-Fetch-Site on every request; a form or fetch from another site says "cross-site" and is
// refused before it reaches a route. Tools without the header (curl, scripts) still work.
export function proxy(request: NextRequest) {
  const site = request.headers.get("sec-fetch-site");
  if (site && site !== "same-origin" && site !== "none") {
    return NextResponse.json({ error: "The labeler only accepts requests from its own page." }, { status: 403 });
  }
  return NextResponse.next();
}

export const config = { matcher: "/api/:path*" };
