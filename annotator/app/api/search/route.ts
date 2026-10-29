import { getCatalog } from "@/lib/data";

// GET /api/search?q=abbey+road&limit=20. At most 500, so a one-word search stays quick to draw.
export function GET(request: Request) {
  const params = new URL(request.url).searchParams;
  const limit = Math.min(Math.max(Number(params.get("limit")) || 20, 1), 500);
  let catalog;
  try {
    catalog = getCatalog();
  } catch (e) {
    // Usually the catalog has not been built yet; the message says how
    return Response.json({ error: (e as Error).message }, { status: 503 });
  }
  const results = catalog
    .search(params.get("q") ?? "", limit)
    .map((r) => ({ ...r, thumb: `https://coverartarchive.org/release/${r.mbid}/front-250` }));
  return Response.json({ results });
}
