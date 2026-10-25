import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { parseSet, setDir } from "@/lib/data";

// GET /api/image/<set>/<sha1> -> the saved JPEG, so the page can show local files
export async function GET(_request: Request, { params }: { params: Promise<{ set: string; sha1: string }> }) {
  const { set: rawSet, sha1 } = await params;
  const set = parseSet(rawSet);
  // The sha1 check also keeps the path inside the set folder
  if (!set || !/^[0-9a-f]{40}$/.test(sha1)) return new Response("Not found", { status: 404 });
  try {
    const jpeg = await readFile(join(setDir(set), "photos", `${sha1}.jpg`));
    return new Response(new Uint8Array(jpeg), {
      headers: { "Content-Type": "image/jpeg", "Cache-Control": "private, max-age=31536000, immutable" },
    });
  } catch {
    return new Response("Not found", { status: 404 });
  }
}
