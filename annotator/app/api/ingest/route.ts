import { parseSet, setDir } from "@/lib/data";
import { LICENSES, splitFor } from "@/lib/labels";
import { ingestImage } from "@/lib/store";

// POST /api/ingest, multipart { set, file, source_url?, license? } -> { sha1, split, file }
export async function POST(request: Request) {
  const form = await request.formData();
  const set = parseSet(form.get("set"));
  const file = form.get("file");
  if (!set) return Response.json({ error: "set must be gold or wild" }, { status: 400 });
  if (!(file instanceof File)) return Response.json({ error: "no image in the upload" }, { status: 400 });
  // D9: a wild photo is only kept with where it came from and a license that allows it
  if (set === "wild") {
    const license = form.get("license");
    if (!form.get("source_url") || !LICENSES.some((l) => l === license)) {
      return Response.json({ error: "A wild photo needs its source URL and license." }, { status: 400 });
    }
  }
  try {
    const saved = await ingestImage(setDir(set), Buffer.from(await file.arrayBuffer()));
    return Response.json({ ...saved, split: splitFor(set, saved.sha1) });
  } catch {
    return Response.json({ error: "That file is not an image the labeler can read." }, { status: 400 });
  }
}
