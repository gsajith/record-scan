import { z } from "zod";
import { parseSet, setDir } from "@/lib/data";
import { splitFor, type SetName } from "@/lib/labels";
import { ingestImage, ingestWild, NearDuplicateError, NotAnImageError } from "@/lib/store";

const MAX_DOWNLOAD = 30 * 1024 * 1024;
const WebUrl = z.url({ protocol: /^https?$/ });

// POST /api/ingest
//   multipart { set, file, source_url? }   a pasted, dropped or chosen file; source_url when a
//                                          drag out of a web page gave one
//   json      { set: "wild", url }         an image address pasted or dragged in; it is also the source
// -> { sha1, split, file }
export async function POST(request: Request) {
  let set: SetName | null;
  let bytes: Buffer;
  let sourceUrl: string | null;
  if (request.headers.get("content-type")?.includes("application/json")) {
    const body = (await request.json().catch(() => null)) as { set?: unknown; url?: unknown } | null;
    if (body?.set !== "wild") return Response.json({ error: "Image links go in the wild set." }, { status: 400 });
    const url = WebUrl.safeParse(body.url);
    if (!url.success) return Response.json({ error: "That address isn't an http or https link." }, { status: 400 });
    set = "wild";
    sourceUrl = url.data;
    try {
      bytes = await download(url.data);
    } catch (e) {
      return Response.json({ error: `Couldn't download that image: ${(e as Error).message}` }, { status: 400 });
    }
  } else {
    const form = await request.formData();
    set = parseSet(form.get("set"));
    const file = form.get("file");
    if (!set) return Response.json({ error: "set must be gold or wild" }, { status: 400 });
    if (!(file instanceof File)) return Response.json({ error: "no image in the upload" }, { status: 400 });
    bytes = Buffer.from(await file.arrayBuffer());
    sourceUrl = WebUrl.safeParse(form.get("source_url")).data ?? null;
  }

  try {
    const saved = set === "wild" ? await ingestWild(setDir(set), bytes, sourceUrl) : await ingestImage(setDir(set), bytes);
    return Response.json({ ...saved, split: splitFor(set, saved.sha1) });
  } catch (e) {
    if (e instanceof NotAnImageError) {
      return Response.json({ error: "That file is not an image the labeler can read." }, { status: 400 });
    }
    if (e instanceof NearDuplicateError) return Response.json({ error: e.message, match: e.match }, { status: 409 });
    // Disk trouble, such as RS_DATA_DIR on an unmounted drive or a full disk
    return Response.json({ error: `Couldn't save the photo: ${(e as Error).message}` }, { status: 500 });
  }
}

/** Fetch an image the user pointed at, with a time and size limit. */
async function download(url: string): Promise<Buffer> {
  const res = await fetch(url, { signal: AbortSignal.timeout(20_000) });
  if (!res.ok || !res.body) throw new Error(`the server answered ${res.status}`);
  if (Number(res.headers.get("content-length")) > MAX_DOWNLOAD) throw new Error("it is over 30 MB");
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return Buffer.concat(chunks);
    size += value.length;
    if (size > MAX_DOWNLOAD) {
      await reader.cancel();
      throw new Error("it is over 30 MB");
    }
    chunks.push(value);
  }
}
