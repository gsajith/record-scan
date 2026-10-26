import { z } from "zod";
import { parseSet, setDir } from "@/lib/data";
import { LICENSES, splitFor, type License, type SetName } from "@/lib/labels";
import { ingestImage, ingestWild, NearDuplicateError, NotAnImageError } from "@/lib/store";

const MAX_DOWNLOAD = 30 * 1024 * 1024;
const WILD_RULE = "A wild photo needs the web page it came from and a license that allows it.";

const UrlSchema = z.object({
  set: z.enum(["gold", "wild"]),
  url: z.url({ protocol: /^https?$/ }),
  source_url: z.string().optional(),
  license: z.string().optional(),
});

// POST /api/ingest
//   multipart { set, file, source_url?, license? }   a pasted, dropped or chosen file
//   json      { set, url, source_url?, license? }    an image address pasted or dropped from a page
// -> { sha1, split, file }
export async function POST(request: Request) {
  let set: SetName | null;
  let bytes: Buffer;
  let sourceUrl: unknown;
  let license: unknown;
  if (request.headers.get("content-type")?.includes("application/json")) {
    const body = UrlSchema.safeParse(await request.json().catch(() => null));
    if (!body.success) return Response.json({ error: "That address isn't an http or https link." }, { status: 400 });
    ({ set, source_url: sourceUrl, license } = body.data);
    try {
      bytes = await download(body.data.url);
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
    sourceUrl = form.get("source_url");
    license = form.get("license");
  }

  try {
    let saved;
    if (set === "wild") {
      // D9: a wild photo is only kept with where it came from and a license that allows it
      const url = z.url({ protocol: /^https?$/ }).safeParse(sourceUrl);
      const allowed = LICENSES.find((l): l is License => l === license);
      if (!url.success || !allowed) return Response.json({ error: WILD_RULE }, { status: 400 });
      saved = await ingestWild(setDir(set), bytes, { url: url.data, license: allowed });
    } else {
      saved = await ingestImage(setDir(set), bytes);
    }
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
  const res = await fetch(url, { signal: AbortSignal.timeout(20_000), redirect: "follow" });
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
