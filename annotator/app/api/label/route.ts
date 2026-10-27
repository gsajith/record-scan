import { existsSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import { setDir } from "@/lib/data";
import { checkLabel, SETS, SHA1_RE, splitFor } from "@/lib/labels";
import { appendLabel, groupOf } from "@/lib/store";

// The label itself is checked by checkLabel once the server has filled in split and time
const RequestSchema = z.object({ set: z.enum(SETS), label: z.record(z.string(), z.unknown()) });

// POST /api/label, json { set, label } -> { ok: true }
export async function POST(request: Request) {
  const body = RequestSchema.safeParse(await request.json().catch(() => null));
  if (!body.success) return Response.json({ error: "expected { set, label }" }, { status: 400 });
  const { set, label } = body.data;
  // The server owns the group, the split and the time, so a client cannot move a photo between halves
  const sha1 = String(label.sha1 ?? "");
  const group = SHA1_RE.test(sha1) ? groupOf(setDir(set), sha1) : null;
  const checked = checkLabel(set, {
    ...label,
    group,
    split: splitFor(set, sha1, group),
    labeled_at: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
  });
  if ("error" in checked) return Response.json({ error: checked.error }, { status: 400 });
  if (checked.label.file !== `photos/${sha1}.jpg` || !existsSync(join(setDir(set), checked.label.file))) {
    return Response.json({ error: "label a photo that has been ingested" }, { status: 400 });
  }
  appendLabel(setDir(set), checked.label);
  return Response.json({ ok: true });
}
