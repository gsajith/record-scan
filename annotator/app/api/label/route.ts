import { existsSync } from "node:fs";
import { join } from "node:path";
import { parseSet, setDir } from "@/lib/data";
import { checkLabel, splitFor } from "@/lib/labels";
import { appendLabel } from "@/lib/store";

// POST /api/label, json { set, label } -> { ok: true }
export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as { set?: unknown; label?: Record<string, unknown> } | null;
  const set = parseSet(body?.set);
  if (!set || typeof body?.label !== "object" || body.label === null) {
    return Response.json({ error: "expected { set, label }" }, { status: 400 });
  }
  // The server owns the split and the time, so a client cannot move a photo between halves
  const sha1 = String(body.label.sha1 ?? "");
  const checked = checkLabel(set, {
    ...body.label,
    split: splitFor(set, sha1),
    labeled_at: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
  });
  if ("error" in checked) return Response.json({ error: checked.error }, { status: 400 });
  if (checked.label.file !== `photos/${sha1}.jpg` || !existsSync(join(setDir(set), checked.label.file))) {
    return Response.json({ error: "label a photo that has been ingested" }, { status: 400 });
  }
  appendLabel(setDir(set), checked.label);
  return Response.json({ ok: true });
}
