import { z } from "zod";
import { setDir } from "@/lib/data";
import { SETS, SHA1_RE } from "@/lib/labels";
import { nextInQueue } from "@/lib/queue";
import { countLabels } from "@/lib/store";

// A POST because it moves inbox frames, and the skip list can grow past what a URL holds
const RequestSchema = z.object({ set: z.enum(SETS), skip: z.array(z.string().regex(SHA1_RE)).default([]) });

// POST /api/queue, json { set, skip } -> { next, remaining, counts }
// counts holds labels per split for every set, for the status strip.
export async function POST(request: Request) {
  const body = RequestSchema.safeParse(await request.json().catch(() => null));
  if (!body.success) return Response.json({ error: "expected { set, skip }" }, { status: 400 });
  const { set, skip } = body.data;
  try {
    const { next, remaining } = await nextInQueue(setDir(set), skip);
    const counts = Object.fromEntries(SETS.map((s) => [s, countLabels(setDir(s))]));
    return Response.json({ next, remaining, counts });
  } catch (e) {
    return Response.json({ error: (e as Error).message }, { status: 500 });
  }
}
