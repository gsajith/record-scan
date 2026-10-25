import { parseSet, setDir } from "@/lib/data";
import { SETS, SHA1_RE } from "@/lib/labels";
import { nextInQueue } from "@/lib/queue";
import { countLabels } from "@/lib/store";

// GET /api/queue?set=gold&skip=<sha1>,<sha1> -> { next, remaining, counts }
// counts holds labels per split for every set, for the status strip.
export async function GET(request: Request) {
  const params = new URL(request.url).searchParams;
  const set = parseSet(params.get("set"));
  if (!set) return Response.json({ error: "set must be gold or wild" }, { status: 400 });
  const skip = (params.get("skip") ?? "").split(",").filter((s) => SHA1_RE.test(s));
  try {
    const { next, remaining } = await nextInQueue(setDir(set), skip);
    const counts = Object.fromEntries(SETS.map((s) => [s, countLabels(setDir(s))]));
    return Response.json({ next, remaining, counts });
  } catch (e) {
    return Response.json({ error: (e as Error).message }, { status: 500 });
  }
}
