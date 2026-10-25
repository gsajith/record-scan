// Folders under RS_DATA_DIR, the same defaults as pipeline/config.py. Point RS_DATA_DIR at
// another folder to label the sealed holdout; the set folder names stay gold and wild.
import { homedir } from "node:os";
import { join } from "node:path";
import { openCatalog } from "./catalog";
import { SETS, type SetName } from "./labels";

export function dataDir(): string {
  const dir = process.env.RS_DATA_DIR ?? "~/Projects/record-scan-data";
  return dir.startsWith("~/") ? join(homedir(), dir.slice(2)) : dir;
}

export function setDir(set: SetName): string {
  return join(dataDir(), set);
}

export function parseSet(value: unknown): SetName | null {
  return SETS.find((s) => s === value) ?? null;
}

let catalog: ReturnType<typeof openCatalog> | undefined;

/** One read-only connection per server process; the catalog is 1.4 GB, so open it once. */
export function getCatalog() {
  catalog ??= openCatalog(join(dataDir(), "metadata", "catalog.sqlite"));
  return catalog;
}
