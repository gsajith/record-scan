"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { Release } from "@/lib/catalog";
import { hotkeyAction, KIND_KEYS, TAG_KEYS } from "@/lib/hotkeys";
import { KINDS, TAGS, type Kind, type SetName, type Split, type Tag } from "@/lib/labels";
import type { QueueItem } from "@/lib/queue";
import styles from "./page.module.css";

type Result = Release & { thumb: string };
type Counts = Record<SetName, Record<Split, number>>;

// Gold only for now; the wild set, with its source and license fields, comes next
const SET = "gold";

// Browsers often give HEIC files no type, so the extension counts too
const IMAGE_EXT = /\.(heic|heif)$/i;

function firstImage(files: FileList | undefined): File | undefined {
  return Array.from(files ?? []).find((f) => f.type.startsWith("image/") || IMAGE_EXT.test(f.name));
}

function errorText(body: unknown, fallback: string): string {
  return typeof body === "object" && body !== null && "error" in body ? String(body.error) : fallback;
}

type QueueResponse = { next: QueueItem | null; remaining: number; counts: Counts };

async function fetchQueue(skip: string[]): Promise<QueueResponse> {
  const res = await fetch(`/api/queue?${new URLSearchParams({ set: SET, skip: skip.join(",") })}`);
  const body = await res.json();
  if (!res.ok) throw new Error(errorText(body, "Couldn't load the next photo."));
  return body;
}

/** Focus is in a field where letters and Enter mean typing, not shortcuts. */
function isTyping(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && target.matches('input[type="search"], input[type="text"], textarea, select');
}

export default function LabelerPage() {
  const [photo, setPhoto] = useState<QueueItem | null>(null);
  const [remaining, setRemaining] = useState(0);
  const [counts, setCounts] = useState<Counts | null>(null);
  const [skipped, setSkipped] = useState<string[]>([]);
  const [busy, setBusy] = useState<"load" | "ingest" | "save" | null>("load");
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Result[]>([]);
  const [active, setActive] = useState(0);
  const [searchError, setSearchError] = useState<string | null>(null);
  // The album and kind carry over between frames of the same video, which show one record
  const [picked, setPicked] = useState<Result | null>(null);
  const [kind, setKind] = useState<Kind>("positive");
  const [tags, setTags] = useState<Tag[]>([]);
  const [notes, setNotes] = useState("");

  const searchRef = useRef<HTMLInputElement>(null);
  const lastVideo = useRef<string | null>(null);
  const [focusSearchTick, setFocusSearchTick] = useState(0);
  const focusSearch = useCallback(() => setFocusSearchTick((n) => n + 1), []);

  // Tags and notes describe one photo, so they start empty on the next
  const clearPerPhoto = useCallback(() => {
    setTags([]);
    setNotes("");
    setQuery("");
    setResults([]);
  }, []);

  // A new record: forget the album and kind, and get ready to search
  const startNewRecord = useCallback(() => {
    setPicked(null);
    setKind("positive");
    focusSearch();
  }, [focusSearch]);

  const applyQueue = useCallback(
    (body: QueueResponse) => {
      setPhoto(body.next);
      setRemaining(body.remaining);
      setCounts(body.counts);
      clearPerPhoto();
      const video = body.next?.video ?? null;
      // Pasted photos and the first frame of each video are a new record
      if (body.next && (video === null || video !== lastVideo.current)) startNewRecord();
      lastVideo.current = video;
    },
    [clearPerPhoto, startNewRecord],
  );

  // Callers set busy first; it starts as "load" for the first fetch
  const loadNext = useCallback(
    (skip: string[]) =>
      fetchQueue(skip)
        .then(applyQueue)
        .catch((e: Error) => setError(e.message))
        .finally(() => setBusy(null)),
    [applyQueue],
  );

  useEffect(() => {
    loadNext([]);
  }, [loadNext]);

  useEffect(() => {
    if (focusSearchTick > 0) searchRef.current?.focus();
  }, [focusSearchTick]);

  const ingest = useCallback(
    async (file: File) => {
      setBusy("ingest");
      setError(null);
      setStatus(null);
      const form = new FormData();
      form.append("set", SET);
      form.append("file", file);
      try {
        const res = await fetch("/api/ingest", { method: "POST", body: form });
        const body = await res.json();
        if (!res.ok) throw new Error(errorText(body, "Couldn't save that photo."));
        setPhoto({ sha1: body.sha1, file: body.file, video: null });
        clearPerPhoto();
        startNewRecord();
        lastVideo.current = null;
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setBusy(null);
      }
    },
    [clearPerPhoto, startNewRecord],
  );

  // Paste an image from anywhere on the page
  useEffect(() => {
    function onPaste(e: ClipboardEvent) {
      const file = firstImage(e.clipboardData?.files);
      if (file) {
        e.preventDefault();
        ingest(file);
      }
    }
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [ingest]);

  // Search as you type, a beat after the last key
  useEffect(() => {
    if (!query.trim()) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`, { signal: controller.signal });
        const body = await res.json();
        if (!res.ok) throw new Error(errorText(body, "Search failed."));
        setResults(body.results);
        setActive(0);
        setSearchError(null);
      } catch (e) {
        if (!controller.signal.aborted) setSearchError((e as Error).message);
      }
    }, 200);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  // Keep the highlighted search result in view while arrowing through the list
  useEffect(() => {
    document.getElementById(`result-${active}`)?.scrollIntoView({ block: "nearest" });
  }, [active]);

  const needsAlbum = kind === "positive" && picked === null;
  const canSave = photo !== null && busy === null && !needsAlbum;

  async function save() {
    if (!photo || !canSave) return;
    setBusy("save");
    setError(null);
    const label = {
      file: photo.file,
      sha1: photo.sha1,
      kind,
      mbid: kind === "positive" ? picked?.mbid ?? null : null,
      tags,
      source_url: null,
      license: "own",
      notes,
    };
    try {
      const res = await fetch("/api/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ set: SET, label }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(errorText(body, "Couldn't save the label."));
      setStatus(kind === "positive" && picked ? `Saved as ${picked.title}` : `Saved as ${kind}`);
    } catch (e) {
      setError((e as Error).message);
      setBusy(null);
      return;
    }
    await loadNext(skipped);
  }

  function skip() {
    if (!photo) return;
    const next = [...skipped, photo.sha1];
    setSkipped(next);
    setStatus(null);
    setBusy("load");
    loadNext(next);
  }

  function toggleTag(t: Tag) {
    setTags((current) => (current.includes(t) ? current.filter((x) => x !== t) : [...current, t]));
  }

  // Shortcuts; see lib/hotkeys.ts. Re-bound each render so the handler sees current state.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      // Enter on a focused button already clicks it
      if (e.key === "Enter" && e.target instanceof HTMLButtonElement) return;
      const action = hotkeyAction({ key: e.key, typing: isTyping(e.target), modifier: e.metaKey || e.ctrlKey || e.altKey });
      if (!action || !photo) return;
      e.preventDefault();
      if (action.type === "kind") setKind(action.kind);
      else if (action.type === "tag") toggleTag(action.tag);
      else if (action.type === "skip") skip();
      else if (action.type === "search" || (action.type === "save" && needsAlbum)) {
        setKind("positive");
        focusSearch();
      } else if (action.type === "save") save();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  function onSearchKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(0, Math.min(results.length - 1, i + (e.key === "ArrowDown" ? 1 : -1))));
    } else if (e.key === "Enter" && results[active]) {
      // Pick and leave the box, so the next Enter saves
      e.preventDefault();
      setPicked(results[active]);
      e.currentTarget.blur();
    } else if (e.key === "Escape") {
      e.currentTarget.blur();
    }
  }

  let emptyText = status ? `${status}. ` : "";
  if (busy === "load") emptyText = "Loading…";
  else if (error) emptyText = error;
  else if (busy === "ingest") emptyText = "Saving photo…";
  else if (remaining > 0) emptyText += "You skipped the rest. Reload the page to see them again.";
  else emptyText += "Nothing left in the inbox. Paste a photo, or drop one here.";

  return (
    <main
      className={styles.main}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        const file = firstImage(e.dataTransfer.files);
        if (file) ingest(file);
      }}
    >
      <section className={styles.photo} aria-label="Photo">
        {photo ? (
          // eslint-disable-next-line @next/next/no-img-element -- a local file served by /api/image, no optimizer needed
          <img src={`/api/image/${SET}/${photo.sha1}`} alt="Photo being labeled" />
        ) : (
          <div className={styles.empty}>
            <p>{emptyText}</p>
            <label className={styles.button}>
              Choose a file
              <input
                type="file"
                accept="image/*,.heic,.heif"
                hidden
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) ingest(file);
                  e.target.value = "";
                }}
              />
            </label>
          </div>
        )}
      </section>

      <section className={styles.panel} aria-label="Label">
        <div className={styles.strip}>
          <h1 className={styles.heading}>Gold set</h1>
          {counts && (
            <p className={styles.hint}>
              Gold {counts.gold.eval} eval · Wild {counts.wild.eval} eval, {counts.wild.sim} sim · {remaining} to label
            </p>
          )}
          <p className={styles.saved} aria-live="polite">
            {photo && status}
          </p>
        </div>

        <fieldset className={styles.kinds} disabled={!photo}>
          <legend>Kind</legend>
          {KINDS.map((k) => (
            <label key={k} className={k === kind ? styles.kindOn : styles.kind}>
              <input type="radio" name="kind" value={k} checked={k === kind} onChange={() => setKind(k)} />
              {k} <kbd>{KIND_KEYS[k].toUpperCase()}</kbd>
            </label>
          ))}
        </fieldset>

        {kind === "positive" && (
          <div className={styles.search}>
            <input
              ref={searchRef}
              type="search"
              placeholder="Search title or artist"
              aria-label="Search the catalog (press / to focus)"
              role="combobox"
              aria-expanded={results.length > 0}
              aria-controls="results"
              aria-activedescendant={results[active] ? `result-${active}` : undefined}
              value={query}
              disabled={!photo}
              onKeyDown={onSearchKeyDown}
              onChange={(e) => {
                setQuery(e.target.value);
                if (!e.target.value.trim()) setResults([]);
              }}
            />
            {picked && (
              <p className={styles.picked}>
                {picked.title} · {picked.artists.join(", ") || "Unknown artist"}
              </p>
            )}
            {searchError && <p className={styles.error}>{searchError}</p>}
            {query.trim() && !searchError && results.length === 0 && <p className={styles.hint}>No releases match.</p>}
            <ul className={styles.results} id="results" role="listbox" aria-label="Search results">
              {results.map((r, i) => (
                <li key={r.mbid}>
                  <button
                    type="button"
                    id={`result-${i}`}
                    role="option"
                    aria-selected={r.mbid === picked?.mbid}
                    className={[
                      r.mbid === picked?.mbid ? styles.resultOn : styles.result,
                      i === active ? styles.resultActive : "",
                    ].join(" ")}
                    tabIndex={-1}
                    onClick={() => setPicked(r)}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element -- Cover Art Archive thumbnail, hot-linked for display only */}
                    <img
                      src={r.thumb}
                      alt=""
                      width={56}
                      height={56}
                      loading="lazy"
                      // Some covers have no thumbnail; show the empty tile instead of a broken image
                      onError={(e) => e.currentTarget.removeAttribute("src")}
                    />
                    <span>
                      <strong>{r.title}</strong>
                      <span>{r.artists.join(", ") || "Unknown artist"}</span>
                      {r.format && <span className={styles.hint}>{r.format}</span>}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        <fieldset className={styles.tags} disabled={!photo}>
          <legend>Tags</legend>
          {TAGS.map((t, i) => (
            <label key={t}>
              <input type="checkbox" checked={tags.includes(t)} onChange={() => toggleTag(t)} />
              {t} <kbd>{TAG_KEYS[i]}</kbd>
            </label>
          ))}
        </fieldset>

        <label className={styles.notes}>
          Notes
          <textarea value={notes} rows={2} disabled={!photo} onChange={(e) => setNotes(e.target.value)} />
        </label>

        {error && (
          <p className={styles.error} role="alert">
            {error}
          </p>
        )}
        <div className={styles.actions}>
          <button type="button" className={styles.skip} disabled={!photo || busy !== null} onClick={skip}>
            Skip <kbd>S</kbd>
          </button>
          <button type="button" className={styles.save} disabled={!canSave} onClick={save}>
            {busy === "save" ? "Saving…" : "Save"} <kbd>↵</kbd>
          </button>
        </div>
      </section>
    </main>
  );
}
