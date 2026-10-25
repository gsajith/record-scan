"use client";
import { useCallback, useEffect, useState } from "react";
import type { Release } from "@/lib/catalog";
import { KINDS, TAGS, type Kind, type Tag } from "@/lib/labels";
import styles from "./page.module.css";

type Photo = { sha1: string; file: string };
type Result = Release & { thumb: string };

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

export default function LabelerPage() {
  const [photo, setPhoto] = useState<Photo | null>(null);
  const [busy, setBusy] = useState<"ingest" | "save" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Result[]>([]);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [picked, setPicked] = useState<Result | null>(null);
  const [kind, setKind] = useState<Kind>("positive");
  const [tags, setTags] = useState<Tag[]>([]);
  const [notes, setNotes] = useState("");

  // Clear the choices for the next photo
  const resetForm = useCallback(() => {
    setQuery("");
    setResults([]);
    setPicked(null);
    setKind("positive");
    setTags([]);
    setNotes("");
  }, []);

  const ingest = useCallback(async (file: File) => {
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
      setPhoto({ sha1: body.sha1, file: body.file });
      resetForm();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }, [resetForm]);

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

  async function save() {
    if (!photo) return;
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
      setStatus(kind === "positive" && picked ? `Saved as ${picked.title}. Paste the next photo.` : `Saved as ${kind}. Paste the next photo.`);
      setPhoto(null);
      resetForm();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  const canSave = photo !== null && busy === null && (kind !== "positive" || picked !== null);

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
            <p>{busy === "ingest" ? "Saving photo…" : status ?? "Paste a photo, or drop one here."}</p>
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
        <h1 className={styles.heading}>Gold set</h1>

        <fieldset className={styles.kinds} disabled={!photo}>
          <legend>Kind</legend>
          {KINDS.map((k) => (
            <label key={k} className={k === kind ? styles.kindOn : styles.kind}>
              <input type="radio" name="kind" value={k} checked={k === kind} onChange={() => setKind(k)} />
              {k}
            </label>
          ))}
        </fieldset>

        {kind === "positive" && (
          <div className={styles.search}>
            <input
              type="search"
              placeholder="Search title or artist"
              aria-label="Search the catalog"
              value={query}
              disabled={!photo}
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
            <ul className={styles.results}>
              {results.map((r) => (
                <li key={r.mbid}>
                  <button
                    type="button"
                    className={r.mbid === picked?.mbid ? styles.resultOn : styles.result}
                    aria-pressed={r.mbid === picked?.mbid}
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
          {TAGS.map((t) => (
            <label key={t}>
              <input
                type="checkbox"
                checked={tags.includes(t)}
                onChange={(e) => setTags(e.target.checked ? [...tags, t] : tags.filter((x) => x !== t))}
              />
              {t}
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
        <button type="button" className={styles.save} disabled={!canSave} onClick={save}>
          {busy === "save" ? "Saving…" : "Save"}
        </button>
      </section>
    </main>
  );
}
