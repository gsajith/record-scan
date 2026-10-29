"""Score a model on the eval half of the gold or wild set.

    python -m pipeline.retrieval.eval --model NAME --set gold|wild [--distractors 50000] [--views 1|3] [--px N] [--seed 0] [--dry-run]

The rules here are locked (plan, Global constraints, eval integrity): the hit rule, the use of
the eval split only, the distractor sampler and its seed. Changing any of them needs a recorded
ruling in the ledger and a commit of its own, never one that also touches model code.
"""

import argparse
import hashlib
import html
import json
import shlex
import subprocess
import sys
from collections.abc import Collection
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TypedDict

import numpy as np
from PIL import Image

from pipeline import config
from pipeline.gallery.exclusions import exclusions_hash, load_exclusions
from pipeline.gold.labels import KINDS, SETS, TAGS, Label, load_labels
from pipeline.metadata.catalog import Catalog, Cover


@dataclass
class EvalItem:
    label: Label
    targets: set[str]  # mbids that count as a hit; empty for negative and unknown photos


def load_eval_set(set_dir: Path, catalog: Catalog, clusters: dict[str, int] | None) -> list[EvalItem]:
    """Eval-split labels with their hit targets.

    A positive whose release is no longer in the catalog stays in, with no targets: the app could
    not find it either, so it counts as a miss rather than leaving the denominator.

    Targets are the labeled release, every release in its MusicBrainz release group
    (Catalog.same_album), and, once duplicate clusters exist, every release in the same cluster
    as any of those.
    """
    members: dict[int, set[str]] = {}
    for mbid, cluster in (clusters or {}).items():
        members.setdefault(cluster, set()).add(mbid)
    items = []
    for label in load_labels(set_dir):
        if label.split != "eval":
            continue
        if label.kind != "positive":
            items.append(EvalItem(label, set()))
            continue
        assert label.mbid is not None  # load_labels guarantees an mbid on positives
        try:
            targets = set(catalog.same_album(label.mbid))
        except KeyError:
            print(f"Warning: {label.file}'s release {label.mbid} is not in the catalog; scored as a miss", file=sys.stderr)
            items.append(EvalItem(label, set()))
            continue
        if clusters:
            targets |= {m for t in list(targets) if t in clusters for m in members[clusters[t]]}
        items.append(EvalItem(label, targets))
    return items


def is_hit(result_mbid: str, item: EvalItem) -> bool:
    return result_mbid in item.targets


def _key(cover: Cover) -> str:
    """The cover's key in the gallery and the exclusion list: its file stem."""
    return Path(cover.file).stem


def build_mini_gallery(items: list[EvalItem], catalog: Catalog, distractors: int, seed: int,
                       excluded: Collection[str]) -> list[Cover]:
    """Every cover of every target release (sorted by mbid), then `distractors` covers drawn at
    random, leaving out the covers whose key is in excluded (pipeline.gallery.exclusions).

    The draw is a seeded permutation of the catalog's covers table, skipping target releases and
    excluded covers, so one seed, one catalog build and one exclusion list always give the same
    gallery. A catalog rebuild or a new list changes it; report.json records a hash of both so
    runs on different galleries are never compared.
    """
    targets = set().union(*(item.targets for item in items))
    target_covers = [c for mbid in sorted(targets) for c in catalog.covers(mbid)]
    gallery = [c for c in target_covers if _key(c) not in excluded]
    for item in items:
        if item.targets and not any(c.mbid in item.targets for c in gallery):
            print(f"Warning: no usable cover on file for {item.label.file}'s release; it cannot be a hit", file=sys.stderr)
    order = np.random.default_rng(seed).permutation(catalog.cover_count()) + 1
    # Target and excluded covers are the only rows to skip, so this many candidates always suffice
    rowids = order[: distractors + len(target_covers) + len(excluded)].tolist()
    candidates = catalog.covers_by_rowid(rowids)
    if len(candidates) != len(rowids):
        raise SystemExit("The catalog's covers table has gaps in its row numbers. Rebuild it with build_catalog_db.")
    return gallery + [c for c in candidates if c.mbid not in targets and _key(c) not in excluded][:distractors]


def fully_excluded(items: list[EvalItem], catalog: Catalog, excluded: Collection[str]) -> int:
    """Positives with covers on file, every one of them excluded: certain misses the list causes."""
    count = 0
    for item in items:
        covers = [c for mbid in item.targets for c in catalog.covers(mbid)]
        count += bool(covers) and all(_key(c) in excluded for c in covers)
    return count


class MatchRow(TypedDict):
    mbid: str
    title: str
    artists: list[str]
    score: float


class Row(TypedDict):
    """One scored photo, as stored in report.json."""

    file: str
    sha1: str
    kind: str
    mbid: str | None
    expected: str  # "Artist - Title" of the labeled release, empty for negatives
    tags: list[str]
    matches: list[MatchRow]  # identify's top k, best first
    hit_rank: int | None  # 1-based rank of the first hit among matches, None if no hit


def _top1(row: Row) -> float:
    return row["matches"][0]["score"] if row["matches"] else 0.0


@dataclass
class EvalReport:
    r1: float  # share of positives whose first match is a hit
    r5: float  # share of positives with a hit anywhere in the top 5
    per_tag: dict[str, tuple[float, int]]  # tag -> (r1 over positives with that tag, how many)
    neg_max: float | None  # highest top-1 score of a negative or unknown photo
    pos_min_top1: float | None  # lowest top-1 score among positives whose first match is a hit
    micro_ap: float  # SSCD: every photo ranked by top-1 score; average precision of "positive and hit at rank 1"
    rows: list[Row]

    @classmethod
    def from_rows(cls, rows: list[Row]) -> "EvalReport":
        """Negatives and unknowns (records missing from the catalog) both count as photos with no right answer."""
        positives = [r for r in rows if r["kind"] == "positive"]
        negatives = [r for r in rows if r["kind"] != "positive"]

        def recall(rs: list[Row], k: int) -> float:
            return sum(r["hit_rank"] is not None and r["hit_rank"] <= k for r in rs) / len(rs) if rs else 0.0

        per_tag = {}
        for tag in TAGS:
            tagged = [r for r in positives if tag in r["tags"]]
            if tagged:
                per_tag[tag] = (recall(tagged, 1), len(tagged))
        hits_at_1 = [_top1(r) for r in positives if r["hit_rank"] == 1]

        def is_correct(r: Row) -> bool:
            return r["kind"] == "positive" and r["hit_rank"] == 1

        precisions, correct = [], 0
        # Highest score first; among equal scores the wrong answers go first, so ties never help
        for rank, r in enumerate(sorted(rows, key=lambda r: (-_top1(r), is_correct(r))), start=1):
            if is_correct(r):
                correct += 1
                precisions.append(correct / rank)
        micro_ap = sum(precisions) / len(positives) if positives else 0.0

        return cls(
            r1=recall(positives, 1),
            r5=recall(positives, 5),
            per_tag=per_tag,
            neg_max=max(map(_top1, negatives), default=None),
            pos_min_top1=min(hits_at_1, default=None),
            micro_ap=micro_ap,
            rows=rows,
        )

    def to_markdown(self) -> str:
        def num(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.3f}"

        positives = sum(r["kind"] == "positive" for r in self.rows)
        lines = [
            "| Photos | Positives | Negatives and unknowns | r1 | r5 | micro AP | Highest negative | Lowest top-1 hit |",
            "|---|---|---|---|---|---|---|---|",
            f"| {len(self.rows)} | {positives} | {len(self.rows) - positives} | {num(self.r1)} | {num(self.r5)} "
            f"| {num(self.micro_ap)} | {num(self.neg_max)} | {num(self.pos_min_top1)} |",
            "",
            "| Tag | r1 | Positives |",
            "|---|---|---|",
            *(f"| {tag} | {num(r1)} | {n} |" for tag, (r1, n) in self.per_tag.items()),
        ]
        return "\n".join(lines) + "\n"


def _thumb(mbid: str) -> str:
    return f"https://coverartarchive.org/release/{mbid}/front-250"


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval misses</title>
<style>
:root {{ color-scheme: light dark; --bg: #fff; --fg: #111; --muted: #666; --hit: #1a7f37; --line: #ddd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #111; --fg: #eee; --muted: #999; --hit: #3fb950; --line: #333; }} }}
body {{ background: var(--bg); color: var(--fg); font: 14px/1.4 system-ui, sans-serif; margin: 16px; }}
.card {{ display: flex; flex-wrap: wrap; gap: 12px; padding: 12px 0; border-bottom: 1px solid var(--line); }}
figure {{ margin: 0; width: 160px; }}
figure.photo {{ width: 240px; }}
img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: var(--line); }}
figure.hit img {{ outline: 3px solid var(--hit); }}
figcaption {{ color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
</style></head><body>
<h1>Misses ({misses})</h1>
<p>Each photo whose first match is not a hit, most confident first, beside the expected cover and the top five. A green outline marks a hit.</p>
{miss_cards}
<h1>Lowest-scoring hits ({hits})</h1>
{hit_cards}
<h1>Highest-scoring negatives ({negatives})</h1>
<p>Photos with no right answer (negative or unknown), highest top-1 score first: the ones a lock threshold must reject.</p>
{negative_cards}
</body></html>
"""


def _card(row: Row) -> str:
    e = html.escape
    tags = ", ".join(row["tags"]) or "no tags"
    parts = [
        f'<figure class="photo"><img src="{e(Path(row["file"]).as_uri())}" alt="">'
        f'<figcaption>{e(Path(row["file"]).name)}<br>{e(tags)}</figcaption></figure>',
    ]
    if row["mbid"]:
        parts.append(f'<figure><img src="{_thumb(row["mbid"])}" alt="" loading="lazy">'
                     f'<figcaption>expected<br>{e(row["expected"])}</figcaption></figure>')
    for rank, m in enumerate(row["matches"], start=1):
        hit = ' class="hit"' if rank == row["hit_rank"] else ""
        parts.append(
            f'<figure{hit}><img src="{_thumb(m["mbid"])}" alt="" loading="lazy">'
            f'<figcaption>{rank}. {m["score"]:.3f}<br>{e(", ".join(m["artists"]))} - {e(m["title"])}</figcaption></figure>'
        )
    return f'<div class="card">{"".join(parts)}</div>'


def write_miss_gallery(report: EvalReport, path: Path) -> None:
    """misses.html: every positive whose first match is not a hit, the 20 lowest-scoring hits,
    and the 20 highest-scoring negatives and unknowns.

    Photos load from their local paths (file:// links, so the page only works on this Mac and
    no photo enters the repo); covers are hot-linked from the Cover Art Archive.
    """
    positives = [r for r in report.rows if r["kind"] == "positive"]
    misses = sorted((r for r in positives if r["hit_rank"] != 1), key=_top1, reverse=True)
    weakest_hits = sorted((r for r in positives if r["hit_rank"] == 1), key=_top1)[:20]
    strongest_negatives = sorted((r for r in report.rows if r["kind"] != "positive"), key=_top1, reverse=True)[:20]
    path.write_text(_PAGE.format(
        misses=len(misses), miss_cards="\n".join(map(_card, misses)),
        hits=len(weakest_hits), hit_cards="\n".join(map(_card, weakest_hits)),
        negatives=len(strongest_negatives), negative_cards="\n".join(map(_card, strongest_negatives)),
    ))


def _short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def config_hash(set_dir: Path, distractors: int, views: int, px: int | None, seed: int) -> str:
    """Same hash, same photos, labels and settings: runs of different models with one hash are comparable."""
    labels = set_dir / "labels.jsonl"
    settings = {"set": set_dir.name, "distractors": distractors, "views": views, "px": px, "seed": seed,
                "labels": _short_hash(labels.read_bytes()) if labels.exists() else None}
    return _short_hash(json.dumps(settings, sort_keys=True).encode())


def gallery_hash(gallery: list[Cover], excluded: dict[str, str]) -> str:
    """The cover list and the exclusion list it was drawn under."""
    return _short_hash(("\n".join(c.file for c in gallery) + f"\nexcluded {exclusions_hash(excluded)}").encode())


def set_dir_for(set_name: str) -> Path:
    return {"gold": config.GOLD_DIR, "wild": config.WILD_DIR}[set_name]


@dataclass
class EvalSet:
    """What an eval run scores against, built the same way for --dry-run and for a real run."""

    set_dir: Path
    catalog: Catalog
    items: list[EvalItem]
    gallery: list[Cover]
    excluded: dict[str, str]  # the gallery's exclusion list, cover key -> reason
    missing: int  # eval positives whose release left the catalog, scored as misses
    all_excluded: int  # eval positives whose every cover is excluded, scored as misses

    @classmethod
    def prepare(cls, set_name: str, distractors: int, seed: int) -> "EvalSet":
        set_dir = set_dir_for(set_name)
        catalog = Catalog()
        items = load_eval_set(set_dir, catalog, clusters=None)  # duplicate clusters join once Task 5 builds them
        missing = sum(item.label.kind == "positive" and not item.targets for item in items)
        excluded = load_exclusions(config.GALLERY_DIR)
        gallery = build_mini_gallery(items, catalog, distractors, seed, excluded)
        return cls(set_dir, catalog, items, gallery, excluded, missing, fully_excluded(items, catalog, excluded))

    def describe(self) -> str:
        kinds = {k: sum(item.label.kind == k for item in self.items) for k in KINDS}
        targets = set().union(*(item.targets for item in self.items))
        n_targets = sum(c.mbid in targets for c in self.gallery)
        return (f"{len(self.items)} eval photos: {kinds['positive']} positive, {kinds['negative']} negative, "
                f"{kinds['unknown']} unknown; {self.missing} positives no longer in the catalog, scored as misses; "
                f"{self.all_excluded} positives with every cover excluded\n"
                f"gallery: {n_targets} target covers + {len(self.gallery) - n_targets} distractors")


def _git_sha() -> str:
    repo = config.REPO_ROOT
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    # Earlier eval output under docs/results is not a code change
    dirty = subprocess.run(["git", "status", "--porcelain", "--", ".", ":!docs/results"],
                           cwd=repo, capture_output=True, text=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def run_eval(model: str, set_name: str, distractors: int, views: int, px: int | None, seed: int, work_dir: Path) -> EvalReport:
    """Embed the mini gallery (cached in work_dir), identify every eval photo, write the report.

    Writes docs/results/<date>-eval-<model>-<set>-<config hash>/ with report.md, report.json (git SHA, config
    hash, gallery hash, command, every row) and misses.html.
    """
    from pipeline.retrieval.build_index import build_index
    from pipeline.retrieval.embed import embed_images
    from pipeline.retrieval.embedders import load_embedder
    from pipeline.retrieval.identify import Identifier

    es = EvalSet.prepare(set_name, distractors, seed)
    catalog, items, gallery = es.catalog, es.items, es.gallery
    if not items:
        raise SystemExit(f"No eval photos labeled in {es.set_dir}")
    embedder = load_embedder(model, input_px=px)

    # Keyed by the embedder's signature, so changed preprocessing embeds a fresh gallery
    index_dir = work_dir / f"{embedder.signature}-{gallery_hash(gallery, es.excluded)}"
    # build_index writes signature.txt last, so a crash mid-build leaves a dir that rebuilds
    if not (index_dir / "signature.txt").exists():
        root = config.require_image_root()
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "errors.log").unlink(missing_ok=True)  # a retried embed logs its errors afresh
        stems, embs = embed_images([root / c.file for c in gallery], embedder, batch=64, workers=6,
                                   errors_log=index_dir / "errors.log")
        # A drive that went away turns every remaining cover into an error; never cache that gallery
        config.require_image_root()
        shard = index_dir / "gallery.npz"
        # Extra fronts are <mbid>-<cover_id>.jpg, so the first 36 characters of a stem are the mbid
        np.savez(shard, uuids=np.array([s[:36] for s in stems], dtype="U36"), embs=embs.astype(np.float16),
                 stems=np.array(stems), signature=embedder.signature)
        build_index([shard], embedder.dim, index_dir, M=32, ef_construction=200)
    identifier = Identifier(embedder, index_dir, catalog)
    # Hash the covers that were embedded, not the planned list: a gallery missing unreadable
    # covers gets its own hash, and a complete one matches the --dry-run hash
    embedded = set(np.load(index_dir / "gallery.npz")["stems"].tolist())
    embedded_gallery = [c for c in gallery if Path(c.file).stem in embedded]
    if len(embedded_gallery) < len(gallery):
        print(f"Warning: {len(gallery) - len(embedded_gallery)} gallery covers could not be read; "
              f"see {index_dir / 'errors.log'}", file=sys.stderr)

    rows: list[Row] = []
    for item in items:
        label = item.label
        matches = identifier.identify(Image.open(label.file), k=5, views=views)
        release = catalog.release(label.mbid) if label.mbid else None
        rows.append(Row(
            file=label.file, sha1=label.sha1, kind=label.kind, mbid=label.mbid,
            expected=f"{', '.join(release.artists)} - {release.title}" if release else "",
            tags=label.tags,
            matches=[MatchRow(mbid=m.mbid, title=m.title, artists=list(m.artists), score=m.score) for m in matches],
            hit_rank=next((rank for rank, m in enumerate(matches, start=1) if is_hit(m.mbid, item)), None),
        ))
    report = EvalReport.from_rows(rows)

    command = shlex.join(["python", "-m", "pipeline.retrieval.eval", "--model", model, "--set", set_name,
                          "--distractors", str(distractors), "--views", str(views), "--seed", str(seed),
                          *(["--px", str(px)] if px else [])])
    cfg_hash = config_hash(es.set_dir, distractors, views, px, seed)
    provenance = {"command": command, "git_sha": _git_sha(), "config_hash": cfg_hash,
                  "gallery_hash": gallery_hash(embedded_gallery, es.excluded), "gallery_size": len(gallery),
                  "gallery_embedded": len(embedded_gallery), "positives_missing": es.missing,
                  "positives_all_covers_excluded": es.all_excluded, "excluded_covers": len(es.excluded)}
    # The config hash keeps a same-day run with other settings from overwriting this one
    out = config.RESULTS_DIR / f"{date.today().isoformat()}-eval-{model}-{set_name}-{cfg_hash}"
    out.mkdir(parents=True, exist_ok=True)
    header = "\n".join([f"# Eval: {model} on {set_name}", "", f"`{command}`", "",
                        *(f"- {k}: {v}" for k, v in provenance.items() if k != "command"), "",
                        es.describe().replace("\n", "  \n"), ""])
    (out / "report.md").write_text(header + "\n" + report.to_markdown())
    metrics = {k: v for k, v in asdict(report).items() if k != "rows"}
    (out / "report.json").write_text(json.dumps({**provenance, **metrics, "rows": rows}, indent=1, ensure_ascii=False))
    write_miss_gallery(report, out / "misses.html")
    print(report.to_markdown())
    print(f"Wrote {out}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--set", required=True, choices=SETS)
    parser.add_argument("--distractors", type=int, default=50_000)
    parser.add_argument("--views", type=int, choices=(1, 3), default=3)
    parser.add_argument("--px", type=int, help="input size, default the model's own")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, help="gallery embedding cache, default EMB_DIR/eval")
    parser.add_argument("--dry-run", action="store_true", help="print the eval set and gallery sizes, embed nothing")
    args = parser.parse_args(argv)

    if args.dry_run:
        es = EvalSet.prepare(args.set, args.distractors, args.seed)
        print(es.describe())
        print(f"config hash: {config_hash(es.set_dir, args.distractors, args.views, args.px, args.seed)}"
              f"  gallery hash: {gallery_hash(es.gallery, es.excluded)}")
        return
    run_eval(args.model, args.set, args.distractors, args.views, args.px, args.seed, args.work_dir or config.EMB_DIR / "eval")


if __name__ == "__main__":
    main()
