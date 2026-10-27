"""Label records for the gold and wild photo sets, one JSON line per label in <set>/labels.jsonl.

The annotator app writes these lines (annotator/lib/labels.ts mirrors this file) and the eval
reads them. A re-label appends a new line; the latest line for a photo wins.
"""

import json
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from pipeline import config

KINDS = ("positive", "negative", "unknown")  # unknown: a record, but not in the catalog
# partial: any cut-off; crate: the bottom hidden by the sleeve in front, the crate-digging case
TAGS = ("tilt", "glare", "dark", "colored-light", "hand", "shrink-wrap", "text-only", "rack",
        "blur", "partial", "crate", "flash")
LICENSES = ("own", "cc0", "cc-by", "cc-by-sa", "pexels", "unsplash", "other-permitted")
SETS = ("gold", "wild")


@dataclass
class Label:
    file: str  # relative to the set dir on disk; absolute after load_labels
    sha1: str
    kind: str
    mbid: str | None
    tags: list[str]
    source_url: str | None
    license: str
    split: str  # "eval" or "sim", always split_for(set, sha1)
    notes: str
    labeled_at: str  # ISO 8601 UTC


def split_for(set_name: str, sha1: str) -> str:
    """Which half a photo belongs to, fixed by its hash. Gold photos are never used for tuning."""
    if set_name == "gold":
        return "eval"
    return "sim" if int(sha1[:8], 16) % 2 == 0 else "eval"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _set_name(set_dir: Path) -> str:
    # The sealed holdout reuses the same layout under another root, so the folder name decides
    if set_dir.name not in SETS:
        raise ValueError(f"{set_dir} is not a gold or wild set folder")
    return set_dir.name


def _validate(label: Label, set_name: str) -> None:
    if label.kind not in KINDS:
        raise ValueError(f"unknown kind {label.kind!r}")
    if bad := [t for t in label.tags if t not in TAGS]:
        raise ValueError(f"unknown tags {bad}")
    if label.license not in LICENSES:
        raise ValueError(f"unknown license {label.license!r}")
    if label.split != split_for(set_name, label.sha1):
        raise ValueError(f"split {label.split!r} does not match the hash rule for {label.sha1}")
    if label.kind == "positive" and not (label.mbid and config.is_uuid(label.mbid)):
        raise ValueError("a positive label needs a valid mbid")
    if label.kind != "positive" and label.mbid is not None:
        raise ValueError(f"a {label.kind} label must not carry an mbid")


def append_label(set_dir: Path, label: Label) -> None:
    """Validate and append one line. Absolute paths inside set_dir are stored relative."""
    _validate(label, _set_name(set_dir))
    record = asdict(label)
    path = Path(record["file"])
    if path.is_absolute():
        record["file"] = str(path.relative_to(set_dir))
    set_dir.mkdir(parents=True, exist_ok=True)
    with (set_dir / "labels.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_labels(set_dir: Path) -> list[Label]:
    """Latest label per photo, in first-seen order, with file paths made absolute."""
    set_name = _set_name(set_dir)
    path = set_dir / "labels.jsonl"
    if not path.exists():
        return []
    names = {f.name for f in fields(Label)}
    latest: dict[str, Label] = {}
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if set(record) != names:
                raise ValueError(f"keys {sorted(set(record) ^ names)} do not match the Label fields")
            label = Label(**record)
            _validate(label, set_name)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{path} line {n}: {e}") from e
        label.file = str(set_dir / label.file)
        latest[label.sha1] = label
    return list(latest.values())
