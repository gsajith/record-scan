import json

import pytest

from pipeline.gold.labels import Label, append_label, fnv1a32, load_labels, split_for

# Shared with annotator/lib/labels.test.ts: both implementations must agree on these.
SPLIT_VECTORS = [
    ("86f7e437faa5a7fce15d1ddcb9eaeaea377667b8", "eval"),
    ("e9d71f5ee7c92d6dc9e92ffdad17b8bd49418f98", "sim"),
    ("84a516841ba77a5b4648de2cd0dfcb30ea46dbb4", "sim"),
    ("4a0a19218e082a343a1b17e5333409af9d98f0f5", "eval"),
    ("54fd1711209fb1c0781092374132c66e79e2241b", "eval"),
]
MBID = "b84ee12a-09ef-421b-82de-0441a926375b"
# Published FNV-1a 32-bit test values (the FNV reference vectors), so the hash is checked against
# an outside source; the split is the parity of the hash, even -> sim. Same literals in labels.test.ts.
FNV_VECTORS = [("", 0x811C9DC5, "eval"), ("a", 0xE40C292C, "sim"), ("foobar", 0xBF9CF968, "sim")]


def make_label(sha1: str, set_name: str = "wild", group: str | None = None, **overrides) -> Label:
    fields = dict(
        file=f"photos/{sha1}.jpg", sha1=sha1, kind="positive", mbid=MBID, tags=["glare"],
        source_url="https://www.flickr.com/photos/x/1", license="cc-by", group=group,
        split=split_for(set_name, sha1, group), notes="", labeled_at="2026-10-25T04:00:00Z",
    )
    return Label(**(fields | overrides))


@pytest.mark.parametrize("sha1,expected", SPLIT_VECTORS)
def test_split_vectors(sha1, expected):
    assert split_for("wild", sha1) == expected


@pytest.mark.parametrize("group,hash_value,expected", FNV_VECTORS)
def test_group_split_follows_the_fnv_hash(group, hash_value, expected):
    assert fnv1a32(group) == hash_value
    # Every photo of a group lands in the group's half, whatever its own hash says
    assert {split_for("wild", sha1, group) for sha1, _ in SPLIT_VECTORS} == {expected}


def test_load_checks_the_split_against_the_group(tmp_path):
    set_dir = tmp_path / "wild"
    sha1 = next(s for s, split in SPLIT_VECTORS if split == "eval")
    append_label(set_dir, make_label(sha1, group="a"))  # "a" puts the whole group in sim
    assert load_labels(set_dir)[0].split == "sim"
    with pytest.raises(ValueError, match="split"):
        append_label(set_dir, make_label(sha1, group="a", split="eval"))


def test_gold_is_always_eval():
    assert {split_for("gold", sha1) for sha1, _ in SPLIT_VECTORS} == {"eval"}


def test_load_keeps_latest_label(tmp_path):
    set_dir = tmp_path / "wild"
    sha1 = SPLIT_VECTORS[1][0]
    append_label(set_dir, make_label(sha1, tags=["glare"]))
    append_label(set_dir, make_label(sha1, tags=["crate", "dark"]))
    labels = load_labels(set_dir)
    assert len(labels) == 1
    assert labels[0].tags == ["crate", "dark"]
    assert labels[0].file == str(set_dir / "photos" / f"{sha1}.jpg")


def test_load_rejects_unknown_tag(tmp_path):
    set_dir = tmp_path / "wild"
    set_dir.mkdir()
    record = make_label(SPLIT_VECTORS[0][0]).__dict__ | {"tags": ["sparkles"]}
    (set_dir / "labels.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="line 1.*sparkles"):
        load_labels(set_dir)


def test_load_rejects_tampered_split(tmp_path):
    set_dir = tmp_path / "wild"
    set_dir.mkdir()
    sha1, real = SPLIT_VECTORS[0]
    record = make_label(sha1).__dict__ | {"split": "sim" if real == "eval" else "eval"}
    (set_dir / "labels.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="split"):
        load_labels(set_dir)


def test_positive_needs_mbid(tmp_path):
    with pytest.raises(ValueError, match="mbid"):
        append_label(tmp_path / "wild", make_label(SPLIT_VECTORS[0][0], mbid=None))


def test_wild_photo_without_source_is_fine(tmp_path):
    # D9 as revised 2026-10-26: the labeler records a source only when the browser gives one
    set_dir = tmp_path / "wild"
    append_label(set_dir, make_label(SPLIT_VECTORS[0][0], source_url=None, license="other-permitted"))
    assert load_labels(set_dir)[0].source_url is None


def test_gold_negative_without_source_is_fine(tmp_path):
    set_dir = tmp_path / "gold"
    sha1 = SPLIT_VECTORS[0][0]
    append_label(set_dir, make_label(sha1, "gold", kind="negative", mbid=None, source_url=None, license="own"))
    assert load_labels(set_dir)[0].kind == "negative"
