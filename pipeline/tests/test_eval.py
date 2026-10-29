import json

import pytest

from pipeline.gold.labels import Label, append_label, now_iso, split_for
from pipeline.metadata.build_catalog_db import build
from pipeline.metadata.catalog import Catalog
from pipeline.retrieval import eval as eval_module
from pipeline.retrieval.eval import (EvalReport, build_mini_gallery, gallery_hash, is_hit, load_eval_set, main,
                                     write_miss_gallery)

BEATLES = "b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d"
BJORK = "87c5dedd-371d-4a53-9f7f-80522fb7f3cb"
ED = "44444444-4444-4444-8444-444444444444"
ABBEY = "11111111-1111-4111-8111-111111111111"
ABBEY_REISSUE = "11111111-1111-4111-8111-222222222222"
LOOKALIKE = "11111111-1111-4111-8111-333333333333"
DEBUT = "22222222-2222-4222-8222-222222222222"
PLUS = "55555555-5555-4555-8555-555555555555"
DIVIDE = "66666666-6666-4666-8666-666666666666"
NOT_IN_CATALOG = "99999999-9999-4999-8999-999999999999"
ABBEY_2019 = "11111111-1111-4111-8111-444444444444"
WEEZER, BLUE_ALBUM, GREEN_ALBUM = ("77777777-7777-4777-8777-77777777777" + c for c in "012")
# Release groups: MusicBrainz's own "these releases are one album"
RG_ABBEY, RG_BLUE, RG_GREEN = ("88888888-8888-4888-8888-88888888888" + c for c in "012")

# sha1 values whose first 8 hex digits are odd go to the wild set's eval half, even to sim
EVAL_SHA = "00000001" + "a" * 32
SIM_SHA = "00000002" + "a" * 32


def make_catalog(tmp_path, releases, artists):
    """A catalog.sqlite with one primary cover per release."""
    covers = [
        {"mbid": r["id"], "cover_id": str(n), "file": f"{r['id'][:2]}/{r['id']}.jpg", "primary": True}
        for n, r in enumerate(releases)
    ]
    (tmp_path / "releases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in releases))
    (tmp_path / "artists.json").write_text(json.dumps(artists))
    (tmp_path / "covers.jsonl").write_text("".join(json.dumps(c) + "\n" for c in covers))
    build(tmp_path / "releases.jsonl", tmp_path / "artists.json", tmp_path / "covers.jsonl", tmp_path / "catalog.sqlite")
    return Catalog(tmp_path / "catalog.sqlite")


FILLER = [f"{n:08x}-0000-4000-8000-000000000000" for n in range(30)]


@pytest.fixture
def catalog(tmp_path):
    releases = [
        {"title": "Abbey Road", "id": ABBEY, "artist_ids": [BEATLES], "release_group": RG_ABBEY},
        {"title": "Abbey Road", "id": ABBEY_REISSUE, "artist_ids": [BEATLES], "release_group": RG_ABBEY},
        {"title": "Abbey Road (2019 Mix)", "id": ABBEY_2019, "artist_ids": [BEATLES], "release_group": RG_ABBEY},
        {"title": "Abbey Road Revisited", "id": LOOKALIKE, "artist_ids": [BEATLES]},
        {"title": "Weezer", "id": BLUE_ALBUM, "artist_ids": [WEEZER], "release_group": RG_BLUE},
        {"title": "Weezer", "id": GREEN_ALBUM, "artist_ids": [WEEZER], "release_group": RG_GREEN},
        {"title": "Debut", "id": DEBUT, "artist_ids": [BJORK]},
        {"title": "+", "id": PLUS, "artist_ids": [ED]},
        {"title": "÷", "id": DIVIDE, "artist_ids": [ED]},
        *({"title": f"Filler {n}", "id": m, "artist_ids": [BJORK]} for n, m in enumerate(FILLER)),
    ]
    artists = {BEATLES: {"name": "Beatles, The"}, BJORK: {"name": "Björk"}, ED: {"name": "Ed Sheeran"},
               WEEZER: {"name": "Weezer"}}
    return make_catalog(tmp_path, releases, artists)


def add_label(set_dir, sha1, kind, mbid=None, tags=()):
    wild = set_dir.name == "wild"
    append_label(set_dir, Label(
        file=f"photos/{sha1}.jpg", sha1=sha1, kind=kind, mbid=mbid, tags=list(tags),
        source_url="https://example.org/photo" if wild else None, license="cc0" if wild else "own", group=None,
        split=split_for(set_dir.name, sha1), notes="", labeled_at=now_iso(),
    ))


def sha(n):
    return f"{n:08x}" + "b" * 32


def test_load_eval_set_uses_eval_split_only(tmp_path, catalog):
    wild = tmp_path / "wild"
    add_label(wild, EVAL_SHA, "positive", ABBEY)
    add_label(wild, SIM_SHA, "positive", DEBUT)

    items = load_eval_set(wild, catalog, clusters=None)

    assert [item.label.sha1 for item in items] == [EVAL_SHA]


def test_is_hit_rules(tmp_path, catalog):
    gold = tmp_path / "gold"
    add_label(gold, sha(1), "positive", ABBEY)
    add_label(gold, sha(2), "positive", PLUS)
    add_label(gold, sha(3), "negative")
    add_label(gold, sha(4), "unknown")
    add_label(gold, sha(5), "positive", NOT_IN_CATALOG)
    add_label(gold, sha(6), "positive", BLUE_ALBUM)

    abbey, plus, negative, unknown, gone, blue = load_eval_set(
        gold, catalog, clusters={ABBEY: 7, LOOKALIKE: 7, DEBUT: 8}
    )

    # The labeled release, any release in its MusicBrainz release group (whatever its title),
    # and lookalikes in its duplicate cluster
    assert is_hit(ABBEY, abbey) and is_hit(ABBEY_REISSUE, abbey) and is_hit(ABBEY_2019, abbey)
    assert is_hit(LOOKALIKE, abbey)
    assert not is_hit(DEBUT, abbey)
    # Two albums with the same title and artist are still two albums
    assert is_hit(BLUE_ALBUM, blue) and not is_hit(GREEN_ALBUM, blue)
    # A release with no release group stands alone
    assert is_hit(PLUS, plus) and not is_hit(DIVIDE, plus)
    # Nothing is a hit for a photo with no answer in the catalog
    assert not is_hit(ABBEY, negative) and not is_hit(ABBEY, unknown)
    # A positive whose release left the catalog stays in the eval as a certain miss, not dropped
    assert gone.label.kind == "positive" and not is_hit(NOT_IN_CATALOG, gone)


def row(kind, top1, hit_rank=None, tags=(), file="/photos/x.jpg"):
    return {"file": file, "sha1": "s", "kind": kind, "mbid": ABBEY if kind == "positive" else None,
            "expected": "Beatles, The - Abbey Road" if kind == "positive" else "", "tags": list(tags),
            "matches": [{"mbid": DEBUT, "title": "<Debut>", "artists": ["Björk"], "score": top1}],
            "hit_rank": hit_rank}


def test_eval_report_metrics_and_slices():
    rows = [
        row("positive", 0.9, hit_rank=1, tags=["tilt"]),
        row("positive", 0.7, hit_rank=3, tags=["tilt", "glare"]),
        row("positive", 0.6, tags=["glare"]),
        row("positive", 0.5, hit_rank=1),
        row("negative", 0.8),
        row("unknown", 0.85),  # a record missing from the catalog: any answer is wrong, so it counts as a negative
        row("negative", 0.4),
    ]

    report = EvalReport.from_rows(rows)

    assert report.r1 == 0.5
    assert report.r5 == 0.75
    assert report.per_tag == {"tilt": (0.5, 2), "glare": (0.0, 2)}
    assert report.neg_max == 0.85
    assert report.pos_min_top1 == 0.5
    # Ranked by top-1 score, correct (positive, hit at rank 1) at ranks 1 and 6, over 4 positives
    assert report.micro_ap == pytest.approx((1 / 1 + 2 / 6) / 4)


def test_micro_ap_ranks_ties_against_the_model():
    # Equal top-1 scores: the wrong answer ranks first, so labeling order cannot raise the score
    report = EvalReport.from_rows([row("positive", 0.5, hit_rank=1), row("negative", 0.5)])

    assert report.micro_ap == 0.5


def test_mini_gallery_holds_every_target_cover_plus_seeded_distractors(tmp_path, catalog):
    gold = tmp_path / "gold"
    add_label(gold, sha(1), "positive", ABBEY)
    add_label(gold, sha(2), "negative")
    items = load_eval_set(gold, catalog, clusters=None)

    gallery = build_mini_gallery(items, catalog, distractors=10, seed=0, excluded=set())

    mbids = [c.mbid for c in gallery]
    assert mbids[:3] == sorted([ABBEY, ABBEY_REISSUE, ABBEY_2019])
    assert len(mbids) == 13 and len(set(mbids)) == 13
    assert not {ABBEY, ABBEY_REISSUE, ABBEY_2019} & set(mbids[3:])
    assert gallery == build_mini_gallery(items, catalog, distractors=10, seed=0, excluded=set())
    assert gallery != build_mini_gallery(items, catalog, distractors=10, seed=1, excluded=set())


def stem(mbid):
    return mbid  # make_catalog files each release's one cover as <mbid>.jpg


def test_mini_gallery_leaves_excluded_covers_out_of_targets_and_distractors(tmp_path, catalog):
    gold = tmp_path / "gold"
    add_label(gold, sha(1), "positive", ABBEY)
    items = load_eval_set(gold, catalog, clusters=None)
    plain = build_mini_gallery(items, catalog, distractors=10, seed=0, excluded=set())
    excluded = {stem(ABBEY_REISSUE), *(stem(c.mbid) for c in plain[3:8])}  # a target and five distractors

    gallery = build_mini_gallery(items, catalog, distractors=10, seed=0, excluded=excluded)

    mbids = [c.mbid for c in gallery]
    assert mbids[:2] == sorted([ABBEY, ABBEY_2019])
    assert len(mbids) == 12 and not set(mbids) & excluded
    assert mbids[2:7] == [c.mbid for c in plain[8:13]]  # the same draw, skipping the excluded


def test_the_gallery_hash_changes_with_the_exclusion_list(tmp_path, catalog):
    gold = tmp_path / "gold"
    add_label(gold, sha(1), "positive", ABBEY)
    gallery = build_mini_gallery(load_eval_set(gold, catalog, clusters=None), catalog, 10, 0, excluded=set())

    assert gallery_hash(gallery, {}) == gallery_hash(gallery, {})
    assert gallery_hash(gallery, {}) != gallery_hash(gallery, {stem(DEBUT): "blank"})


def test_miss_gallery_shows_misses_the_weakest_hits_and_the_strongest_negatives(tmp_path):
    rows = [row("positive", 0.6, file="/photos/missed.jpg"), row("negative", 0.9, file="/photos/wall.jpg")]
    rows += [row("positive", 0.5 + n / 100, hit_rank=1, file=f"/photos/hit{n}.jpg") for n in range(25)]

    write_miss_gallery(EvalReport.from_rows(rows), tmp_path / "misses.html")

    html = (tmp_path / "misses.html").read_text()
    assert "file:///photos/missed.jpg" in html
    assert f"https://coverartarchive.org/release/{DEBUT}/front-250" in html
    assert f"https://coverartarchive.org/release/{ABBEY}/front-250" in html  # the expected cover
    assert "&lt;Debut&gt;" in html and "<Debut>" not in html
    # The twenty lowest-scoring hits, so hit0 to hit19 but not hit20 and up
    assert "/photos/hit19.jpg" in html and "/photos/hit20.jpg" not in html
    # The negative that sets neg_max is shown in its own section, after the misses
    assert html.index("/photos/wall.jpg") > html.index("Highest-scoring negatives") > html.index("/photos/missed.jpg")


def test_dry_run_on_a_temp_set(tmp_path, reload_config, capsys, monkeypatch):
    data = tmp_path / "data"
    (data / "metadata").mkdir(parents=True)
    make_catalog(data / "metadata", [
        {"title": "Abbey Road", "id": ABBEY, "artist_ids": [BEATLES]},
        {"title": "Debut", "id": DEBUT, "artist_ids": [BJORK]},
        *({"title": f"Filler {n}", "id": m, "artist_ids": [BEATLES]} for n, m in enumerate(FILLER)),
    ], {BEATLES: {"name": "Beatles, The"}, BJORK: {"name": "Björk"}})
    reload_config(RS_DATA_DIR=data, RS_IMAGE_ROOT=tmp_path / "no-drive")
    add_label(data / "gold", sha(1), "positive", ABBEY, tags=["glare"])
    add_label(data / "gold", sha(2), "negative")
    add_label(data / "gold", sha(3), "positive", DEBUT)  # its one cover is excluded, so it can't be a hit
    monkeypatch.setattr(eval_module, "load_exclusions", lambda gallery_dir: {stem(DEBUT): "blank"})

    main(["--model", "clip-vit-b-32", "--set", "gold", "--distractors", "5", "--dry-run"])

    out = capsys.readouterr().out
    assert "3 eval photos: 2 positive, 1 negative, 0 unknown" in out
    assert "0 positives no longer in the catalog" in out
    assert "1 positives with every cover excluded" in out
    assert "gallery: 1 target covers + 5 distractors" in out
    assert "config hash:" in out
