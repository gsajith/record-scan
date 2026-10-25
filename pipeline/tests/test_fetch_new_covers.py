import json
import shutil

import pytest
import requests

from pipeline.metadata.catalog import Cover
from pipeline.metadata.fetch_new_covers import Plan, best_front, covers_after, get_with_backoff, plan_covers, run
from pipeline.tests.test_catalog import write_jsonl


def image(id, front=False, types=(), approved=True, thumbs=("250", "500", "1200")):
    return {"id": id, "front": front, "types": list(types), "approved": approved,
            "thumbnails": {t: f"https://example.test/{id}-{t}.jpg" for t in thumbs}}


def test_best_front_prefers_approved_front():
    images = [image(1, types=["Back"]), image(2, front=True, approved=False), image(3, front=True)]
    assert best_front({"images": images}) == "3"


def test_best_front_falls_back_to_pending_front_then_any_approved():
    assert best_front({"images": [image(1, types=["Back"]), image(2, front=True, approved=False)]}) == "2"
    assert best_front({"images": [image(1, types=["Back"]), image(2, approved=False)]}) == "1"
    assert best_front({"images": [image(2, approved=False)]}) is None


def test_best_front_counts_a_front_type_and_needs_a_500px_thumbnail():
    assert best_front({"images": [image(1, front=True, thumbs=("250",)), image(2, types=["Front"])]}) == "2"
    # Older CAA records name the 500 px thumbnail "large"
    assert best_front({"images": [image(1, front=True, thumbs=("small", "large"))]}) == "1"
    assert best_front({"images": []}) is None


def mbid(n: int) -> str:
    return f"{n:08d}-0000-4000-8000-000000000000"


def old(n: int, cover_id: str, primary=True, file=None) -> Cover:
    return Cover(mbid(n), cover_id, file or f"{mbid(n)[:2]}/{mbid(n)}.jpg", primary)


def test_plan_covers_keeps_old_covers_adds_new_fronts_and_drops_deleted():
    previous = {
        mbid(1): [old(1, "100")],                     # still upstream, gained a new front
        mbid(2): [old(2, "200")],                     # deleted upstream, release still in MusicBrainz
        mbid(4): [old(4, "400")],                     # release gone from the new dump
        mbid(5): [old(5, "500")],                     # unchanged
        mbid(6): [old(6, "600"), old(6, "601", primary=False, file="00/x-601.jpg")],  # primary deleted
        mbid(7): [old(7, "700")],                     # 2025 fell back to a back cover, now has a front
    }
    releases = {mbid(1), mbid(2), mbid(3), mbid(5), mbid(6), mbid(7)}  # 3 is new to us
    cover_release = {100: 11, 400: 44, 500: 55, 601: 66, 700: 77}  # internal release ids, images still upstream
    fronts = {100, 400, 500, 601}                                  # which of our images are typed Front
    new_fronts = {11: [101, 100], 55: [], 77: [701], 99: [999]}    # fronts uploaded since the last dump

    plan = plan_covers(previous, releases, cover_release, new_fronts, fronts)

    assert plan.keep == [old(1, "100"), old(5, "500"), old(6, "601", file="00/x-601.jpg"),
                         old(7, "700", primary=False)]
    assert plan.download == [Cover(mbid(1), "101", f"00/{mbid(1)}-101.jpg", primary=False),
                             Cover(mbid(7), "701", f"00/{mbid(7)}-701.jpg", primary=True)]
    assert sorted(plan.lookup) == [mbid(2), mbid(3)]
    assert plan.dropped == [old(2, "200"), old(4, "400"), old(6, "600")]


class FakeResponse:
    def __init__(self, status: int, body: object = None):
        self.status_code = status
        self.body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self.body

    @property
    def content(self) -> bytes:
        assert isinstance(self.body, bytes)
        return self.body


class ScriptedSession:
    """Answers each GET with the next scripted response for that URL."""

    def __init__(self, script: dict[str, list]):
        self.script = script
        self.calls: list[str] = []

    def get(self, url, timeout):
        self.calls.append(url)
        answer = self.script[url].pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_get_backs_off_on_503_and_dropped_connections():
    session = ScriptedSession({"u": [FakeResponse(503), requests.ConnectionError(), FakeResponse(429),
                                     requests.exceptions.ChunkedEncodingError(), FakeResponse(503),
                                     FakeResponse(200, "ok")]})
    waits: list[float] = []
    assert get_with_backoff(session, "u", sleep=waits.append).json() == "ok"
    assert waits == [5, 10, 20, 40, 60]


def test_get_returns_none_on_404_and_gives_up_eventually():
    waits: list[float] = []
    assert get_with_backoff(ScriptedSession({"u": [FakeResponse(404)]}), "u", sleep=waits.append) is None
    assert waits == []
    with pytest.raises(requests.HTTPError):
        get_with_backoff(ScriptedSession({"u": [FakeResponse(503)] * 3}), "u", sleep=waits.append, tries=3)
    assert waits == [5, 10]


JPEG = b"\xff\xd8\xff\xe0 fake jpeg"
DATE, SINCE = "2026-10-24", "2025-08-13"


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines))


def caa_json(m: int) -> str:
    return f"https://coverartarchive.org/release/{mbid(m)}"


def caa_image(m: int, cover_id: int) -> str:
    return f"https://coverartarchive.org/release/{mbid(m)}/{cover_id}-500.jpg"


@pytest.fixture
def world(tmp_path, reload_config):
    """2025 covers (1 and 2), a newer dump (1, 2 and new 3), the cover dump, and the drive."""
    cfg = reload_config(RS_DATA_DIR=tmp_path / "data", RS_IMAGE_ROOT=tmp_path / "drive")
    (cfg.IMAGE_ROOT / "00").mkdir(parents=True)
    for m in (1, 2):
        (cfg.IMAGE_ROOT / "00" / f"{mbid(m)}.jpg").write_bytes(b"2025 cover")
    cfg.META_DIR.mkdir(parents=True)
    write_jsonl(cfg.META_DIR / f"covers-{SINCE}.jsonl", [  # what the previous update left
        {"mbid": mbid(1), "cover_id": "100", "file": f"00/{mbid(1)}.jpg", "primary": True},
        {"mbid": mbid(2), "cover_id": "200", "file": f"00/{mbid(2)}.jpg", "primary": True},
    ])
    write_jsonl(cfg.META_DIR / f"releases-{DATE}.jsonl",
                [{"id": mbid(m), "title": f"t{m}", "release_group": f"rg{m}"} for m in (1, 2, 3)])
    mbdump = cfg.DATA_DIR / f"musicbrainz-dump-{DATE}" / "mbdump"
    # id, release, comment, edit, ordering, date_uploaded, edits_pending, mime_type, ...
    write_lines(mbdump / "cover_art_archive.cover_art", [
        "100\t11\t\t1\t1\t2012-04-30 22:00:14+00\t0\timage/jpeg\t\\N",
        "101\t11\t\t2\t2\t2026-01-02 10:00:00+00\t0\timage/jpeg\t\\N",   # new front on release 1
        "102\t11\t\t3\t3\t2026-01-03 10:00:00+00\t0\timage/jpeg\t\\N",   # new back, ignored
        "103\t11\t\t4\t4\t2026-01-04 10:00:00+00\t0\tapplication/pdf\t\\N",  # PDF front, ignored
        "300\t33\t\t5\t1\t2026-02-01 10:00:00+00\t0\timage/jpeg\t\\N",   # release 3, new to us
    ])  # 200 is gone: deleted upstream
    write_lines(mbdump / "cover_art_archive.cover_art_type", ["100\t1", "101\t1", "102\t2", "103\t1", "300\t1"])
    return cfg


def read_manifest(cfg):
    return [Cover(**json.loads(line)) for line in (cfg.META_DIR / f"covers-{DATE}.jsonl").read_text().splitlines()]


def test_run_resumes_without_repeating_requests_and_never_overwrites(world):
    drive = world.IMAGE_ROOT / "00"
    first = ScriptedSession({
        caa_image(1, 101): [FakeResponse(200, JPEG)],
        caa_json(2): [FakeResponse(200, {"images": [image(201, front=True)]})],
        caa_image(2, 201): [FakeResponse(200, JPEG)],
    })
    assert run(DATE, SINCE, limit=2, session=first, sleep=lambda s: None).failed == 0
    assert sorted(first.calls) == sorted(first.script)  # the extra front first, then lookups in mbid order

    kept_1 = Cover(mbid(1), "100", f"00/{mbid(1)}.jpg", True)
    extra_1 = Cover(mbid(1), "101", f"00/{mbid(1)}-101.jpg", False)
    new_2 = Cover(mbid(2), "201", f"00/{mbid(2)}-201.jpg", True)
    assert read_manifest(world) == [kept_1, extra_1, new_2]
    assert (drive / f"{mbid(1)}-101.jpg").read_bytes() == JPEG
    # 2025 files stay as they were, even the one deleted upstream
    assert (drive / f"{mbid(1)}.jpg").read_bytes() == b"2025 cover"
    assert (drive / f"{mbid(2)}.jpg").read_bytes() == b"2025 cover"

    # The rerun only does what is left: release 3, whose lookup finds no usable image
    second = ScriptedSession({caa_json(3): [FakeResponse(200, {"images": [image(300, types=["Back"], approved=False)]})]})
    counts = run(DATE, SINCE, session=second, sleep=lambda s: None)
    assert (counts.pending, counts.no_cover, counts.failed) == (1, 1, 0)
    assert second.calls == [caa_json(3)]
    assert read_manifest(world) == [kept_1, extra_1, new_2]

    # Nothing left: no requests at all
    idle = ScriptedSession({})
    assert run(DATE, SINCE, session=idle, sleep=lambda s: None).pending == 0
    assert idle.calls == []


def test_run_reruns_work_cut_short_by_a_crash(world):
    log = world.META_DIR / f"fetched-{DATE}.jsonl"
    log.write_text(json.dumps({"mbid": mbid(3), "cover_id": None}) + '\n\n{"mbid": "' + mbid(2))
    session = ScriptedSession({
        caa_image(1, 101): [FakeResponse(200, JPEG)],
        caa_json(2): [FakeResponse(200, {"images": [image(201, front=True)]})],
        caa_image(2, 201): [FakeResponse(200, JPEG)],
    })
    assert run(DATE, SINCE, session=session, sleep=lambda s: None).failed == 0
    assert caa_json(3) not in session.calls  # its finished lookup is reused
    # The cut-short line no longer swallows the records after it
    records = [json.loads(line) for line in log.read_text().splitlines()[3:]]
    assert {"mbid": mbid(2), "cover_id": "201", "file": f"00/{mbid(2)}-201.jpg", "primary": True} in records


def test_run_stops_when_the_drive_goes_away(world):
    def unplug_then_answer(url, timeout):
        shutil.rmtree(world.IMAGE_ROOT)  # the drive unmounts mid-run
        raise requests.ConnectionError()

    session = ScriptedSession({})
    session.get = unplug_then_answer
    with pytest.raises(SystemExit, match="Remount"):
        run(DATE, SINCE, session=session, workers=1, sleep=lambda s: None)


def test_dry_run_counts_the_backlog_without_requests(world, capsys):
    counts = run(DATE, SINCE, dry_run=True, session=ScriptedSession({}))
    assert (counts.keep, counts.download, counts.lookup, counts.dropped, counts.pending) == (1, 1, 2, 1, 3)
    assert "3 to fetch" in capsys.readouterr().out
    assert not (world.META_DIR / f"covers-{DATE}.jsonl").exists()


def test_covers_after_keeps_one_primary_and_earlier_extras():
    back = old(7, "700", primary=False)
    front = Cover(mbid(7), "701", f"00/{mbid(7)}-701.jpg", primary=True)
    earlier = Cover(mbid(7), "702", f"00/{mbid(7)}-702.jpg", primary=True)  # landed by an earlier plan
    plan = Plan(keep=[back], download=[front])
    # The new front failed to download: the back stays primary, the earlier image joins as an extra
    assert covers_after(plan, {earlier.file: earlier}, {mbid(7)}) == [
        old(7, "700"), Cover(mbid(7), "702", earlier.file, primary=False)]
    # Once it lands, the front is the primary
    assert covers_after(plan, {front.file: front}, {mbid(7)}) == [back, front]
