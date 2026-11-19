"""The review page's script (`scripts/review_scenes.py`, run by hand): it runs TagScenes on the stand-in tagger and
writes the result and a self-contained page; `--set` reaches a setting or a constant, and puts constants back; a saved
review is compared with the new run."""

import importlib
import json
import re
import shutil
import subprocess
import sys

import pytest

from image_tags import frames as framing
from image_tags import scenes
from tests.conftest import PYTHON
from tests.test_classify_frames import SHEETS, thumbnail
from tests.test_tag_scenes import EVERY, SCENE_IMAGES

sys.path.insert(0, str(PYTHON / "scripts"))
review_scenes = importlib.import_module("review_scenes")


def planted(tmp_path, seconds=30):
    """A folder of VideoThumbnail thumbnails: each scene image a frame every `EVERY` s for `seconds`."""
    folder = tmp_path / "ep01.1c8fcf3f1311"
    folder.mkdir()
    t = 0.0
    for image in SCENE_IMAGES:
        for _ in range(round(seconds / EVERY)):
            thumbnail(folder, f"{len(list(folder.iterdir())):04}.jpg", t, image=image.name)
            t += EVERY
    return folder


def review(tmp_path, data_dir, *inputs, sets=()):
    out = tmp_path / "review"
    argv = [*map(str, inputs), "--tagger", "stand-in", "--out", str(out), "--data-dir", str(data_dir)]
    for item in sets:
        argv += ["--set", item]
    return out, review_scenes.main(argv)


def embedded(page: str) -> dict:
    found = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S)
    return json.loads(found.group(1))


def script_of(page: str) -> str:
    return re.findall(r"<script>(.*?)</script>", page, re.S)[-1]


def test_a_folder_gives_its_result_and_a_page_of_one_card_per_scene(tagger_hub, data_dir, tmp_path, capsys):
    folder = planted(tmp_path)

    out, written = review(tmp_path, data_dir, folder)

    # Named after the thumbnails' video.
    assert written == [out / "holiday.result.json", out / "holiday.review.html"]
    result = json.loads(written[0].read_text(encoding="utf-8"))
    assert result["count"] == 3
    page = written[1].read_text(encoding="utf-8")
    data = embedded(page)
    assert len(data["scenes"]) == 3
    assert data["boundaries"] == [scene["start"] for scene in result["scenes"][1:]]
    assert [frame["t"] for frame in data["frames"]] == [EVERY * i for i in range(45)]
    assert all(frame["src"].startswith("file:///") for frame in data["frames"])
    assert data["earlier"] is None
    # Self-contained: nothing fetched from anywhere.
    assert not re.search(r"https?:|//cdn|@import|<link", page)
    # Times, counts and paths only: no tag of the result is printed.
    printed = capsys.readouterr().out
    names = {tag["name"] for scene in result["scenes"] for tag in scene["tags"] + scene["highlights"]}
    assert names and not any(name in printed for name in names)
    assert "3 scenes" in printed


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on the PATH")
def test_the_page_s_script_parses(tagger_hub, data_dir, tmp_path):
    _, written = review(tmp_path, data_dir, planted(tmp_path))
    script = tmp_path / "page.js"
    script.write_text(script_of(written[1].read_text(encoding="utf-8")), encoding="utf-8")

    subprocess.run(["node", "--check", str(script)], check=True)


def test_a_sheet_s_frames_are_cropped_into_the_page(tagger_hub, data_dir, tmp_path):
    _, written = review(tmp_path, data_dir, SHEETS / "sample.jpg")

    data = embedded(written[1].read_text(encoding="utf-8"))
    assert data["frames"] and all(frame["src"].startswith("data:image/jpeg;base64,") for frame in data["frames"])


def test_set_parses_settings_and_constants_in_their_types():
    settings, constants = review_scenes.parse_sets(
        ["minSceneDuration=20", 'ignoreTags=["a"]', "LINK_WINDOW=40", 'SHOT_SPECIFIC_TAGS=["solo"]', "BLANK_STD=6"]
    )

    assert settings == {"minSceneDuration": 20, "ignoreTags": ["a"]}
    assert constants == {"LINK_WINDOW": 40.0, "SHOT_SPECIFIC_TAGS": frozenset({"solo"}), "BLANK_STD": 6}
    assert isinstance(constants["LINK_WINDOW"], float)


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ("minSceneDuration", r"is not NAME=VALUE"),
        ("minSceneLength=3", r"^No setting `minSceneLength`"),
        ("LINK_WINDOWS=3", r"^No constant `LINK_WINDOWS`"),
        ("LINK_WINDOW=far", r"^`LINK_WINDOW` is a number"),
        ("SHOT_SPECIFIC_TAGS=solo", r"^`SHOT_SPECIFIC_TAGS` is a list"),
    ],
)
def test_set_refuses_what_it_does_not_know(item, message):
    with pytest.raises(ValueError, match=message):
        review_scenes.parse_sets([item])


def test_constants_reach_the_run_and_are_put_back_after_it(tagger_hub, data_dir, tmp_path):
    window, blank = scenes.LINK_WINDOW, framing.BLANK_STD

    _, written = review(tmp_path, data_dir, planted(tmp_path), sets=["LINK_WINDOW=12", "minSceneDuration=60"])

    data = embedded(written[1].read_text(encoding="utf-8"))
    assert data["settings"]["constants"]["LINK_WINDOW"] == 12.0
    assert data["settings"]["minSceneDuration"] == 60
    assert data["settings"]["tagger"] == "stand-in"
    assert len(data["scenes"]) == 1  # every scene is shorter than a minute
    assert (scenes.LINK_WINDOW, framing.BLANK_STD) == (window, blank)


def test_overridden_puts_constants_back_after_a_failure():
    window = scenes.LINK_WINDOW

    with pytest.raises(RuntimeError), review_scenes.overridden({"LINK_WINDOW": 1.0}):
        assert scenes.LINK_WINDOW == 1.0
        raise RuntimeError

    assert scenes.LINK_WINDOW == window


# Since the last review.


def result_of(*spans, tags=(), highlights=(), rating="general", peaks=()):
    """A result whose scenes run over `spans`, each listing `tags`, `highlights` and the rating with `peaks`."""
    return {
        "scenes": [
            {
                "start": start,
                "end": end,
                "tags": [{"name": name} for name in tags],
                "highlights": [{"name": name} for name in highlights],
                "rating": {"label": rating, "peaks": [{"label": label} for label in peaks]},
            }
            for start, end in spans
        ]
    }


def statuses(found):
    return [(comment["kind"], comment["status"]) for comment in found["comments"]]


def test_boundaries_within_a_second_are_the_same_and_others_changed():
    found = review_scenes.since_last({"boundaries": [30.0, 60.0]}, result_of((0, 30.8), (30.8, 75), (75, 90)))

    assert found["added"] == [75]
    assert found["gone"] == [60.0]


def test_a_merge_is_addressed_once_its_boundary_is_gone_and_a_cut_once_one_is_there():
    saved = {"boundaries": [30.0], "merges": [30.0], "cuts": [50.0]}

    before = review_scenes.since_last(saved, result_of((0, 30), (30, 90)))
    after = review_scenes.since_last(saved, result_of((0, 50.6), (50.6, 90)))

    assert statuses(before) == [("merge", "open"), ("cut", "open")]
    assert statuses(after) == [("merge", "addressed"), ("cut", "addressed")]


def test_a_wrong_chip_is_addressed_once_the_scene_at_its_time_no_longer_lists_it():
    saved = {
        "wrong": [
            {"tag": "rain", "kind": "tag", "start": 0, "end": 30},
            {"tag": "rain", "kind": "highlight", "start": 30, "end": 90},
            {"tag": "sensitive", "kind": "rating", "start": 30, "end": 90},
        ],
        "notes": [{"start": 0, "end": 30, "text": "a missing tag"}],
    }

    still = review_scenes.since_last(saved, result_of((0, 90), tags=["rain"], highlights=["rain"], peaks=["sensitive"]))
    gone = review_scenes.since_last(saved, result_of((0, 90), highlights=["rain"]))

    assert statuses(still) == [("wrong", "open"), ("wrong", "open"), ("wrong", "open"), ("note", "note")]
    assert statuses(gone) == [("wrong", "addressed"), ("wrong", "open"), ("wrong", "addressed"), ("note", "note")]
    assert [comment["what"] for comment in gone["comments"][:3]] == ["tag", "highlight", "rating"]
    # A review saved without boundaries cannot say which changed.
    assert (gone["added"], gone["gone"]) == (None, None)


def test_a_saved_review_beside_the_page_is_compared_with_the_new_run(tagger_hub, data_dir, tmp_path, capsys):
    folder = planted(tmp_path)
    out, written = review(tmp_path, data_dir, folder)
    first = json.loads(written[0].read_text(encoding="utf-8"))
    boundary = first["scenes"][1]["start"]
    saved = {"video": first["video"], "settings": {}, "boundaries": [boundary, 500.0], "merges": [boundary]}
    (out / "holiday.review.json").write_text(json.dumps(saved), encoding="utf-8")

    _, written = review(tmp_path, data_dir, folder)

    earlier = embedded(written[1].read_text(encoding="utf-8"))["earlier"]
    assert earlier["gone"] == [500.0]
    assert len(earlier["added"]) == 1  # the second boundary, not in the saved review
    assert statuses(earlier) == [("merge", "open")]
    assert "2 boundaries changed; 0 comments addressed, 1 still open" in capsys.readouterr().out
