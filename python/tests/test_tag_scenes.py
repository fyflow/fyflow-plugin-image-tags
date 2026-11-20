"""`TagScenes` as the Host runs it: a video's frames, each with a timestamp, tagged and embedded, then grouped into
scenes with their tags, highlights, rating and keyframe, and the video's summary.

The Worker runs the stand-in tagger (`stand_in.py`), whose scores and embedding come from an image's mean colour: so
a "scene" here is one image, a frame every 2 s. `test_tag_scenes_planted.py` runs the real tagger on a planted video.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from image_tags import frames as framing
from image_tags.frames import BLANK_STD, is_blank, read_frame_list, read_frames
from image_tags.catalogue import TAGGERS
from image_tags.worker import TagFrames, TagScenes, keyframe_tags_left_out
from tests.conftest import IMAGES, Context, Task, run
from tests.test_classify_frames import SHEETS, frames_folder, thumbnail
from tests.test_tag_frames import cpu_warning

#: Three "scenes", each one image: their mean colours, and so the stand-in's embeddings and tags, differ.
SCENE_IMAGES = [
    IMAGES / "anime-01-rainy-bus-stop.jpg",
    IMAGES / "cg-02-sintel.jpg",
    IMAGES / "live_action-03-redwood-creek.jpg",
]
EVERY = 2.0


def black(tmp_path) -> Path:
    path = tmp_path / "black.png"
    if not path.exists():
        Image.new("RGB", (64, 36)).save(path)
    return path


def video(tmp_path, seconds=(30, 30, 30), blanks=1) -> tuple[list[dict], list[float]]:
    """A `frames` list of a planted video: each scene's image a frame every `EVERY` s for `seconds`, with `blanks`
    black frames between the first scene and the second; and the times the planted scenes start, after the first."""
    frames, starts, t = [], [], 0.0
    for k, (image, length) in enumerate(zip(SCENE_IMAGES, seconds, strict=True)):
        if k:
            starts.append(t - EVERY / 2)
        for _ in range(round(length / EVERY)):
            frames.append({"path": str(image), "timestamp": t})
            t += EVERY
        if k == 0:
            for _ in range(blanks):
                frames.append({"path": str(black(tmp_path)), "timestamp": t})
                t += EVERY
    return frames, starts


def tag_scenes(config, input, ctx=None, worker=None, **settings):
    return run(worker or TagScenes(config), Task(input, {"tagger": "stand-in", **settings}), ctx)


# What it hands on.


def test_a_video_s_frames_are_grouped_into_its_planted_scenes(tagger_hub, config, tmp_path):
    frames, starts = video(tmp_path)

    result = tag_scenes(config, {"frames": frames})

    assert list(result) == [
        "tagger",
        "frameCount",
        "duration",
        "threshold",
        "summary",
        "count",
        "scenes",
        "warnings",
    ]
    assert result["tagger"] == "stand-in"
    assert result["frameCount"] == len(frames)
    assert result["count"] == len(result["scenes"]) == 3
    # The black frame between the first two scenes ends the first.
    assert [scene["start"] for scene in result["scenes"][1:]] == starts
    assert result["scenes"][0]["frames"] == list(range(16))
    assert sorted(i for scene in result["scenes"] for i in scene["frames"]) == list(range(len(frames)))
    # Without a duration, the last frame's span ends half a gap after it.
    assert result["duration"] == result["scenes"][-1]["end"] == frames[-1]["timestamp"] + EVERY / 2
    assert isinstance(result["threshold"], float)
    for k, scene in enumerate(result["scenes"]):
        assert scene["index"] == k
        assert ("boundary" in scene) == (k > 0)
        assert scene["keyframe"]["path"] == str(SCENE_IMAGES[k])
        assert scene["rating"]["label"] in scene["rating"]["scores"]
        assert list(scene["rating"]["scores"]) == ["general", "sensitive", "questionable", "explicit"]
        assert {"tags", "highlights"} <= set(scene)
    assert set(result["summary"]) == {"tags", "rating"}
    assert result["warnings"] == [cpu_warning()]
    # No frame's own tags anywhere.
    assert "frames" not in result


def test_include_frames_adds_each_frame_s_own_tags_rating_blank_and_scene(tagger_hub, config, tmp_path):
    frames, _ = video(tmp_path)

    result = tag_scenes(config, {"frames": frames}, includeFrames=True, maxTags=3)

    assert list(result)[-2:] == ["frames", "warnings"]
    alone = run(TagFrames(config), Task({"frames": frames}, {"tagger": "stand-in", "maxTags": 3}))
    for i, (frame, want) in enumerate(zip(result["frames"], alone["frames"], strict=True)):
        assert set(frame) == {"path", "timestamp", "tags", "rating", "blank", "scene"}
        assert (frame["path"], frame["timestamp"]) == (frames[i]["path"], frames[i]["timestamp"])
        assert [t["name"] for t in frame["tags"]] == [t["name"] for t in want["tags"]]
        assert frame["rating"]["scores"] == pytest.approx(want["rating"]["scores"], abs=1e-5)
        assert frame["blank"] is (frame["path"] == str(black(tmp_path)))
        assert i in result["scenes"][frame["scene"]]["frames"]
    assert sum(frame["blank"] for frame in result["frames"]) == 1


def test_include_tags_false_leaves_tags_and_highlights_out(tagger_hub, config, tmp_path):
    frames, _ = video(tmp_path)

    result = tag_scenes(config, {"frames": frames}, includeTags=False)

    assert result["count"] == 3
    for scene in result["scenes"]:
        assert not {"tags", "highlights"} & set(scene)
        assert {"start", "end", "frames", "keyframe", "rating"} <= set(scene)
    assert list(result["summary"]) == ["rating"]


def test_the_scenes_settings_reach_grouping(tagger_hub, config, tmp_path):
    frames, _ = video(tmp_path)

    # Every scene is shorter than a minute: they all merge into one.
    result = tag_scenes(config, {"frames": frames}, minSceneDuration=60)

    assert result["count"] == 1
    assert result["scenes"][0]["frames"] == list(range(len(frames)))


def test_keyframe_tags_the_tagger_lacks_or_ignore_tags_drops_are_left_out_with_a_warning(tagger_hub, config, tmp_path):
    frames, _ = video(tmp_path)

    result = tag_scenes(
        config,
        {"frames": frames},
        keyframeTags={"1girl": 1, "rating:explicit": -2, "no_such_tag": 1, "outdoors": 1},
        keyframeShare=1,
        ignoreTags=["outdoors"],
    )

    assert result["count"] == 3
    assert result["warnings"] == [cpu_warning(), keyframe_tags_left_out(TAGGERS["stand-in"], ["no_such_tag", "outdoors"])]
    # Each scene's frames are one image: every frame weighs the same, so each keyframe is still its scene's image.
    for k, scene in enumerate(result["scenes"]):
        assert scene["keyframe"]["path"] == str(SCENE_IMAGES[k])


def test_a_folder_of_thumbnails_gives_the_video_and_its_duration(tagger_hub, config, tmp_path):
    folder = tmp_path / "thumbnails"
    folder.mkdir()
    for i in range(4):
        thumbnail(folder, f"{i:03}.jpg", 100 + EVERY * i)

    result = tag_scenes(config, {"path": str(folder)})

    assert result["path"] == str(folder)
    assert result["video"] == "D:/Videos/holiday.mp4"
    assert result["duration"] == 1621.12  # the thumbnails' duration
    assert result["scenes"][-1]["end"] == 1621.12
    assert result["scenes"][0]["start"] == 0


# Timestamps are needed.


def test_a_frame_without_a_timestamp_fails_the_task_before_any_model_loads(tagger_hub, config, tmp_path):
    frames, _ = video(tmp_path)
    frames[3] = {"path": frames[3]["path"]}
    ctx = Context()

    with pytest.raises(
        ValueError, match=r"^1 of 46 frames has no timestamp, and TagScenes groups frames by their time"
    ):
        tag_scenes(config, {"frames": frames}, ctx)

    assert tagger_hub.requests == []
    assert ctx.progress_reports == []


def test_a_folder_of_plain_images_fails_saying_how_to_get_timestamps(tagger_hub, config, tmp_path):
    with pytest.raises(
        ValueError, match=r"^3 of 3 frames have no timestamp.*VideoThumbnail.*DetectScenes or ExtractFrames"
    ):
        tag_scenes(config, {"path": str(frames_folder(tmp_path, 3))})

    assert tagger_hub.requests == []


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"embeddingWeight": 1.5}, r"^`embeddingWeight` is not a number from 0 to 1: 1.5"),
        ({"minCoverage": "0.3"}, r"^`minCoverage` is not a number from 0 to 1"),
        ({"minSceneDuration": -1}, r"^`minSceneDuration` is not a number of seconds, 0 or more: -1"),
        ({"ignoreTags": "solo"}, r"^`ignoreTags` is not a list of tags: 'solo'"),
        ({"ignoreTags": [""]}, r"^`ignoreTags` is not a list of tags"),
        ({"includeTags": "no"}, r"^`includeTags` is not true or false"),
        ({"includeFrames": 1}, r"^`includeFrames` is not true or false"),
        ({"maxTags": 0}, r"^`maxTags` is not a whole number of 1 or more"),
        ({"thresholds": {"copyright": 0.3}}, r"^stand-in has no category `copyright`"),
        ({"lists": "both"}, r'^`lists` is not "result" or "file"'),
        ({"keyframeTags": ["smile"]}, r"^`keyframeTags` is not a map of tags to numbers"),
        ({"keyframeTags": {"smile": "1"}}, r"^`keyframeTags` is not a map of tags to numbers"),
        ({"keyframeTags": {"smile": True}}, r"^`keyframeTags` is not a map of tags to numbers"),
        ({"keyframeTags": {"": 1}}, r"^`keyframeTags` is not a map of tags to numbers"),
        ({"keyframeShare": 0}, r"^`keyframeShare` is not a number over 0, up to 1: 0"),
        ({"keyframeShare": 1.5}, r"^`keyframeShare` is not a number over 0, up to 1: 1.5"),
    ],
)
def test_bad_settings_fail_the_task_before_anything_downloads(tagger_hub, config, tmp_path, settings, message):
    frames, _ = video(tmp_path)

    with pytest.raises(ValueError, match=message):
        tag_scenes(config, {"frames": frames}, **settings)

    assert tagger_hub.requests == []


# Answers in a file.


def test_a_file_holds_the_scenes_and_frames_and_the_task_hands_on_the_summary(tagger_hub, config, data_dir, tmp_path):
    frames, _ = video(tmp_path)
    whole = tag_scenes(config, {"frames": frames}, includeFrames=True)

    result = tag_scenes(config, {"frames": frames}, includeFrames=True, lists="file")

    file = Path(result["file"])
    assert file.parent == data_dir / "results"
    written = json.loads(file.read_text(encoding="utf-8"))
    assert set(written) >= {"scenes", "frames", "summary"}
    assert written["scenes"] == whole["scenes"]
    assert written["frames"] == whole["frames"]
    assert not {"scenes", "frames"} & set(result)
    assert result["count"] == 3
    assert result["reused"] is False
    assert {key: value for key, value in result.items() if key not in ("file", "reused")} == {
        key: value for key, value in whole.items() if key not in ("scenes", "frames")
    }


def test_the_same_frames_and_settings_reuse_the_file_without_running_the_tagger(tagger_hub, config, tmp_path):
    frames, _ = video(tmp_path)
    first = tag_scenes(config, {"frames": frames}, lists="file")
    ctx = Context()

    again = tag_scenes(config, {"frames": frames}, ctx, lists="file")

    assert again == {**first, "reused": True}
    assert ctx.progress_reports == [(1.0, "Reused the answers of 3 scenes")]
    other = tag_scenes(config, {"frames": frames}, lists="file", embeddingWeight=1)
    assert other["file"] != first["file"]


# Batches, progress and Cancel.


def test_tagging_takes_the_progress_to_0_9_and_grouping_and_pooling_the_rest(tagger_hub, config, tmp_path):
    frames, _ = video(tmp_path, seconds=(8, 8, 8), blanks=0)  # 12 frames
    worker = TagScenes(config)
    tag_scenes(config, {"frames": frames}, worker=worker)
    ctx = Context()

    result = tag_scenes(config, {"frames": frames}, ctx, worker=worker)

    assert ctx.progress_reports == [
        (0.0, "Tagging 0 of 12 frames"),
        (pytest.approx(0.3), "Tagged 4 of 12 frames"),
        (pytest.approx(0.6), "Tagged 8 of 12 frames"),
        (pytest.approx(0.9), "Tagged 12 of 12 frames"),
        (0.9, "Grouping"),
        (pytest.approx(0.95), "Pooling"),
        (1.0, f"Found {result['count']} scenes in 12 frames"),
    ]


def test_it_runs_the_tagger_s_embedding_session_only(tagger_hub, config, tmp_path):
    frames, _ = video(tmp_path, seconds=(4, 4, 4), blanks=0)
    worker = TagScenes(config)

    tag_scenes(config, {"frames": frames}, worker=worker)

    assert set(worker.models._embedding_sessions) == {"stand-in"}
    assert worker.models._sessions == {}


class CancelledOn(Context):
    """A Context whose Task is cancelled once it says `said`."""

    def __init__(self, said):
        super().__init__()
        self.said = said

    def progress(self, fraction, message=None):
        super().progress(fraction, message)
        if message == self.said:
            self.cancelled = True


def test_a_cancel_stops_the_task_before_its_next_batch(tagger_hub, config, tmp_path):
    frames, _ = video(tmp_path, seconds=(8, 8, 8), blanks=0)
    ctx = CancelledOn("Tagged 4 of 12 frames")

    with pytest.raises(Exception, match="^Cancelled after 4 of 12 frames$"):
        tag_scenes(config, {"frames": frames}, ctx)

    assert [message for _, message in ctx.progress_reports if message.startswith("Tagged")] == ["Tagged 4 of 12 frames"]


@pytest.mark.parametrize(
    ("said", "message"),
    [
        ("Tagged 12 of 12 frames", "Cancelled after tagging 12 frames"),
        ("Grouping", "Cancelled after grouping 12 frames"),
    ],
)
def test_a_cancel_stops_the_task_between_its_stages(tagger_hub, config, tmp_path, said, message):
    frames, _ = video(tmp_path, seconds=(8, 8, 8), blanks=0)
    ctx = CancelledOn(said)

    with pytest.raises(Exception, match=f"^{message}$"):
        tag_scenes(config, {"frames": frames}, ctx)

    assert "Pooling" not in [message for _, message in ctx.progress_reports]


# What `frames.py` adds for it: blank frames, and the video's duration.


@pytest.mark.parametrize(
    ("image", "blank"),
    [
        (Image.new("RGB", (640, 360)), True),
        (Image.new("RGB", (640, 360), (255, 255, 255)), True),
        (Image.new("RGB", (640, 360), (20, 60, 200)), True),  # a solid card
        (Image.open(IMAGES / "anime-01-rainy-bus-stop.jpg").convert("RGB"), False),
    ],
)
def test_a_black_white_or_solid_frame_is_blank(image, blank):
    assert is_blank(image) is blank


def test_a_frame_just_over_blank_std_is_not_blank():
    # Half the frame at 0, half at 2 · (BLANK_STD + 1): a standard deviation of BLANK_STD + 1.
    image = Image.new("L", (64, 64))
    image.paste(2 * (BLANK_STD + 1), (0, 0, 64, 32))
    assert is_blank(image.convert("RGB")) is False
    assert framing.BLANK_STD == 4


def allow(path, access="read"):
    return str(path)


def test_frames_have_the_duration_their_thumbnails_agree_on(tmp_path):
    a, b = thumbnail(tmp_path, "a.jpg", 1), thumbnail(tmp_path, "b.jpg", 2)

    assert read_frames(str(tmp_path), allow).duration == 1621.12
    assert read_frame_list([{"path": str(a)}, {"path": str(b)}], allow).duration == 1621.12
    assert read_frames(str(a), allow).duration == 1621.12


def test_frames_have_no_duration_when_their_thumbnails_disagree_or_give_none(tmp_path):
    a = thumbnail(tmp_path, "a.jpg", 1)
    other = Image.open(a)
    meta = json.loads(other.info["comment"])
    other.save(tmp_path / "b.jpg", comment=json.dumps({**meta, "duration": 99.5}))

    assert read_frames(str(tmp_path), allow).duration is None
    assert read_frames(str(SHEETS / "sample.jpg"), allow).duration is None
    assert read_frames(str(IMAGES / "tiny.jpg"), allow).duration is None
