"""`ClassifyFrames` as the Host runs it: a whole video from its frame sheet, or a folder of frames, one answer over
them all and each frame's own."""

import json
from pathlib import Path

import pytest
from PIL import Image

from image_tags import frames as frames_module
from image_tags.worker import ClassifyFrames, ClassifyImage
from tests.conftest import FIXTURES, IMAGES, Context, Task, run

SHEETS = FIXTURES / "sheets"
#: `sample.jpg`: what `VideoThumbnail` made of its test clip (`pnpm sheets`): a 3x3 grid of 160x90 cells, 7 frames
#: and 2 empty cells. Its first frame was black and skipped, so the service took the one at 3 s instead.
SAMPLE_TIMESTAMPS = [3.0, 1.08, 2.0, 3.0, 3.16, 4.0, 5.0]
GOLDEN = json.loads((FIXTURES / "golden.json").read_text(encoding="utf-8"))
#: Reference frames (see `fixtures/images/CREDITS.md`), each 640 wide.
ANIME = "anime-02-train-window.jpg"
LIVE_ACTION = "live_action-02-bristol-street.jpg"


def sheet(config, cells, render_timestamps=False, video="D:/Videos/clip.mp4", name="sheet.jpg"):
    """A frame sheet as the thumbnail service writes one, of golden images: per cell, None or (an image of
    `tests/fixtures/images`, where its top-left corner goes, its timestamp). Each image keeps its own size, so the
    rect is its size; the canvas is grey around them. Written next to the data directory."""
    images = [None if cell is None else (Image.open(IMAGES / cell[0]).convert("RGB"), *cell[1:]) for cell in cells]
    width = max((x + image.width for image, (x, _), _ in filter(None, images)), default=64) + 10
    height = max((y + image.height for image, (_, y), _ in filter(None, images)), default=64) + 10
    canvas = Image.new("RGB", (width, height), (128, 128, 128))
    frames = []
    for i, cell in enumerate(images):
        if cell is None:
            frames.append(None)
            continue
        image, (x, y), timestamp = cell
        canvas.paste(image, (x, y))
        rect = {"x": x, "y": y, "w": image.width, "h": image.height}
        frames.append({"timestamp_requested": float(i), "timestamp_actual": timestamp, "rect": rect})
    meta = {
        "duration": 60.0,
        "size": {"w": width, "h": height},
        "frames": frames,
        "generated_at": "2026-11-10T00:00:00+00:00",
        "config": {"fit": "stretch", "render_timestamps": render_timestamps, "skip_blank_frames": True},
        **({} if video is None else {"video": video}),
    }
    path = Path(config["host"]["dataDir"]).parent / name
    canvas.save(path, quality=100, subsampling=0, comment=json.dumps(meta))
    return path


def classify(config, path, ctx=None, **settings):
    return run(ClassifyFrames(config), Task({"path": str(path)}, settings), ctx)


def test_a_frame_sheet_is_classified_frame_by_frame_with_its_timestamps_and_video(hub, config):
    result = classify(config, SHEETS / "sample.jpg")

    assert result["path"] == str(SHEETS / "sample.jpg")
    assert result["video"] == "sample.mp4"
    assert [frame["timestamp"] for frame in result["frames"]] == SAMPLE_TIMESTAMPS
    assert result["classifier"] == "style"
    assert result["labels"] == ["anime", "cg", "live_action"]
    assert result["models"] == ["anime_real", "anime_classification"]
    assert "warnings" not in result


def test_each_frame_is_cut_from_the_sheet_at_its_rect(hub, config):
    # Golden images pasted at their own size onto a grey canvas, apart, with a null cell between them, the later one
    # higher up: each frame must score as imgutils scores that image alone. (Two that survive the sheet's JPEG: the
    # train window, for one, does not quite, its not_painting .004 coming out .035.)
    festival = "anime-03-festival-night-market.jpg"
    path = sheet(config, [("tiny.jpg", (10, 400), 12.5), None, (festival, (100, 20), 30.0)])

    result = classify(config, path, classifier="anime_classification")

    assert [frame["timestamp"] for frame in result["frames"]] == [12.5, 30.0]
    tiny, anime = (frame["scores"] for frame in result["frames"])
    assert tiny == pytest.approx(GOLDEN["anime_classification"]["tiny.jpg"], abs=0.02)
    assert anime == pytest.approx(GOLDEN["anime_classification"][festival], abs=0.02)


def test_the_sample_sheet_s_frames_score_as_its_cells_cut_out_by_hand(hub, config, tmp_path):
    # The sample's grid: 3 columns of 160x90 cells, its 7 frames in the first 7.
    cells = []
    with Image.open(SHEETS / "sample.jpg") as image:
        for i in range(7):
            cell = tmp_path / f"cell{i}.png"
            image.crop((i % 3 * 160, i // 3 * 90, i % 3 * 160 + 160, i // 3 * 90 + 90)).save(cell)
            cells.append(cell)

    result = classify(config, SHEETS / "sample.jpg")

    worker = ClassifyImage(config)
    alone = [run(worker, Task({"path": str(cell)}))["scores"] for cell in cells]
    assert [frame["scores"] for frame in result["frames"]] == [pytest.approx(scores, abs=1e-4) for scores in alone]


def test_the_answer_is_the_mean_over_frames_and_agreement_the_share_of_frames_whose_top_label_it_is(hub, config):
    # Two anime frames (anime 1.00 and 1.00) around a live-action one (live_action .93, anime .01): the mean is anime
    # (1.000 + .007 + .999) / 3 = .668 over live_action (.000 + .933 + .000) / 3 = .311.
    anime_again = "anime-01-rainy-bus-stop.jpg"
    path = sheet(config, [(ANIME, (0, 0), 1.0), (LIVE_ACTION, (650, 0), 2.0), (anime_again, (1300, 0), 3.0)])

    result = classify(config, path)

    assert result["tag"] == "anime"
    assert result["confidence"] == pytest.approx(0.668, abs=0.02)
    assert result["confidence"] == result["scores"]["anime"]
    assert result["scores"]["live_action"] == pytest.approx(0.311, abs=0.02)
    assert result["agreement"] == pytest.approx(2 / 3)
    assert [frame["tag"] for frame in result["frames"]] == ["anime", "live_action", "anime"]
    assert result["video"] == "D:/Videos/clip.mp4"


def test_an_uncertain_answer_keeps_its_top_label_s_mean_and_agreement(hub, config):
    # As above, the last frame one `style` is less sure of (the small synthetic one, anime .85): the mean is anime
    # (1.000 + .007 + .848) / 3 = .618.
    less_sure = "tiny.jpg"
    path = sheet(config, [(ANIME, (0, 0), 1.0), (LIVE_ACTION, (650, 0), 2.0), (less_sure, (1300, 0), 3.0)])

    result = classify(config, path, minConfidence=0.9)

    assert result["tag"] == "uncertain"
    assert result["confidence"] == pytest.approx(0.618, abs=0.02)
    assert result["confidence"] == result["scores"]["anime"]
    assert result["agreement"] == pytest.approx(2 / 3)
    # Each frame is held to the same threshold: the first two (1.00 and .93) reach it, the last one's .85 not.
    assert [frame["tag"] for frame in result["frames"]] == ["anime", "live_action", "uncertain"]


def test_a_sheet_with_timestamps_drawn_is_classified_with_a_warning(hub, config):
    result = classify(config, SHEETS / "sample-timestamps.jpg")

    assert [frame["timestamp"] for frame in result["frames"]] == SAMPLE_TIMESTAMPS
    assert len(result["warnings"]) == 1
    assert "timestamps drawn" in result["warnings"][0]


def test_a_sheet_without_a_video_path_has_no_video(hub, config):
    result = classify(config, sheet(config, [(ANIME, (0, 0), 1.0)], video=None))

    assert "video" not in result


def test_a_sheet_with_no_frames_fails(config):
    path = sheet(config, [None, None])

    with pytest.raises(Exception, match=r"^The frame sheet has no frames: .*sheet\.jpg"):
        classify(config, path)


def test_a_plain_image_is_one_frame(hub, config):
    result = classify(config, IMAGES / ANIME, classifier="anime_classification")

    assert len(result["frames"]) == 1
    assert result["frames"][0]["timestamp"] is None
    assert result["frames"][0]["scores"] == result["scores"]
    assert result["scores"] == pytest.approx(GOLDEN["anime_classification"][ANIME], abs=0.02)
    assert result["agreement"] == 1
    assert "video" not in result


def test_a_folder_is_its_images_in_name_order_each_a_frame_without_a_timestamp(hub, config, tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    (folder / "b.jpg").write_bytes((IMAGES / ANIME).read_bytes())
    (folder / "a.jpg").write_bytes((IMAGES / "tiny.jpg").read_bytes())
    # A frame sheet in a folder is a plain image: one frame, no video, no warning.
    (folder / "c-sheet.jpg").write_bytes((SHEETS / "sample-timestamps.jpg").read_bytes())
    (folder / "notes.txt").write_text("not an image")
    (folder / "inner").mkdir()
    (folder / "inner" / "0.jpg").write_bytes((IMAGES / ANIME).read_bytes())

    result = classify(config, folder, classifier="anime_classification")

    assert result["path"] == str(folder)
    assert [frame["timestamp"] for frame in result["frames"]] == [None, None, None]
    tiny, anime, _ = (frame["scores"] for frame in result["frames"])
    assert tiny == pytest.approx(GOLDEN["anime_classification"]["tiny.jpg"], abs=0.02)
    assert anime == pytest.approx(GOLDEN["anime_classification"][ANIME], abs=0.02)
    assert "video" not in result
    assert "warnings" not in result


def test_a_folder_with_no_images_fails(config, tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    (folder / "notes.txt").write_text("not an image")

    with pytest.raises(Exception, match=r"^The folder has no images: .*empty"):
        classify(config, folder)


def test_a_folder_of_more_than_10000_images_fails(config, tmp_path, monkeypatch):
    assert frames_module.MAX_FOLDER_IMAGES == 10_000
    monkeypatch.setattr(frames_module, "MAX_FOLDER_IMAGES", 3)  # rather than writing 10001 images
    folder = tmp_path / "many"
    folder.mkdir()
    pixel = Image.new("RGB", (1, 1))
    for i in range(4):
        pixel.save(folder / f"{i:04}.png")

    with pytest.raises(Exception, match=r"^The folder has more than 3 images: .*many"):
        classify(config, folder)


def frames_folder(tmp_path, count):
    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(count):
        (folder / f"{i:03}.jpg").write_bytes((IMAGES / "tiny.jpg").read_bytes())
    return folder


def classified(ctx):
    return [(fraction, message) for fraction, message in ctx.progress_reports if message.startswith("Classified")]


def test_frames_run_in_batches_of_16_reporting_progress_after_each(hub, config, tmp_path):
    ctx = Context()

    result = classify(config, frames_folder(tmp_path, 35), ctx, classifier="anime_real")

    assert len(result["frames"]) == 35
    assert classified(ctx) == [
        (16 / 35, "Classified 16 of 35 frames"),
        (32 / 35, "Classified 32 of 35 frames"),
        (1.0, "Classified 35 of 35 frames"),
    ]


def test_the_profile_s_batch_size_sets_how_many_frames_run_at_once(hub, config, tmp_path):
    config["batchSize"] = 4
    ctx = Context()

    result = classify(config, frames_folder(tmp_path, 10), ctx, classifier="anime_real")

    assert len(result["frames"]) == 10
    assert [message for _, message in classified(ctx)] == [
        "Classified 4 of 10 frames",
        "Classified 8 of 10 frames",
        "Classified 10 of 10 frames",
    ]


def test_a_cancel_stops_the_task_before_its_next_batch(hub, config, tmp_path):
    class CancelledAfterOneBatch(Context):
        def progress(self, fraction, message=None):
            super().progress(fraction, message)
            if message and message.startswith("Classified"):
                self.cancelled = True

    ctx = CancelledAfterOneBatch()

    with pytest.raises(Exception, match="Cancelled after 16 of 35 frames"):
        classify(config, frames_folder(tmp_path, 35), ctx, classifier="anime_real")

    assert len(classified(ctx)) == 1


def test_classify_image_scores_a_frame_sheet_as_one_image_with_a_warning(hub, config):
    result = run(ClassifyImage(config), Task({"path": str(SHEETS / "sample.jpg")}))

    assert result["video"] == "sample.mp4"
    assert "frames" not in result
    assert len(result["warnings"]) == 1
    assert "ClassifyFrames" in result["warnings"][0]


# A list of frames: each its own image, with its timestamp.


def classify_list(config, frames, ctx=None, **settings):
    return run(ClassifyFrames(config), Task({"frames": frames}, settings), ctx)


def test_a_list_of_frames_is_classified_in_its_order_with_each_frame_s_timestamp_and_path(hub, config):
    anime, tiny, sheet = str(IMAGES / ANIME), str(IMAGES / "tiny.jpg"), str(SHEETS / "sample.jpg")

    result = classify_list(
        config,
        [{"path": tiny, "timestamp": 12.5}, {"path": anime, "timestamp": 0}, {"path": sheet}],
        classifier="anime_classification",
    )

    assert "path" not in result
    assert [(frame["path"], frame["timestamp"]) for frame in result["frames"]] == [
        (tiny, 12.5),
        (anime, 0.0),
        (sheet, None),  # a frame sheet in a list is a plain image, as in a folder
    ]
    assert result["frames"][0]["scores"] == pytest.approx(GOLDEN["anime_classification"]["tiny.jpg"], abs=0.02)
    assert result["frames"][1]["scores"] == pytest.approx(GOLDEN["anime_classification"][ANIME], abs=0.02)
    assert "video" not in result
    assert "warnings" not in result


def test_a_folder_s_frames_name_their_files(hub, config, tmp_path):
    folder = frames_folder(tmp_path, 2)

    result = classify(config, folder, classifier="anime_real")

    assert [frame["path"] for frame in result["frames"]] == [str(folder / "000.jpg"), str(folder / "001.jpg")]


def test_a_sheet_s_and_an_image_s_frames_have_no_path_of_their_own(hub, config):
    for path in (SHEETS / "sample.jpg", IMAGES / ANIME):
        result = classify(config, path, classifier="anime_real")
        assert all("path" not in frame for frame in result["frames"])


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        ("D:/frames", r"^`frames` is not a list"),
        ([], r"^`frames` is empty"),
        (["D:/a.jpg"], r"^Frame 0 is not an object with a `path`"),
        ([{"timestamp": 1}], r"^Frame 0 is not an object with a `path`"),
        ([{"path": ""}], r"^Frame 0 is not an object with a `path`"),
        ([{"path": str(IMAGES / ANIME)}, {"path": str(IMAGES / ANIME), "timestamp": -1}], r"^Frame 1's `timestamp`"),
        ([{"path": str(IMAGES / ANIME), "timestamp": "1.5"}], r"^Frame 0's `timestamp` is not a number of 0 or more"),
        ([{"path": str(IMAGES / ANIME), "timestamp": True}], r"^Frame 0's `timestamp`"),
        ([{"path": str(IMAGES / ANIME), "timestamp": float("nan")}], r"^Frame 0's `timestamp`"),
        ([{"path": str(IMAGES)}], r"^Frame 0 is not an image: .*images"),
        ([{"path": str(IMAGES / "CREDITS.md")}], r"^Frame 0 is not an image: .*CREDITS\.md"),
        ([{"path": str(IMAGES / "missing.jpg")}], r"^Frame 0 is not an image: .*missing\.jpg"),
    ],
)
def test_a_list_that_is_not_one_of_frames_fails(config, frames, message):
    with pytest.raises(Exception, match=message):
        classify_list(config, frames)


def test_a_list_of_more_than_10000_frames_fails(config):
    frames = [{"path": str(IMAGES / "tiny.jpg")}] * 10_001

    with pytest.raises(Exception, match=r"^`frames` has more than 10000 frames"):
        classify_list(config, frames)


@pytest.mark.parametrize(
    ("input", "message"),
    [
        ({}, r"^The input has no `path` or `frames`: the frames to classify"),
        ({"path": str(IMAGES / ANIME), "frames": [{"path": str(IMAGES / ANIME)}]}, r"^The input has both `path` and"),
    ],
)
def test_an_input_needs_a_path_or_frames_not_both(config, input, message):
    with pytest.raises(Exception, match=message):
        run(ClassifyFrames(config), Task(input))


# Thumbnails: one frame each, whose comment says where in which video it is.


def thumbnail(folder, name, timestamp, video="D:/Videos/holiday.mp4", image=ANIME):
    """A thumbnail as VideoThumbnail writes one: a JPEG of one frame, its JSON in the comment."""
    meta = {"duration": 1621.12, "index": 7, "scene": 3, "timestamp": timestamp, "fingerprint": "fp1:test"}
    if video is not None:
        meta["video"] = video
    path = folder / name
    Image.open(IMAGES / image).convert("RGB").save(path, quality=95, comment=json.dumps(meta))
    return path


def test_a_thumbnail_alone_is_one_frame_with_its_timestamp_and_video(hub, config, tmp_path):
    result = classify(config, thumbnail(tmp_path, "t.jpg", 141.17), classifier="anime_real")

    assert result["frames"][0]["timestamp"] == 141.17
    assert result["video"] == "D:/Videos/holiday.mp4"


def test_a_folder_of_thumbnails_has_their_timestamps_and_their_one_video(hub, config, tmp_path):
    thumbnail(tmp_path, "a.jpg", 141.17)
    thumbnail(tmp_path, "b.jpg", 200, video=None)
    (tmp_path / "c.jpg").write_bytes((IMAGES / "tiny.jpg").read_bytes())

    result = classify(config, tmp_path, classifier="anime_real")

    assert [frame["timestamp"] for frame in result["frames"]] == [141.17, 200.0, None]
    assert result["video"] == "D:/Videos/holiday.mp4"


def test_thumbnails_of_different_videos_name_no_video(hub, config, tmp_path):
    thumbnail(tmp_path, "a.jpg", 1, video="D:/Videos/one.mp4")
    thumbnail(tmp_path, "b.jpg", 2, video="D:/Videos/two.mp4")

    result = classify(config, tmp_path, classifier="anime_real")

    assert [frame["timestamp"] for frame in result["frames"]] == [1.0, 2.0]
    assert "video" not in result


def test_a_listed_timestamp_wins_over_the_thumbnail_s_own(hub, config, tmp_path):
    a, b = thumbnail(tmp_path, "a.jpg", 141.17), thumbnail(tmp_path, "b.jpg", 9.5)

    result = classify_list(config, [{"path": str(a), "timestamp": 5}, {"path": str(b)}], classifier="anime_real")

    assert [frame["timestamp"] for frame in result["frames"]] == [5.0, 9.5]
    assert result["video"] == "D:/Videos/holiday.mp4"


def test_a_thumbnail_s_bad_timestamp_is_no_timestamp(hub, config, tmp_path):
    result = classify(config, thumbnail(tmp_path, "t.jpg", "soon"), classifier="anime_real")

    assert result["frames"][0]["timestamp"] is None
