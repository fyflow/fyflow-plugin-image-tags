"""`ClassifyFrames` as the Host runs it: a whole video from its frame sheet, or a folder of frames, one answer over
them all and each frame's own."""

import json
from pathlib import Path

import pytest
from PIL import Image

from image_tags.worker import ClassifyFrames, ClassifyImage
from tests.conftest import FIXTURES, IMAGES, Context, Task

SHEETS = FIXTURES / "sheets"
#: `sample.jpg`: what `VideoThumbnail` made of its test clip (`pnpm sheets`): a 3x3 grid of 160x90 cells, 7 frames
#: and 2 empty cells. Its first frame was black and skipped, so the service took the one at 3 s instead.
SAMPLE_TIMESTAMPS = [3.0, 1.08, 2.0, 3.0, 3.16, 4.0, 5.0]
GOLDEN = json.loads((FIXTURES / "golden.json").read_text(encoding="utf-8"))


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
    return ClassifyFrames(config).run(Task({"path": str(path)}, settings), ctx or Context())


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
    # higher up: each frame must score as imgutils scores that image alone. (Two that survive the sheet's JPEG.)
    path = sheet(config, [("tiny.jpg", (10, 400), 12.5), None, ("cartoon.png", (100, 20), 30.0)])

    result = classify(config, path, classifier="anime_classification")

    assert [frame["timestamp"] for frame in result["frames"]] == [12.5, 30.0]
    tiny, cartoon = (frame["scores"] for frame in result["frames"])
    assert tiny == pytest.approx(GOLDEN["anime_classification"]["tiny.jpg"], abs=0.02)
    assert cartoon == pytest.approx(GOLDEN["anime_classification"]["cartoon.png"], abs=0.02)


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
    alone = [worker.run(Task({"path": str(cell)}), Context())["scores"] for cell in cells]
    assert [frame["scores"] for frame in result["frames"]] == [pytest.approx(scores, abs=1e-4) for scores in alone]


def test_the_answer_is_the_mean_over_frames_and_agreement_the_share_of_frames_whose_top_label_it_is(hub, config):
    # Two cartoons (illustration .66, not_painting .09) and the tiny image (not_painting .90, illustration .06): the
    # mean is illustration (2 × .656 + .064) / 3 = .459 over not_painting (2 × .094 + .900) / 3 = .362.
    path = sheet(config, [("cartoon.png", (0, 0), 1.0), ("tiny.jpg", (520, 0), 2.0), ("cartoon.png", (600, 0), 3.0)])

    result = classify(config, path, classifier="anime_classification")

    assert result["tag"] == "illustration"
    assert result["confidence"] == pytest.approx(0.459, abs=0.02)
    assert result["confidence"] == result["scores"]["illustration"]
    assert result["scores"]["not_painting"] == pytest.approx(0.362, abs=0.02)
    assert result["agreement"] == pytest.approx(2 / 3)
    assert [frame["tag"] for frame in result["frames"]] == ["illustration", "not_painting", "illustration"]
    assert result["video"] == "D:/Videos/clip.mp4"


def test_an_uncertain_answer_keeps_its_top_label_s_mean_and_agreement(hub, config):
    path = sheet(config, [("cartoon.png", (0, 0), 1.0), ("tiny.jpg", (520, 0), 2.0), ("cartoon.png", (600, 0), 3.0)])

    result = classify(config, path, classifier="anime_classification", minConfidence=0.7)

    assert result["tag"] == "uncertain"
    assert result["confidence"] == pytest.approx(0.459, abs=0.02)
    assert result["agreement"] == pytest.approx(2 / 3)
    # Each frame is held to the same threshold: the tiny image's not_painting (.90) reaches it, a cartoon's .66 not.
    assert [frame["tag"] for frame in result["frames"]] == ["uncertain", "not_painting", "uncertain"]


def test_a_sheet_with_timestamps_drawn_is_classified_with_a_warning(hub, config):
    result = classify(config, SHEETS / "sample-timestamps.jpg")

    assert [frame["timestamp"] for frame in result["frames"]] == SAMPLE_TIMESTAMPS
    assert len(result["warnings"]) == 1
    assert "timestamps drawn" in result["warnings"][0]


def test_a_sheet_without_a_video_path_has_no_video(hub, config):
    result = classify(config, sheet(config, [("cartoon.png", (0, 0), 1.0)], video=None))

    assert "video" not in result


def test_a_sheet_with_no_frames_fails(config):
    path = sheet(config, [None, None])

    with pytest.raises(Exception, match=r"^The frame sheet has no frames: .*sheet\.jpg"):
        classify(config, path)


def test_a_plain_image_is_one_frame(hub, config):
    result = classify(config, IMAGES / "cartoon.png", classifier="anime_classification")

    assert len(result["frames"]) == 1
    assert result["frames"][0]["timestamp"] is None
    assert result["frames"][0]["scores"] == result["scores"]
    assert result["scores"] == pytest.approx(GOLDEN["anime_classification"]["cartoon.png"], abs=0.02)
    assert result["agreement"] == 1
    assert "video" not in result


def test_a_folder_is_its_images_in_name_order_each_a_frame_without_a_timestamp(hub, config, tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    (folder / "b.png").write_bytes((IMAGES / "cartoon.png").read_bytes())
    (folder / "a.jpg").write_bytes((IMAGES / "tiny.jpg").read_bytes())
    # A frame sheet in a folder is a plain image: one frame, no video, no warning.
    (folder / "c-sheet.jpg").write_bytes((SHEETS / "sample-timestamps.jpg").read_bytes())
    (folder / "notes.txt").write_text("not an image")
    (folder / "inner").mkdir()
    (folder / "inner" / "0.png").write_bytes((IMAGES / "cartoon.png").read_bytes())

    result = classify(config, folder, classifier="anime_classification")

    assert result["path"] == str(folder)
    assert [frame["timestamp"] for frame in result["frames"]] == [None, None, None]
    tiny, cartoon, _ = (frame["scores"] for frame in result["frames"])
    assert tiny == pytest.approx(GOLDEN["anime_classification"]["tiny.jpg"], abs=0.02)
    assert cartoon == pytest.approx(GOLDEN["anime_classification"]["cartoon.png"], abs=0.02)
    assert "video" not in result
    assert "warnings" not in result


def test_a_folder_with_no_images_fails(config, tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    (folder / "notes.txt").write_text("not an image")

    with pytest.raises(Exception, match=r"^The folder has no images: .*empty"):
        classify(config, folder)


def test_a_folder_of_more_than_1000_images_fails(config, tmp_path):
    folder = tmp_path / "many"
    folder.mkdir()
    pixel = Image.new("RGB", (1, 1))
    for i in range(1001):
        pixel.save(folder / f"{i:04}.png")

    with pytest.raises(Exception, match=r"^The folder has more than 1000 images: .*many"):
        classify(config, folder)


def frames_folder(tmp_path, count):
    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(count):
        (folder / f"{i:03}.jpg").write_bytes((IMAGES / "tiny.jpg").read_bytes())
    return folder


def classified(ctx):
    return [(fraction, message) for fraction, message in ctx.progress_reports if message.startswith("Classified")]


def test_frames_run_in_batches_of_32_reporting_progress_after_each(hub, config, tmp_path):
    ctx = Context()

    result = classify(config, frames_folder(tmp_path, 70), ctx, classifier="anime_real")

    assert len(result["frames"]) == 70
    assert classified(ctx) == [
        (32 / 70, "Classified 32 of 70 frames"),
        (64 / 70, "Classified 64 of 70 frames"),
        (1.0, "Classified 70 of 70 frames"),
    ]


def test_a_cancel_stops_the_task_before_its_next_batch(hub, config, tmp_path):
    class CancelledAfterOneBatch(Context):
        def progress(self, fraction, message=None):
            super().progress(fraction, message)
            if message and message.startswith("Classified"):
                self.cancelled = True

    ctx = CancelledAfterOneBatch()

    with pytest.raises(Exception, match="Cancelled after 32 of 70 frames"):
        classify(config, frames_folder(tmp_path, 70), ctx, classifier="anime_real")

    assert len(classified(ctx)) == 1


def test_classify_image_scores_a_frame_sheet_as_one_image_with_a_warning(hub, config):
    result = ClassifyImage(config).run(Task({"path": str(SHEETS / "sample.jpg")}), Context())

    assert result["video"] == "sample.mp4"
    assert "frames" not in result
    assert len(result["warnings"]) == 1
    assert "ClassifyFrames" in result["warnings"][0]
