"""`TagFrames` as the Host runs it: each frame of a frame sheet, a folder or one image through one tagger, the frames'
own tags and rating in order.

The Worker runs the stand-in tagger (`stand_in.py`), as `test_tag_image.py` does; `test_tag_frames_golden.py` holds
the real one's frames to `TagImage`'s.
"""

from dataclasses import replace

import pytest
from PIL import Image

from image_tags.catalogue import TAGGERS
from image_tags.frames import TIMESTAMPS_DRAWN
from image_tags.worker import TagFrames, TagImage, ran_on_the_cpu
from tests.conftest import IMAGES, Context, Task, run
from tests.test_classify_frames import SAMPLE_TIMESTAMPS, SHEETS, frames_folder
from tests.test_tag_image import NEEDS_MORE, OUT_OF_MEMORY, on_a_gpu  # noqa: F401  (a fixture)

SHEET = SHEETS / "sample.jpg"
ANIME = IMAGES / "anime-02-train-window.jpg"
EVERY_TAG = {"general": 0, "character": 0}


def frames_task(path, **settings):
    return Task({"path": str(path)}, {"tagger": "stand-in", **settings})


def tag_frames(config, path, ctx=None, **settings):
    return run(TagFrames(config), frames_task(path, **settings), ctx)


def cpu_warning():
    """`pnpm check` has the `cpu` build only."""
    return ran_on_the_cpu(TAGGERS["stand-in"])


# What it reads.


def test_a_frame_sheet_is_tagged_frame_by_frame_with_its_timestamps_and_video(tagger_hub, config):
    result = tag_frames(config, SHEET)

    assert result["path"] == str(SHEET)
    assert result["video"] == "sample.mp4"
    assert result["tagger"] == "stand-in"
    assert [frame["timestamp"] for frame in result["frames"]] == SAMPLE_TIMESTAMPS
    for frame in result["frames"]:
        assert set(frame) == {"timestamp", "tags", "rating"}
        scores = [t["score"] for t in frame["tags"]]
        assert scores == sorted(scores, reverse=True)
        assert frame["rating"]["label"] == max(frame["rating"]["scores"], key=frame["rating"]["scores"].get)
    assert result["warnings"] == [cpu_warning()]


def test_each_frame_is_tagged_as_tag_image_tags_its_cell_cut_out_by_hand(tagger_hub, config, tmp_path):
    # The sample's grid: 3 columns of 160x90 cells, its 7 frames in the first 7.
    cells = []
    with Image.open(SHEET) as image:
        for i in range(7):
            cell = tmp_path / f"cell{i}.png"
            image.crop((i % 3 * 160, i // 3 * 90, i % 3 * 160 + 160, i // 3 * 90 + 90)).save(cell)
            cells.append(cell)

    result = tag_frames(config, SHEET, thresholds=EVERY_TAG)

    alone = TagImage(config)
    for frame, cell in zip(result["frames"], cells, strict=True):
        want = run(alone, Task({"path": str(cell)}, {"tagger": "stand-in", "thresholds": EVERY_TAG}))
        assert [t["name"] for t in frame["tags"]] == [t["name"] for t in want["tags"]]
        assert [t["score"] for t in frame["tags"]] == pytest.approx([t["score"] for t in want["tags"]], abs=1e-5)
        assert frame["rating"]["scores"] == pytest.approx(want["rating"]["scores"], abs=1e-5)


def test_a_sheet_with_timestamps_drawn_is_tagged_with_a_warning(tagger_hub, config):
    result = tag_frames(config, SHEETS / "sample-timestamps.jpg")

    assert [frame["timestamp"] for frame in result["frames"]] == SAMPLE_TIMESTAMPS
    assert result["warnings"] == [TIMESTAMPS_DRAWN, cpu_warning()]


def test_a_folder_is_its_images_in_name_order_each_a_frame_without_a_timestamp(tagger_hub, config, tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    (folder / "b.jpg").write_bytes(ANIME.read_bytes())
    (folder / "a.jpg").write_bytes((IMAGES / "tiny.jpg").read_bytes())
    (folder / "c-sheet.jpg").write_bytes(SHEET.read_bytes())  # a plain image in a folder
    (folder / "notes.txt").write_text("not an image")

    result = tag_frames(config, folder, thresholds=EVERY_TAG)

    assert result["path"] == str(folder)
    assert [frame["timestamp"] for frame in result["frames"]] == [None, None, None]
    alone = TagImage(config)
    for frame, name in zip(result["frames"], ["a.jpg", "b.jpg", "c-sheet.jpg"], strict=True):
        want = run(alone, Task({"path": str(folder / name)}, {"tagger": "stand-in", "thresholds": EVERY_TAG}))
        assert frame["rating"]["scores"] == pytest.approx(want["rating"]["scores"], abs=1e-5)
    assert "video" not in result
    assert result["warnings"] == [cpu_warning()]


def test_a_plain_image_is_one_frame_tagged_as_tag_image_tags_it(tagger_hub, config):
    result = tag_frames(config, ANIME, thresholds=EVERY_TAG)

    want = run(TagImage(config), frames_task(ANIME, thresholds=EVERY_TAG))
    assert len(result["frames"]) == 1
    frame = result["frames"][0]
    assert frame["timestamp"] is None
    assert [t["name"] for t in frame["tags"]] == [t["name"] for t in want["tags"]]
    assert frame["rating"]["scores"] == pytest.approx(want["rating"]["scores"], abs=1e-5)
    assert "video" not in result


def test_a_folder_with_no_images_fails(tagger_hub, config, tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()

    with pytest.raises(Exception, match=r"^The folder has no images: .*empty"):
        tag_frames(config, folder)


def test_an_input_without_a_path_fails(tagger_hub, config):
    with pytest.raises(Exception, match=r"no `path`: the frames to tag"):
        run(TagFrames(config), Task({}, {"tagger": "stand-in"}))


# Its settings, each frame's own.


def test_thresholds_merge_over_the_tagger_s_defaults_for_every_frame(tagger_hub, config):
    worker = TagFrames(config)
    every = run(worker, frames_task(SHEET, thresholds=EVERY_TAG))
    no_characters = run(worker, frames_task(SHEET, thresholds={"character": 1}))
    default = run(worker, frames_task(SHEET))

    for all_tags, without, plain in zip(every["frames"], no_characters["frames"], default["frames"], strict=True):
        assert len(all_tags["tags"]) == 6
        assert [t for t in without["tags"] if t["category"] == "character"] == []
        general = [t for t in plain["tags"] if t["category"] == "general"]
        assert [t for t in without["tags"] if t["category"] == "general"] == general
        for t in plain["tags"]:
            assert t["score"] > {"general": 0.35, "character": 0.85}[t["category"]]


def test_max_tags_hands_on_each_frame_s_highest_only(tagger_hub, config):
    worker = TagFrames(config)
    every = run(worker, frames_task(SHEET, thresholds=EVERY_TAG))

    result = run(worker, frames_task(SHEET, thresholds=EVERY_TAG, maxTags=2))

    assert [frame["tags"] for frame in result["frames"]] == [frame["tags"][:2] for frame in every["frames"]]


def test_a_threshold_for_a_category_the_tagger_lacks_fails_the_task_before_anything_downloads(tagger_hub, config):
    lacks = r"^stand-in has no category `copyright`: its categories are general, character"
    with pytest.raises(Exception, match=lacks):
        tag_frames(config, SHEET, thresholds={"copyright": 0.3})

    assert tagger_hub.requests == []


def test_an_unknown_tagger_fails_naming_the_ones_there_are(tagger_hub, config):
    with pytest.raises(Exception, match=r"^No tagger `nope`: one of wd-eva02-large, stand-in$"):
        tag_frames(config, SHEET, tagger="nope")


def test_the_default_tagger_is_wd_eva02_large(tagger_hub, config, monkeypatch):
    monkeypatch.setitem(TAGGERS, "wd-eva02-large", replace(TAGGERS["stand-in"], id="wd-eva02-large"))

    result = run(TagFrames(config), Task({"path": str(ANIME)}))

    assert result["tagger"] == "wd-eva02-large"


# Batches and progress.

DOWNLOADING = "Downloading the model stand-in"


def tagged(ctx):
    return [(fraction, message) for fraction, message in ctx.progress_reports if message.startswith("Tagged")]


def test_a_task_whose_tagger_downloads_says_each_step_in_order(tagger_hub, config, tmp_path):
    ctx = Context()

    tag_frames(config, frames_folder(tmp_path, 3), ctx)

    messages = [message for _, message in ctx.progress_reports]
    assert messages.index("Downloading the tag list of stand-in") < messages.index(DOWNLOADING)
    last = max(i for i, message in enumerate(messages) if message == DOWNLOADING)
    assert ctx.progress_reports[last:] == [
        (1.0, DOWNLOADING),
        (0.0, "Loading the model stand-in on the CPU"),
        (0.0, "Tagging 0 of 3 frames"),
        (1.0, "Tagged 3 of 3 frames"),
    ]


def test_a_task_whose_tagger_is_loaded_says_only_how_far_it_has_tagged(tagger_hub, config, tmp_path):
    worker = TagFrames(config)
    task = frames_task(frames_folder(tmp_path, 3))
    run(worker, task)
    ctx = Context()

    run(worker, task, ctx)

    assert ctx.progress_reports == [(0.0, "Tagging 0 of 3 frames"), (1.0, "Tagged 3 of 3 frames")]


def test_frames_run_in_batches_of_tag_batch_size_four_by_default(tagger_hub, config, tmp_path):
    ctx = Context()

    result = tag_frames(config, frames_folder(tmp_path, 10), ctx)

    assert len(result["frames"]) == 10
    assert tagged(ctx) == [
        (4 / 10, "Tagged 4 of 10 frames"),
        (8 / 10, "Tagged 8 of 10 frames"),
        (1.0, "Tagged 10 of 10 frames"),
    ]


def test_the_profile_s_tag_batch_size_sets_how_many_frames_run_at_once_not_batch_size(tagger_hub, config, tmp_path):
    config["batchSize"] = 16
    config["tagBatchSize"] = 3
    ctx = Context()

    tag_frames(config, frames_folder(tmp_path, 7), ctx)

    assert [message for _, message in tagged(ctx)] == [
        "Tagged 3 of 7 frames",
        "Tagged 6 of 7 frames",
        "Tagged 7 of 7 frames",
    ]


def test_a_cancel_stops_the_task_before_its_next_batch(tagger_hub, config, tmp_path):
    class CancelledAfterOneBatch(Context):
        def progress(self, fraction, message=None):
            super().progress(fraction, message)
            if message and message.startswith("Tagged"):
                self.cancelled = True

    ctx = CancelledAfterOneBatch()

    with pytest.raises(Exception, match="Cancelled after 4 of 10 frames"):
        tag_frames(config, frames_folder(tmp_path, 10), ctx)

    assert len(tagged(ctx)) == 1


# Teardown, and a GPU (faked).


def test_teardown_drops_the_tagger_s_session_and_a_second_is_harmless(tagger_hub, config):
    worker = TagFrames(config)
    run(worker, frames_task(ANIME))
    assert set(worker.models._sessions) == {"stand-in"}

    worker.teardown()
    worker.teardown()

    assert worker.models._sessions == {}


def test_on_a_gpu_there_is_no_cpu_warning(tagger_hub, config, on_a_gpu):  # noqa: F811
    result = tag_frames(config, SHEET)

    assert "warnings" not in result
    assert [len(frame["tags"]) for frame in result["frames"]] == [6] * 7


def test_a_forward_pass_out_of_gpu_memory_fails_the_task_saying_what_to_do(tagger_hub, config, on_a_gpu):  # noqa: F811
    on_a_gpu.run_with = RuntimeError(OUT_OF_MEMORY)

    with pytest.raises(Exception) as failed:
        tag_frames(config, SHEET)

    assert str(failed.value) == NEEDS_MORE
