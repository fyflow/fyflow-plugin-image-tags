"""`TagImage` as the Host runs it: one image per Task, through one tagger of the catalogue.

The Worker runs a stand-in tagger (`stand_in.py`) with `wd-eva02-large`'s input, output and categories; the real one's
scores are held to the reference pipeline's by `test_tag_image_golden.py`.
"""

from dataclasses import replace

import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

from image_tags import devices, models
from image_tags.catalogue import (
    ANIME_CLASSIFICATION,
    ANIME_REAL,
    TAGGERS,
    WD_EVA02_LARGE,
    Category,
    Labels,
    labels_from_csv,
    tag,
    thresholds_of,
)
from image_tags.models import Models
from image_tags.preprocess import BGR, NCHW, NHWC, RGB, WHITE, Preprocessing, to_input
from image_tags.worker import DEFAULT_TAG_BATCH_SIZE, TAG_TASKS_AT_ONCE, TagImage, ran_on_the_cpu
from tests.conftest import IMAGES, Context, Running, Task, run, served, wait_until
from tests.stand_in import TAG_LIST, lay_out
from tests.test_classify_image_batching import ForwardPasses, lining_up

IMAGE = IMAGES / "anime-02-train-window.jpg"
FRAMES = sorted(path for path in IMAGES.glob("*-*.jpg"))
RATINGS = ["general", "sensitive", "questionable", "explicit"]


def tag_task(path=IMAGE, **settings):
    return Task({"path": str(path)}, {"tagger": "stand-in", **settings})


# The catalogue entry.


def test_wd_eva02_large_is_pinned_to_its_commit_files_and_sha256s():
    wd = TAGGERS["wd-eva02-large"]

    assert wd is WD_EVA02_LARGE
    assert wd.kind == "tagger"
    assert wd.repo == "SmilingWolf/wd-eva02-large-tagger-v3"
    assert wd.commit == "c5303bb7139430db980e4c680a778fe79d72b541"
    assert wd.remote_path == f"SmilingWolf/wd-eva02-large-tagger-v3/resolve/{wd.commit}/model.onnx"
    assert wd.local_path == "SmilingWolf/wd-eva02-large-tagger-v3/c5303bb71394/model.onnx"
    assert wd.sha256 == "9e768793060c7939b277ccb382783e8670e8a042d29d77aa736be0c8cc898bfc"
    assert wd.tag_list.local_path == "SmilingWolf/wd-eva02-large-tagger-v3/c5303bb71394/selected_tags.csv"
    assert wd.tag_list.sha256 == "298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217"
    assert wd.categories == (Category("general", 0, 0.35), Category("character", 4, 0.85))
    assert wd.rating_code == 9
    assert wd.sigmoided


def test_wd_eva02_large_takes_a_white_padded_square_bicubic_bgr_nhwc_from_0_to_255():
    assert WD_EVA02_LARGE.preprocessing == Preprocessing(
        size=(448, 448),
        pad=WHITE,
        resample=Image.Resampling.BICUBIC,
        channels=BGR,
        layout=NHWC,
        scale=1.0,
        mean=0.0,
        std=1.0,
    )


def test_the_classifiers_preprocessing_is_imgutils_recipe_in_the_same_description():
    imgutils = Preprocessing(
        size=(384, 384),
        pad=None,
        resample=Image.Resampling.BILINEAR,
        channels=RGB,
        layout=NCHW,
        scale=255.0,
        mean=0.5,
        std=0.5,
    )
    assert ANIME_REAL.preprocessing == imgutils
    assert ANIME_CLASSIFICATION.preprocessing == imgutils


# Preprocessing.

#: The WD recipe on a 4×4 input, so a small synthetic image is resized to itself and each pixel can be looked at.
SMALL = replace(WD_EVA02_LARGE.preprocessing, size=(4, 4))
RED, BLUE = (255, 0, 0), (0, 0, 255)


def test_a_wide_image_is_centred_on_a_white_square_in_bgr_nhwc_unscaled():
    image = Image.new("RGB", (4, 2), RED)
    image.putpixel((0, 0), BLUE)

    pixels = to_input(image, SMALL)

    assert pixels.shape == (4, 4, 3)
    assert pixels.dtype == np.float32
    # Padded top and bottom, one row each, in white.
    assert (pixels[0] == 255).all()
    assert (pixels[3] == 255).all()
    # BGR: red is (0, 0, 255), blue (255, 0, 0); 0-255, not scaled.
    assert pixels[1, 0].tolist() == [255, 0, 0]
    assert pixels[1, 1].tolist() == [0, 0, 255]
    assert pixels[2].tolist() == [[0, 0, 255]] * 4


def test_a_tall_image_is_padded_left_and_right_the_odd_column_after_it():
    pixels = to_input(Image.new("RGB", (1, 4), RED), SMALL)

    # (4 - 1) // 2 = 1 column before it, 2 after, as the Space's `app.py` pastes it.
    assert pixels[:, 0].tolist() == [[255, 255, 255]] * 4
    assert pixels[:, 1].tolist() == [[0, 0, 255]] * 4
    assert pixels[:, 2:].tolist() == [[[255, 255, 255]] * 2] * 4


def test_the_classifiers_input_is_squashed_rgb_nchw_from_minus_1_to_1():
    pixels = to_input(Image.new("RGB", (8, 2), RED), ANIME_REAL.preprocessing)

    assert pixels.shape == (3, 384, 384)
    assert pixels[:, 0, 0].tolist() == [1.0, -1.0, -1.0]


# Thresholds, tags and the rating, from the scores.

LABELS = labels_from_csv(TAG_LIST)
#: The stand-in's tags' scores: ratings first, then 1girl, long_hair, outdoors, simple_background (general) and
#: hatsune_miku, kagamine_rin (character).
SCORES = np.array([0.1, 0.6, 0.2, 0.05, 0.99, 0.5, 0.35, 0.36, 0.9, 0.86], dtype=np.float32)


def test_the_tag_list_is_read_with_each_tag_s_category():
    assert LABELS == Labels(
        names=(*RATINGS, "1girl", "long_hair", "outdoors", "simple_background", "hatsune_miku", "kagamine_rin"),
        codes=(9, 9, 9, 9, 0, 0, 0, 0, 4, 4),
    )


def test_thresholds_left_empty_are_the_tagger_s_defaults():
    assert thresholds_of(WD_EVA02_LARGE, None) == {"general": 0.35, "character": 0.85}
    assert thresholds_of(WD_EVA02_LARGE, {}) == {"general": 0.35, "character": 0.85}


def test_a_threshold_set_overrides_only_its_own_category():
    assert thresholds_of(WD_EVA02_LARGE, {"character": 0.5}) == {"general": 0.35, "character": 0.5}
    assert thresholds_of(WD_EVA02_LARGE, {"general": 0}) == {"general": 0, "character": 0.85}


def test_a_threshold_for_a_category_the_tagger_lacks_fails_naming_its_categories():
    lacks = r"^wd-eva02-large has no category `copyright`: its categories are general, character$"
    with pytest.raises(ValueError, match=lacks):
        thresholds_of(WD_EVA02_LARGE, {"general": 0.3, "copyright": 0.5})


def test_tags_over_their_category_s_threshold_highest_first_names_unchanged():
    result = tag(WD_EVA02_LARGE, LABELS, SCORES, {"general": 0.35, "character": 0.85})

    assert [(t["name"], t["category"]) for t in result["tags"]] == [
        ("1girl", "general"),
        ("hatsune_miku", "character"),
        ("kagamine_rin", "character"),
        ("long_hair", "general"),
        ("simple_background", "general"),
    ]
    # Over, not at: outdoors' 0.35 is not over 0.35. Scores are plain floats, as JSON takes them.
    assert result["tags"][0] == {"name": "1girl", "score": pytest.approx(0.99), "category": "general"}
    assert all(type(t["score"]) is float for t in result["tags"])


def test_max_tags_keeps_the_highest():
    result = tag(WD_EVA02_LARGE, LABELS, SCORES, {"general": 0.35, "character": 0.85}, max_tags=2)

    assert [t["name"] for t in result["tags"]] == ["1girl", "hatsune_miku"]


def test_the_rating_is_the_highest_of_the_four_with_every_one_s_score():
    result = tag(WD_EVA02_LARGE, LABELS, SCORES, {"general": 0.35, "character": 0.85})

    assert result["rating"]["label"] == "sensitive"
    assert result["rating"]["scores"] == pytest.approx(
        {"general": 0.1, "sensitive": 0.6, "questionable": 0.2, "explicit": 0.05}
    )
    assert list(result["rating"]["scores"]) == RATINGS


def test_a_tagger_without_ratings_rates_nothing_and_tags_its_rating_rows_as_nothing():
    unrated = replace(WD_EVA02_LARGE, rating_code=None)

    result = tag(unrated, LABELS, SCORES, {"general": 0, "character": 0})

    assert result["rating"] is None
    assert {t["name"] for t in result["tags"]} == set(LABELS.names[4:])


def test_a_tagger_whose_output_is_not_sigmoided_is_sigmoided_first():
    logits = replace(WD_EVA02_LARGE, sigmoided=False)

    result = tag(logits, LABELS, np.zeros(10, dtype=np.float32), {"general": 0, "character": 0})

    assert {t["score"] for t in result["tags"]} == {0.5}


# The Worker, with the stand-in tagger.


def test_tags_an_image(tagger_hub, config):
    result = run(TagImage(config), tag_task())

    assert result["path"] == str(IMAGE)
    assert result["tagger"] == "stand-in"
    scores = [t["score"] for t in result["tags"]]
    assert scores == sorted(scores, reverse=True)
    for t in result["tags"]:
        assert set(t) == {"name", "score", "category"}
        assert t["score"] > {"general": 0.35, "character": 0.85}[t["category"]]
    assert result["rating"]["label"] in RATINGS
    assert result["rating"]["label"] == max(result["rating"]["scores"], key=result["rating"]["scores"].get)
    # `pnpm check` has the `cpu` build only.
    assert result["warnings"] == [ran_on_the_cpu(TAGGERS["stand-in"])]


def test_the_default_tagger_is_wd_eva02_large(tagger_hub, config, monkeypatch):
    # Its catalogue entry pointed at the stand-in's files, so nothing over 1 GB downloads.
    monkeypatch.setitem(TAGGERS, "wd-eva02-large", replace(TAGGERS["stand-in"], id="wd-eva02-large"))

    result = run(TagImage(config), Task({"path": str(IMAGE)}))

    assert result["tagger"] == "wd-eva02-large"


def test_thresholds_merge_over_the_tagger_s_defaults(tagger_hub, config):
    worker = TagImage(config)
    every = run(worker, tag_task(thresholds={"general": 0, "character": 0}))
    no_characters = run(worker, tag_task(thresholds={"character": 1}))
    default = run(worker, tag_task())

    assert len(every["tags"]) == 6
    assert [t for t in no_characters["tags"] if t["category"] == "character"] == []
    general = [t for t in default["tags"] if t["category"] == "general"]
    assert [t for t in no_characters["tags"] if t["category"] == "general"] == general


def test_a_threshold_for_a_category_the_tagger_lacks_fails_the_task_before_anything_downloads(tagger_hub, config):
    lacks = r"^stand-in has no category `copyright`: its categories are general, character"
    with pytest.raises(Exception, match=lacks):
        run(TagImage(config), tag_task(thresholds={"copyright": 0.3}))

    assert tagger_hub.requests == []


def test_max_tags_hands_on_the_highest_only(tagger_hub, config):
    worker = TagImage(config)
    every = run(worker, tag_task(thresholds={"general": 0, "character": 0}))

    result = run(worker, tag_task(thresholds={"general": 0, "character": 0}, maxTags=2))

    assert result["tags"] == every["tags"][:2]


def test_an_unknown_tagger_fails_naming_the_ones_there_are(tagger_hub, config):
    with pytest.raises(Exception, match=r"^No tagger `nope`: one of wd-eva02-large, stand-in$"):
        run(TagImage(config), tag_task(tagger="nope"))


def test_an_input_without_a_path_fails(tagger_hub, config):
    with pytest.raises(Exception, match=r"no `path`: the image to tag"):
        run(TagImage(config), Task({}, {"tagger": "stand-in"}))


def test_the_model_and_its_tag_list_are_downloaded_on_first_use_and_kept(tagger_hub, config, data_dir, stand_in):
    worker = TagImage(config)

    run(worker, tag_task())
    run(worker, tag_task())
    run(TagImage(config), tag_task())

    assert sorted(tagger_hub.requests) == sorted(f"/{p.remote_path}" for p in (stand_in, stand_in.tag_list))
    folder = data_dir / "models" / "tests" / "stand-in-tagger" / "0123456789ab"
    assert sorted(p.name for p in folder.iterdir()) == ["model.onnx", "selected_tags.csv"]


def test_a_tag_list_other_than_the_pinned_one_fails_the_task(tagger_hub, config, stand_in):
    file = served(tagger_hub, stand_in.tag_list)
    file.write_text(TAG_LIST + "7,extra,0,1\n")

    with pytest.raises(Exception, match=r"^Could not get the tag list of stand-in: .*sha256"):
        run(TagImage(config), tag_task())


def test_a_model_folder_holds_the_tagger_and_its_tag_list(stand_in, config, tmp_path):
    root = tmp_path / "offline-models"
    lay_out(stand_in, tmp_path / "stand-in", root, remote=False)
    config["modelDir"] = str(root)

    assert run(TagImage(config), tag_task())["tagger"] == "stand-in"

    (root / stand_in.tag_list.local_path).unlink()
    with pytest.raises(Exception, match=r"^Could not get the tag list of stand-in: it is not in the model folder"):
        run(TagImage(config), tag_task())


DOWNLOADING = "Downloading the model stand-in"


def test_a_task_whose_tagger_downloads_says_each_step_in_order(tagger_hub, config):
    ctx = Context()

    run(TagImage(config), tag_task(), ctx)

    messages = [message for _, message in ctx.progress_reports]
    # The tag list first, then the model, each said to its end.
    assert messages.index("Downloading the tag list of stand-in") < messages.index(DOWNLOADING)
    last = max(i for i, message in enumerate(messages) if message == DOWNLOADING)
    assert ctx.progress_reports[last:] == [
        (1.0, DOWNLOADING),
        (0.0, "Loading the model stand-in on the CPU"),
        (0.0, "Tagging"),
    ]


def test_a_task_whose_tagger_is_loaded_says_only_that_it_tags(tagger_hub, config):
    worker = TagImage(config)
    run(worker, tag_task())
    ctx = Context()

    run(worker, tag_task(), ctx)

    assert ctx.progress_reports == [(0.0, "Tagging")]


def test_tag_batch_size_is_a_start_up_field_four_by_default(tagger_hub, config):
    assert TagImage(config).batch_size == DEFAULT_TAG_BATCH_SIZE == 4
    config["batchSize"] = 16
    config["tagBatchSize"] = 2
    assert TagImage(config).batch_size == 2


def busy_tagger(config):
    """A Worker with the stand-in loaded, and a Task in a forward pass held busy: what arrives now waits."""
    worker = TagImage(config)
    run(worker, tag_task(FRAMES[0]))
    passes = ForwardPasses(worker)
    busy = Running(worker, tag_task(FRAMES[0]))
    wait_until(lambda: passes.batches, what="a first forward pass")
    lining_up()
    return worker, passes, busy


def test_tasks_at_once_share_batches_of_up_to_tag_batch_size_whatever_their_thresholds(tagger_hub, config):
    config["tagBatchSize"] = 2
    worker, passes, busy = busy_tagger(config)
    settings = [{"thresholds": {"general": 0}}, {"thresholds": {"character": 0.1}, "maxTags": 1}, {}, {}, {}]
    tasks = [tag_task(FRAMES[i + 1], **one) for i, one in enumerate(settings)]
    running = [Running(worker, one) for one in tasks]
    lining_up()

    passes.let_go()
    results = [one.result() for one in running]
    busy.result()

    assert passes.sizes("stand-in") == [1, 2, 2, 1]
    alone = TagImage(config)
    for got, one in zip(results, tasks, strict=True):
        want = run(alone, one)
        assert [t["name"] for t in got["tags"]] == [t["name"] for t in want["tags"]]
        assert got["rating"]["scores"] == pytest.approx(want["rating"]["scores"], abs=1e-5)


def test_tag_image_makes_room_for_as_many_tasks_at_once_as_its_pool_runs(stand_in, config):
    assert TagImage(config)._passes.blocking_threads == TAG_TASKS_AT_ONCE
    config["host"]["pool"] = {"maxConcurrentTasks": 3}
    assert TagImage(config)._passes.blocking_threads == 3


def test_teardown_drops_the_tagger_s_session_and_a_second_is_harmless(tagger_hub, config):
    worker = TagImage(config)
    run(worker, tag_task())
    assert set(worker.models._sessions) == {"stand-in"}

    worker.teardown()
    worker.teardown()

    assert worker.models._sessions == {}


# On a GPU, faked: a session that says it runs on CUDA.

OUT_OF_MEMORY = (
    "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running MatMul node. "
    "Name:'MatMul_1' Status Message: D:\\a\\_work\\1\\s\\onnxruntime\\core\\framework\\bfc_arena.cc:359 "
    "onnxruntime::BFCArena::AllocateRawInternal Failed to allocate memory for requested buffer of size 1073741824"
)
NEEDS_MORE = (
    "stand-in needs more GPU memory than this GPU has: use a smaller tagger, a smaller `tagBatchSize`, or the `cpu` "
    "build"
)


class FakeGpuSession:
    """A session made on CUDA, its forward pass `run_with` (raised when an exception, else the scores of each image)."""

    run_with = None
    made_with = None

    def __init__(self, path, options, providers, provider_options):
        if isinstance(FakeGpuSession.made_with, Exception):
            raise FakeGpuSession.made_with
        self._options = options

    def get_providers(self):
        return [devices.CUDA, devices.CPU]

    def get_session_options(self):
        return self._options

    def run(self, names, feeds):
        if isinstance(FakeGpuSession.run_with, Exception):
            raise FakeGpuSession.run_with
        return [np.tile(np.float32(0.9), (len(feeds["input"]), 10))]


@pytest.fixture
def on_a_gpu(monkeypatch):
    monkeypatch.setattr(ort, "get_available_providers", lambda: [devices.CUDA, devices.CPU])
    monkeypatch.setattr(devices, "prepare", lambda chosen: None)
    monkeypatch.setattr(models.ort, "InferenceSession", FakeGpuSession)
    FakeGpuSession.run_with = FakeGpuSession.made_with = None
    return FakeGpuSession


def test_on_a_gpu_there_is_no_cpu_warning(tagger_hub, config, on_a_gpu):
    result = run(TagImage(config), tag_task())

    assert "warnings" not in result
    assert len(result["tags"]) == 6


def test_a_forward_pass_out_of_gpu_memory_fails_the_task_saying_what_to_do(tagger_hub, config, on_a_gpu):
    on_a_gpu.run_with = RuntimeError(OUT_OF_MEMORY)

    with pytest.raises(Exception) as failed:
        run(TagImage(config), tag_task())

    assert str(failed.value) == NEEDS_MORE


def test_a_session_out_of_gpu_memory_fails_the_task_saying_what_to_do(tagger_hub, config, on_a_gpu):
    on_a_gpu.made_with = RuntimeError("CUDA failure 2: out of memory ; GPU=0 ; hostname=x ; expr=cudaMalloc(&p, size);")

    with pytest.raises(Exception) as failed:
        run(TagImage(config), tag_task())

    assert str(failed.value) == NEEDS_MORE


def test_another_failure_on_a_gpu_is_its_own(tagger_hub, config, on_a_gpu):
    on_a_gpu.run_with = RuntimeError("[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions")

    with pytest.raises(Exception, match="INVALID_ARGUMENT"):
        run(TagImage(config), tag_task())


def test_memory_that_runs_out_on_the_cpu_is_not_called_gpu_memory(stand_in, data_dir):
    session_models = Models(str(data_dir))

    class Failing:
        def get_providers(self):
            return [devices.CPU]

        def run(self, names, feeds):
            raise RuntimeError(OUT_OF_MEMORY)

    session_models._sessions[stand_in.id] = Failing()
    with pytest.raises(RuntimeError, match="Failed to allocate memory"):
        session_models.scores(stand_in, np.zeros((1, 448, 448, 3), dtype=np.float32))
