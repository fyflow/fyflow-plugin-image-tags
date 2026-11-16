"""What a Task says as it goes, so that its progress never outlives the step it names: a download says it has ended,
loading a model says so and where, and classifying says so before its first forward pass (which may be slow: cuDNN
searches for its algorithms then)."""

import pytest

from image_tags.catalogue import MODELS
from image_tags.devices import CPU, CUDA, DIRECTML, name_of
from image_tags.models import Models
from image_tags.worker import ClassifyFrames, ClassifyImage
from tests.conftest import IMAGES, Context, Task, run, served

MODEL = MODELS["anime_real"]
DOWNLOADING = "Downloading the model anime_real"
#: `pnpm check` has the `cpu` build only.
LOADING = "Loading the model anime_real on the CPU"


def frames_task(tmp_path, count=3):
    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(count):
        (folder / f"{i:03}.jpg").write_bytes((IMAGES / "tiny.jpg").read_bytes())
    return Task({"path": str(folder)}, {"classifier": MODEL.id})


def image_task():
    return Task({"path": str(IMAGES / "anime-02-train-window.jpg")}, {"classifier": MODEL.id})


def after_download(ctx):
    """The last download report, and what the Task said after it."""
    reports = ctx.progress_reports
    last = max(i for i, (_, message) in enumerate(reports) if message == DOWNLOADING)
    return reports[last:]


def test_a_classify_frames_task_whose_model_downloads_says_each_step_in_order(hub, config, tmp_path):
    ctx = Context()

    run(ClassifyFrames(config), frames_task(tmp_path), ctx)

    assert after_download(ctx) == [
        (1.0, DOWNLOADING),
        (0.0, LOADING),
        (0.0, "Classifying 0 of 3 frames"),
        (1.0, "Classified 3 of 3 frames"),
    ]


def test_a_classify_image_task_whose_model_downloads_says_each_step_in_order(hub, config):
    ctx = Context()

    run(ClassifyImage(config), image_task(), ctx)

    assert after_download(ctx) == [(1.0, DOWNLOADING), (0.0, LOADING), (0.0, "Classifying")]


def test_a_classify_frames_task_whose_model_is_loaded_says_no_loading(hub, config, tmp_path):
    worker = ClassifyFrames(config)
    task = frames_task(tmp_path)
    run(worker, task)
    ctx = Context()

    run(worker, task, ctx)

    assert ctx.progress_reports == [(0.0, "Classifying 0 of 3 frames"), (1.0, "Classified 3 of 3 frames")]


def test_a_classify_image_task_whose_model_is_loaded_says_no_loading(hub, config):
    worker = ClassifyImage(config)
    run(worker, image_task())
    ctx = Context()

    run(worker, image_task(), ctx)

    assert ctx.progress_reports == [(0.0, "Classifying")]


def test_a_failed_download_never_says_it_ended(hub, config):
    file = served(hub, MODEL)
    file.write_bytes(file.read_bytes()[:-1] + b"\0")
    ctx = Context()

    with pytest.raises(Exception, match="sha256"):
        run(ClassifyImage(config), image_task(), ctx)

    assert (1.0, DOWNLOADING) not in ctx.progress_reports
    assert all(message == DOWNLOADING for _, message in ctx.progress_reports)


def test_a_model_already_on_disk_says_only_that_it_loads(config, model_dir):
    config["modelDir"] = str(model_dir)
    ctx = Context()

    run(ClassifyImage(config), image_task(), ctx)

    assert ctx.progress_reports == [(0.0, LOADING), (0.0, "Classifying")]


def test_a_session_made_without_a_task_says_nothing(data_dir, model_dir):
    # `ctx` is `None` for a caller without a Task (the eval script, the batcher).
    session = Models(str(data_dir), model_dir=str(model_dir)).session(MODEL)

    assert session is not None


@pytest.mark.parametrize(
    ("provider", "name"),
    [(CUDA, "CUDA"), (DIRECTML, "DirectML"), (CPU, "the CPU")],
)
def test_the_provider_a_session_loads_on_is_named_for_people(provider, name):
    assert name_of([(provider, {}), (CPU, {})]) == name
