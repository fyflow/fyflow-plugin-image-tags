"""How `ClassifyImage` gets its models under real use: Tasks at once, Cancel, a bad file, and an offline machine."""

import pytest

from image_tags.catalogue import MODELS
from image_tags.worker import ClassifyImage
from tests.conftest import IMAGES, Context, Running, Task, served, wait_until

MODEL = MODELS["anime_real"]
MODEL_URL = f"/{MODEL.remote_path}"


def image_task():
    return Task({"path": str(IMAGES / "cartoon.png")})


def downloading(running):
    """Whether the Task has said it is downloading: it waits on the download."""
    return bool(running.ctx.progress_reports)


def files_under(folder):
    return [path for path in folder.rglob("*") if path.is_file()] if folder.exists() else []


def test_tasks_that_need_a_model_at_once_share_one_download(hub, config):
    hub.hold()
    worker = ClassifyImage(config)
    tasks = [Running(worker, image_task()) for _ in range(3)]
    for task in tasks:
        wait_until(lambda task=task: downloading(task), what="a progress report")

    hub.release()

    assert [task.result()["tag"] for task in tasks] == ["anime"] * 3
    assert hub.requests == [MODEL_URL]


def test_a_download_reports_its_progress_through_the_task(hub, config):
    hub.hold()
    task = Running(ClassifyImage(config), image_task())
    wait_until(lambda: downloading(task), what="a progress report")

    hub.release()
    task.result()

    fractions = [fraction for fraction, _ in task.ctx.progress_reports]
    assert 0 < fractions[0] < 1
    assert fractions == sorted(fractions)
    assert all(0 < fraction <= 1 for fraction in fractions)
    assert {message for _, message in task.ctx.progress_reports} == {"Downloading the model anime_real"}


def test_a_cancelled_task_stops_at_once_and_its_download_leaves_no_file(hub, config, data_dir):
    hub.hold()
    task = Running(ClassifyImage(config), image_task())
    wait_until(lambda: downloading(task), what="a progress report")

    task.cancel()

    with pytest.raises(BaseException):  # noqa: B017, PT011  (whatever it raises, the shim reports "Cancelled")
        task.result(timeout=1)
    hub.release()
    wait_until(lambda: files_under(data_dir / "models") == [], what="the download's file going")


def test_a_cancelled_task_leaves_the_download_to_the_tasks_still_waiting(hub, config):
    hub.hold()
    worker = ClassifyImage(config)
    cancelled, waiting = Running(worker, image_task()), Running(worker, image_task())
    for task in (cancelled, waiting):
        wait_until(lambda task=task: downloading(task), what="a progress report")

    cancelled.cancel()
    with pytest.raises(BaseException):  # noqa: B017, PT011
        cancelled.result(timeout=1)
    hub.release()

    assert waiting.result()["tag"] == "anime"
    assert hub.requests == [MODEL_URL]


def test_after_a_cancelled_download_the_next_task_downloads_again(hub, config):
    hub.hold()
    worker = ClassifyImage(config)
    task = Running(worker, image_task())
    wait_until(lambda: downloading(task), what="a progress report")
    task.cancel()
    with pytest.raises(BaseException):  # noqa: B017, PT011
        task.result(timeout=1)

    hub.release()

    assert worker.run(image_task(), Context())["tag"] == "anime"
    assert hub.requests == [MODEL_URL, MODEL_URL]


def test_a_model_other_than_the_pinned_one_fails_every_task_waiting_and_leaves_no_file(hub, config, data_dir):
    file = served(hub, MODEL)
    file.write_bytes(file.read_bytes()[:-1] + b"\0")
    hub.hold()
    worker = ClassifyImage(config)
    tasks = [Running(worker, image_task()) for _ in range(2)]
    for task in tasks:
        wait_until(lambda task=task: downloading(task), what="a progress report")

    hub.release()

    for task in tasks:
        with pytest.raises(Exception, match=r"^Could not get the model anime_real: .*sha256"):
            task.result()
    assert files_under(data_dir / "models") == []
    assert hub.requests == [MODEL_URL]


def test_a_model_dir_is_used_and_nothing_is_downloaded(hub, config, model_dir, data_dir):
    config["modelDir"] = str(model_dir)

    result = ClassifyImage(config).run(image_task(), Context())

    assert result["tag"] == "anime"
    assert hub.requests == []
    assert files_under(data_dir / "models") == []


def test_a_model_missing_from_the_model_dir_fails_naming_the_folder_and_downloads_nothing(hub, config, model_dir):
    (model_dir / MODEL.local_path).unlink()
    config["modelDir"] = str(model_dir)

    with pytest.raises(Exception, match=r"^Could not get the model anime_real: ") as failure:
        ClassifyImage(config).run(image_task(), Context())

    assert str(model_dir) in str(failure.value)
    assert str(model_dir / MODEL.local_path) in str(failure.value)
    assert hub.requests == []


def test_a_model_in_the_model_dir_other_than_the_pinned_one_fails_the_task(hub, config, model_dir):
    file = model_dir / MODEL.local_path
    file.write_bytes(file.read_bytes()[:-1] + b"\0")
    config["modelDir"] = str(model_dir)

    with pytest.raises(Exception, match=r"^Could not get the model anime_real: .*sha256"):
        ClassifyImage(config).run(image_task(), Context())
    assert hub.requests == []
