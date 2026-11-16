"""A Worker that ends (Reload, its pool idling out, the Host stopping) drops its models' sessions, so their memory, on
the GPU too, is freed with them; the shim may call `teardown` twice (its message, then stdin closing)."""

import pytest

from image_tags.catalogue import MODELS
from image_tags.models import Models
from image_tags.worker import ClassifyFrames, ClassifyImage
from tests.conftest import IMAGES, Task, run

IMAGE = IMAGES / "anime-02-train-window.jpg"


def style_task():
    """A Task through `style`, so its Worker holds a session of each model."""
    return Task({"path": str(IMAGE)}, {"classifier": "style"})


@pytest.mark.parametrize("Worker", [ClassifyImage, ClassifyFrames])
def test_teardown_drops_every_session_of_either_worker(hub, config, Worker):
    worker = Worker(config)
    run(worker, style_task())
    assert set(worker.models._sessions) == {"anime_real", "anime_classification"}

    worker.teardown()

    assert worker.models._sessions == {}


@pytest.mark.parametrize("Worker", [ClassifyImage, ClassifyFrames])
def test_a_second_teardown_is_harmless(hub, config, Worker):
    worker = Worker(config)
    run(worker, style_task())

    worker.teardown()
    worker.teardown()

    assert worker.models._sessions == {}


def test_a_worker_torn_down_before_any_task_tears_down_cleanly(config):
    for worker in (ClassifyImage(config), ClassifyFrames(config)):
        worker.teardown()
        worker.teardown()


def test_a_session_asked_for_after_close_is_made_again(hub, data_dir):
    models = Models(str(data_dir))
    model = MODELS["anime_real"]
    before = models.session(model)

    models.close()
    models.close()
    after = models.session(model)

    assert after is not before
    assert models.session(model) is after
