"""`ClassifyImage` as the Host runs it: one image per Task, through one catalogue classifier."""

import json

import numpy as np
import pytest

from image_tags.catalogue import MODELS
from image_tags.models import Models
from image_tags.preprocess import load_image, to_input
from image_tags.worker import ClassifyImage
from tests.conftest import FIXTURES, IMAGES, Context, Task, served

GOLDEN = json.loads((FIXTURES / "golden.json").read_text(encoding="utf-8"))


def classify(config, path, **settings):
    return ClassifyImage(config).run(Task({"path": str(path)}, settings), Context())


def test_tells_what_an_image_is_with_the_default_classifier(hub, config):
    result = classify(config, IMAGES / "cartoon.png")

    assert result["path"] == str(IMAGES / "cartoon.png")
    assert result["classifier"] == "anime_real"
    assert result["labels"] == ["anime", "real"]
    assert result["tag"] == "anime"
    assert result["confidence"] == result["scores"]["anime"]
    assert result["scores"]["anime"] + result["scores"]["real"] == pytest.approx(1, abs=1e-5)
    assert result["models"] == ["anime_real"]
    assert "warnings" not in result


@pytest.mark.parametrize("model", list(GOLDEN))
@pytest.mark.parametrize("image", sorted(next(iter(GOLDEN.values()))))
def test_scores_as_imgutils_does(hub, config, model, image):
    expected = GOLDEN[model][image]

    result = classify(config, IMAGES / image, classifier=model)

    assert result["tag"] == max(expected, key=expected.get)
    assert result["scores"] == pytest.approx(expected, abs=0.02)


@pytest.mark.parametrize("model", list(MODELS.values()), ids=list(MODELS))
def test_images_scored_in_one_batch_score_as_they_do_alone(hub, data_dir, model):
    models = Models(str(data_dir))
    inputs = [to_input(load_image(str(IMAGES / name)), model.size) for name in ("cartoon.png", "photo.jpg")]

    together = models.scores(model, np.stack(inputs))
    alone = [models.scores(model, one[np.newaxis])[0] for one in inputs]

    assert together.shape == (2, len(model.labels))
    assert together == pytest.approx(np.stack(alone), abs=1e-5)


def test_a_model_is_downloaded_on_first_use_and_kept(hub, config, data_dir):
    model = MODELS["anime_real"]
    worker = ClassifyImage(config)

    worker.run(Task({"path": str(IMAGES / "cartoon.png")}), Context())
    worker.run(Task({"path": str(IMAGES / "photo.jpg")}), Context())
    ClassifyImage(config).run(Task({"path": str(IMAGES / "tiny.jpg")}), Context())

    stored = data_dir / "models" / model.repo / model.commit[:12] / model.name / model.file
    assert stored.is_file()
    assert hub.requests == [f"/{model.repo}/resolve/{model.commit}/{model.name}/{model.file}"]
    assert [p.name for p in stored.parent.iterdir()] == [model.file]


def test_a_model_other_than_the_pinned_one_fails_the_task_and_is_not_kept(hub, config, data_dir):
    file = served(hub, MODELS["anime_real"])
    file.write_bytes(file.read_bytes()[:-1] + b"\0")

    with pytest.raises(Exception, match=r"^Could not get the model anime_real: .*sha256"):
        classify(config, IMAGES / "cartoon.png")

    assert not any(path.is_file() for path in (data_dir / "models").rglob("*"))


def test_a_model_that_cannot_be_downloaded_fails_the_task(hub, config):
    served(hub, MODELS["anime_real"]).unlink()

    with pytest.raises(Exception, match=r"^Could not get the model anime_real: .*404"):
        classify(config, IMAGES / "cartoon.png")


def test_a_model_that_cannot_be_written_fails_the_task_as_one_it_cannot_get(hub, config, data_dir):
    (data_dir / "models").write_text("a file where the models' folder goes")

    with pytest.raises(Exception, match=r"^Could not get the model anime_real: "):
        classify(config, IMAGES / "cartoon.png")


def test_the_next_task_tries_the_download_again(hub, config):
    file = served(hub, MODELS["anime_real"])
    good = file.read_bytes()
    file.write_bytes(b"not a model")
    worker = ClassifyImage(config)
    with pytest.raises(Exception, match="Could not get the model"):
        worker.run(Task({"path": str(IMAGES / "cartoon.png")}), Context())

    file.write_bytes(good)

    assert worker.run(Task({"path": str(IMAGES / "cartoon.png")}), Context())["tag"] == "anime"


def test_a_file_that_is_not_an_image_fails_naming_it(hub, config, tmp_path):
    path = tmp_path / "notes.jpg"
    path.write_text("not an image")

    with pytest.raises(Exception, match=r"^Not an image .*notes\.jpg"):
        classify(config, path)


def test_a_missing_file_fails_naming_it(hub, config, tmp_path):
    path = tmp_path / "gone.png"

    with pytest.raises(Exception, match=r"^Cannot read the image .*gone\.png"):
        classify(config, path)


def test_an_input_without_a_path_fails(hub, config):
    with pytest.raises(Exception, match="no `path`"):
        ClassifyImage(config).run(Task({}), Context())
