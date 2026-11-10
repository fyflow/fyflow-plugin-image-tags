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


def test_tells_anime_cg_or_live_action_by_default(hub, config):
    result = classify(config, IMAGES / "cartoon.png")

    assert result["path"] == str(IMAGES / "cartoon.png")
    assert result["classifier"] == "style"
    assert result["labels"] == ["anime", "cg", "live_action"]
    assert result["tag"] == "anime"
    assert result["confidence"] == result["scores"]["anime"]
    assert result["confidence"] >= 0.6
    assert sum(result["scores"].values()) == pytest.approx(1, abs=1e-5)
    assert result["models"] == ["anime_real", "anime_classification"]
    assert "warnings" not in result


def test_style_is_its_rule_over_both_models_scores(hub, config):
    # cartoon.png's golden scores, through the rule by hand: what is not real, shared by the four painted kinds.
    real = GOLDEN["anime_real"]["cartoon.png"]["real"]
    kinds = GOLDEN["anime_classification"]["cartoon.png"]
    painted = kinds["3d"] + kinds["bangumi"] + kinds["comic"] + kinds["illustration"]

    result = classify(config, IMAGES / "cartoon.png", classifier="style")

    assert result["scores"] == pytest.approx(
        {
            "live_action": real,
            "cg": (1 - real) * kinds["3d"] / painted,
            "anime": (1 - real) * (kinds["bangumi"] + kinds["comic"] + kinds["illustration"]) / painted,
        },
        abs=0.02,
    )


@pytest.mark.parametrize("model", list(MODELS.values()), ids=list(MODELS))
def test_each_model_is_a_classifier_with_its_own_labels(hub, config, model):
    result = classify(config, IMAGES / "cartoon.png", classifier=model.id)

    assert result["classifier"] == model.id
    assert result["labels"] == list(model.labels)
    assert list(result["scores"]) == list(model.labels)
    assert result["tag"] == max(result["scores"], key=result["scores"].get)
    assert result["confidence"] == result["scores"][result["tag"]]
    assert result["models"] == [model.id]


def test_a_raw_model_answers_its_top_label_however_low(hub, config):
    # photo.jpg is about 0.6 real to anime_real: about what `style` asks for, but a raw model asks for nothing.
    result = classify(config, IMAGES / "photo.jpg", classifier="anime_real")

    assert result["confidence"] < 0.7
    assert result["tag"] == "real"


def test_a_top_score_below_min_confidence_is_uncertain_and_keeps_its_scores(hub, config):
    sure = classify(config, IMAGES / "cartoon.png")

    result = classify(config, IMAGES / "cartoon.png", minConfidence=0.99)

    assert result["tag"] == "uncertain"
    assert result["confidence"] == sure["confidence"]
    assert result["scores"] == sure["scores"]
    assert result["labels"] == ["anime", "cg", "live_action"]


def test_min_confidence_holds_for_a_raw_model_too(hub, config):
    result = classify(config, IMAGES / "cartoon.png", classifier="anime_real", minConfidence=0.99)

    assert result["tag"] == "uncertain"
    assert result["confidence"] == result["scores"]["anime"]


def test_an_unknown_classifier_fails_naming_the_ones_there_are(hub, config):
    with pytest.raises(Exception, match=r"^No classifier `nope`: one of .*style.*anime_real.*anime_classification"):
        classify(config, IMAGES / "cartoon.png", classifier="nope")


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


def test_the_models_are_downloaded_on_first_use_and_kept(hub, config, data_dir):
    worker = ClassifyImage(config)

    worker.run(Task({"path": str(IMAGES / "cartoon.png")}), Context())
    worker.run(Task({"path": str(IMAGES / "photo.jpg")}), Context())
    ClassifyImage(config).run(Task({"path": str(IMAGES / "tiny.jpg")}), Context())

    # The default classifier, `style`, needs both: each downloaded once, the first time.
    assert hub.requests == [f"/{MODELS[id].remote_path}" for id in ("anime_real", "anime_classification")]
    for model in MODELS.values():
        stored = data_dir / "models" / model.repo / model.commit[:12] / model.name / model.file
        assert stored.is_file()
        assert [p.name for p in stored.parent.iterdir()] == [model.file]


def test_a_model_other_than_the_pinned_one_fails_the_task_and_is_not_kept(hub, config, data_dir):
    file = served(hub, MODELS["anime_real"])
    file.write_bytes(file.read_bytes()[:-1] + b"\0")

    with pytest.raises(Exception, match=r"^Could not get the model anime_real: .*sha256"):
        classify(config, IMAGES / "cartoon.png", classifier="anime_real")

    assert not any(path.is_file() for path in (data_dir / "models").rglob("*"))


def test_a_model_that_cannot_be_downloaded_fails_the_task(hub, config):
    served(hub, MODELS["anime_real"]).unlink()

    with pytest.raises(Exception, match=r"^Could not get the model anime_real: .*404"):
        classify(config, IMAGES / "cartoon.png", classifier="anime_real")


def test_a_model_that_cannot_be_written_fails_the_task_as_one_it_cannot_get(hub, config, data_dir):
    (data_dir / "models").write_text("a file where the models' folder goes")

    with pytest.raises(Exception, match=r"^Could not get the model anime_real: "):
        classify(config, IMAGES / "cartoon.png", classifier="anime_real")


def test_the_next_task_tries_the_download_again(hub, config):
    file = served(hub, MODELS["anime_real"])
    good = file.read_bytes()
    file.write_bytes(b"not a model")
    worker = ClassifyImage(config)
    task = Task({"path": str(IMAGES / "cartoon.png")}, {"classifier": "anime_real"})
    with pytest.raises(Exception, match="Could not get the model"):
        worker.run(task, Context())

    file.write_bytes(good)

    assert worker.run(task, Context())["tag"] == "anime"


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
