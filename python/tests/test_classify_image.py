"""`ClassifyImage` as the Host runs it: one image per Task, through one catalogue classifier."""

import json

import numpy as np
import pytest

from image_tags.catalogue import MODELS
from image_tags.models import Models
from image_tags.preprocess import load_image, to_input
from image_tags.worker import CPU_UNITS, TASKS_AT_ONCE, ClassifyFrames, ClassifyImage
from tests.conftest import FIXTURES, IMAGES, Pool, Task, run, served

GOLDEN = json.loads((FIXTURES / "golden.json").read_text(encoding="utf-8"))
#: Reference frames (see `fixtures/images/CREDITS.md`): one every classifier gets right with room to spare, a
#: live-action one, and one `style` is torn on (a CG film: anime_real fairly sure it is real, anime_classification of
#: the painted kinds sure it is 3d).
ANIME = IMAGES / "anime-02-train-window.jpg"
LIVE_ACTION = IMAGES / "live_action-02-bristol-street.jpg"
TORN = IMAGES / "cg-02-sintel.jpg"


def classify(config, path, **settings):
    return run(ClassifyImage(config), Task({"path": str(path)}, settings))


def test_tells_anime_cg_or_live_action_by_default(hub, config):
    result = classify(config, ANIME)

    assert result["path"] == str(ANIME)
    assert result["classifier"] == "style"
    assert result["labels"] == ["anime", "cg", "live_action"]
    assert result["tag"] == "anime"
    assert result["confidence"] == result["scores"]["anime"]
    assert result["confidence"] >= 0.55
    assert sum(result["scores"].values()) == pytest.approx(1, abs=1e-5)
    assert result["models"] == ["anime_real", "anime_classification"]
    assert "warnings" not in result


@pytest.mark.parametrize(
    "image",
    [
        "anime-01-rainy-bus-stop.jpg",
        "anime-02-train-window.jpg",
        "anime-03-festival-night-market.jpg",
        "anime-04-seaside-railway.jpg",
        "anime-05-bakery-interior.jpg",
        "anime-06-riverside-picnic.jpg",
        "anime-07-yakitori-stall-portrait.jpg",
        "live_action-01-office-interview.jpg",
        "live_action-02-bristol-street.jpg",
        "live_action-03-redwood-creek.jpg",
        "live_action-04-dim-kitchen-ghana.jpg",
        "live_action-05-oral-history-interview.jpg",
    ],
)
def test_style_tags_a_clear_reference_frame_as_its_category(hub, config, image):
    # The CG films are left out: `style` takes some of them for live action (a known weakness of the models).
    result = classify(config, IMAGES / image)

    assert result["tag"] == image.split("-")[0]


def test_style_is_its_rule_over_both_models_scores(hub, config):
    # A frame that gets some of all three: its golden scores through the rule by hand. Live action is real^4; what is
    # not, shared by the four painted kinds (3d .58, illustration .41).
    frame = IMAGES / "cg-03-elephants-dream.jpg"
    live = GOLDEN["anime_real"][frame.name]["real"] ** 4
    kinds = GOLDEN["anime_classification"][frame.name]
    painted = kinds["3d"] + kinds["bangumi"] + kinds["comic"] + kinds["illustration"]

    result = classify(config, frame, classifier="style")

    assert result["scores"] == pytest.approx(
        {
            "live_action": live,
            "cg": (1 - live) * kinds["3d"] / painted,
            "anime": (1 - live) * (kinds["bangumi"] + kinds["comic"] + kinds["illustration"]) / painted,
        },
        abs=0.02,
    )


@pytest.mark.parametrize("model", list(MODELS.values()), ids=list(MODELS))
def test_each_model_is_a_classifier_with_its_own_labels(hub, config, model):
    result = classify(config, ANIME, classifier=model.id)

    assert result["classifier"] == model.id
    assert result["labels"] == list(model.labels)
    assert list(result["scores"]) == list(model.labels)
    assert result["tag"] == max(result["scores"], key=result["scores"].get)
    assert result["confidence"] == result["scores"][result["tag"]]
    assert result["models"] == [model.id]


def test_a_raw_model_answers_its_top_label_however_low(hub, config):
    # anime_classification is torn on this frame: bangumi .51, illustration .38. Below what `style` asks for, but a
    # raw model asks for nothing.
    result = classify(config, IMAGES / "anime-04-seaside-railway.jpg", classifier="anime_classification")

    assert result["confidence"] < 0.55
    assert result["tag"] == "bangumi"


def test_a_top_score_below_min_confidence_is_uncertain_and_keeps_its_scores(hub, config):
    # No reference frame is uncertain under `style`'s own threshold; this one is torn: anime_real is .79 sure it is
    # real (.39 live action), anime_classification sure the rest is 3d: cg .60, just over the default.
    sure = classify(config, TORN)

    result = classify(config, TORN, minConfidence=0.7)

    assert sure["tag"] == "cg"
    assert 0.55 <= sure["confidence"] < 0.7
    assert 0.3 < sure["scores"]["live_action"] < 0.5
    assert result["tag"] == "uncertain"
    assert result["confidence"] == sure["confidence"]
    assert result["scores"] == sure["scores"]
    assert result["labels"] == ["anime", "cg", "live_action"]


def test_min_confidence_holds_for_a_raw_model_too(hub, config):
    result = classify(config, ANIME, classifier="anime_real", minConfidence=0.99)

    assert result["tag"] == "uncertain"
    assert result["confidence"] == result["scores"]["anime"]


def test_an_unknown_classifier_fails_naming_the_ones_there_are(hub, config):
    with pytest.raises(Exception, match=r"^No classifier `nope`: one of .*style.*anime_real.*anime_classification"):
        classify(config, ANIME, classifier="nope")


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
    inputs = [to_input(load_image(str(name)), model.size) for name in (ANIME, LIVE_ACTION)]

    together = models.scores(model, np.stack(inputs))
    alone = [models.scores(model, one[np.newaxis])[0] for one in inputs]

    assert together.shape == (2, len(model.labels))
    assert together == pytest.approx(np.stack(alone), abs=1e-5)


def test_the_models_are_downloaded_on_first_use_and_kept(hub, config, data_dir):
    worker = ClassifyImage(config)

    run(worker, Task({"path": str(ANIME)}))
    run(worker, Task({"path": str(LIVE_ACTION)}))
    run(ClassifyImage(config), Task({"path": str(IMAGES / "tiny.jpg")}))

    # The default classifier, `style`, needs both: each downloaded once, the first time (both at once).
    assert sorted(hub.requests) == sorted(f"/{model.remote_path}" for model in MODELS.values())
    for model in MODELS.values():
        stored = data_dir / "models" / model.repo / model.commit[:12] / model.name / model.file
        assert stored.is_file()
        assert [p.name for p in stored.parent.iterdir()] == [model.file]


def test_a_model_other_than_the_pinned_one_fails_the_task_and_is_not_kept(hub, config, data_dir):
    file = served(hub, MODELS["anime_real"])
    file.write_bytes(file.read_bytes()[:-1] + b"\0")

    with pytest.raises(Exception, match=r"^Could not get the model anime_real: .*sha256"):
        classify(config, ANIME, classifier="anime_real")

    assert not any(path.is_file() for path in (data_dir / "models").rglob("*"))


def test_a_model_that_cannot_be_downloaded_fails_the_task(hub, config):
    served(hub, MODELS["anime_real"]).unlink()

    with pytest.raises(Exception, match=r"^Could not get the model anime_real: .*404"):
        classify(config, ANIME, classifier="anime_real")


def test_a_model_that_cannot_be_written_fails_the_task_as_one_it_cannot_get(hub, config, data_dir):
    (data_dir / "models").write_text("a file where the models' folder goes")

    with pytest.raises(Exception, match=r"^Could not get the model anime_real: "):
        classify(config, ANIME, classifier="anime_real")


def test_the_next_task_tries_the_download_again(hub, config):
    file = served(hub, MODELS["anime_real"])
    good = file.read_bytes()
    file.write_bytes(b"not a model")
    worker = ClassifyImage(config)
    task = Task({"path": str(ANIME)}, {"classifier": "anime_real"})
    with pytest.raises(Exception, match="Could not get the model"):
        run(worker, task)

    file.write_bytes(good)

    assert run(worker, task)["tag"] == "anime"


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
        run(ClassifyImage(config), Task({}))


def forward_pass_threads(worker):
    run(worker, Task({"path": str(ANIME)}, {"classifier": "anime_real"}))
    return worker.models.session(MODELS["anime_real"]).get_session_options().intra_op_num_threads


def with_pool(config, **pool):
    """`config` as a Host on SDK 0.11 gives it: with the pool options the Worker runs under, at `host.pool`."""
    return {**config, "host": {**config["host"], "pool": pool}}


@pytest.mark.parametrize("worker_class", [ClassifyImage, ClassifyFrames])
def test_a_forward_pass_runs_on_as_many_threads_as_the_cpu_units_the_process_holds(hub, config, worker_class):
    booked = with_pool(config, residentGroups={"cpu": {"cost": 6}, "gpu": {"cost": 1}})
    assert forward_pass_threads(worker_class(booked, Pool({"gpu": [0]}))) == 6


@pytest.mark.parametrize(
    "pool",
    [None, {}, {"maxThreads": 1}, {"residentGroups": {}}, {"residentGroups": {"gpu": {"cost": 1}}}],
)
def test_a_forward_pass_runs_on_cpu_units_threads_when_the_host_does_not_say(hub, config, pool):
    # `None`: a Host before 0.11, which gives no `host.pool`.
    given = config if pool is None else with_pool(config, **pool)
    assert forward_pass_threads(ClassifyImage(given, Pool({"gpu": [0]}))) == CPU_UNITS


def blocking_threads(worker):
    return worker._blocking._max_workers


def test_classify_image_makes_room_for_as_many_tasks_at_once_as_its_pool_runs(hub, config):
    worker = ClassifyImage(with_pool(config, maxConcurrentTasks=8))
    assert blocking_threads(worker) == 8 * len(MODELS)
    worker.teardown()


@pytest.mark.parametrize("pool", [None, {}, {"maxThreads": 1}])
def test_classify_image_makes_room_for_tasks_at_once_tasks_when_the_host_does_not_say(hub, config, pool):
    worker = ClassifyImage(config if pool is None else with_pool(config, **pool))
    assert blocking_threads(worker) == TASKS_AT_ONCE * len(MODELS)
    worker.teardown()
