"""PixAI's results against its authors' pipeline (`fixtures/golden-pixai.json`, written by `scripts/golden_pixai.py`
with their PyTorch code): the reference of both `pixai` and `pixai-fp16`.

As `test_tag_image_golden.py`, it runs only when the tagger is in `python/.cache/models/`, and is marked slow. PixAI
takes some 20 s an image on a laptop's CPU, in fp32 or fp16 alike, so it runs on a few images chosen to cover what
differs: a tall frame, a wide (2.35:1) film frame, and a see-through image made white. `keyed.png` is never one: the
authors' pipeline ignores an RGB image's colour key, which the Plugin makes white.
"""

import json
import os

import pytest

from image_tags.catalogue import TAGGERS
from image_tags.worker import TagImage
from tests.conftest import FIXTURES, IMAGES, TAGGER_CACHE, Task, run
from tests.test_tag_image_golden import tagger_params as _tagger_params

GOLDEN = json.loads((FIXTURES / "golden-pixai.json").read_text(encoding="utf-8"))["pixai"]
#: Ticket 15's largest end-to-end differences from the authors' scores were 0.0023 (fp32) and 0.0049 (fp16), on CUDA;
#: about half that again, for other builds and machines. On the CPU, these images come within 1e-5 (fp32) and 0.0019
#: (fp16).
TOLERANCE = {"pixai": 0.004, "pixai-fp16": 0.008}
IMAGES_RUN = ["anime-07-yakitori-stall-portrait.jpg", "cg-02-sintel.jpg", "transparent.png"]

pytestmark = pytest.mark.slow


def tagger_params():
    return _tagger_params(["pixai", "pixai-fp16"])


@pytest.fixture(scope="module")
def workers(tmp_path_factory):
    """A Worker per tagger, reading the model folder, its forward pass on every processor (PixAI on 4 threads, a
    Worker's default, takes about twice as long): each tagger's session made once for the module."""
    made = {}

    def worker(tagger_id):
        if tagger_id not in made:
            data = tmp_path_factory.mktemp("data")
            pool = {"residentGroups": {"cpu": {"cost": os.cpu_count() or 4}}}
            made[tagger_id] = TagImage({"host": {"dataDir": str(data), "pool": pool}, "modelDir": str(TAGGER_CACHE)})
        return made[tagger_id]

    yield worker
    for one in made.values():
        one.teardown()


def test_keyed_png_is_never_held_to_the_reference():
    assert "keyed.png" in GOLDEN
    assert "keyed.png" not in IMAGES_RUN


@pytest.mark.parametrize("tagger_id", tagger_params())
def test_the_graph_and_tag_list_are_what_the_catalogue_says(workers, tagger_id):
    tagger = TAGGERS[tagger_id]
    worker = workers(tagger_id)
    session = worker.models.session(tagger)
    labels = worker.models.labels(tagger)

    (given,), (output,) = session.get_inputs(), session.get_outputs()
    assert (given.name, given.type, given.shape[1:]) == ("input", "tensor(float)", [3, 1008, 1008])
    assert (output.name, output.type, output.shape[1]) == (tagger.output, "tensor(float)", 30877)
    counts = {code: labels.codes.count(code) for code in set(labels.codes)}
    assert counts == {0: 15043, 4: 8308, 3: 2460, 1: 4917, 5: 145, 9: 4}
    assert labels.names[-4:] == ("rating:s", "rating:g", "rating:q", "rating:e")


@pytest.mark.parametrize("image", IMAGES_RUN)
@pytest.mark.parametrize("tagger_id", tagger_params())
def test_tags_as_the_authors_pipeline_does(workers, tagger_id, image):
    tagger = TAGGERS[tagger_id]
    expected = GOLDEN[image]
    every_tag = {category.name: 0 for category in tagger.categories}
    task = Task({"path": str(IMAGES / image)}, {"tagger": tagger_id, "thresholds": every_tag})

    result = run(workers(tagger_id), task, timeout=300)

    scores = {t["name"]: t["score"] for t in result["tags"]}
    tolerance = TOLERANCE[tagger_id]
    assert {name: scores.get(name) for name in expected["tags"]} == pytest.approx(expected["tags"], abs=tolerance)
    assert result["tags"][0]["name"] == next(iter(expected["tags"]))
    ratings = dict(tagger.rating_names)
    want = {ratings[name]: score for name, score in expected["rating"].items()}
    assert result["rating"]["scores"] == pytest.approx(want, abs=tolerance)
    assert result["rating"]["label"] == max(want, key=want.get)

