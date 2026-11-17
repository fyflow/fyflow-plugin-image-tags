"""The real taggers' results against their reference pipelines' (`fixtures/golden-tags.json`, written by
`scripts/golden_tags.py`).

A tagger is over 1 GB, so the tests never download one: they run only when it is in `python/.cache/models/` (a
`modelDir`; see the README's Development), and are marked slow.
"""

import json

import pytest

from image_tags.catalogue import TAGGERS
from image_tags.worker import TagImage
from tests.conftest import FIXTURES, IMAGES, TAGGER_CACHE, Task, run

GOLDEN = json.loads((FIXTURES / "golden-tags.json").read_text(encoding="utf-8"))
#: How far a score may be from the reference's: the same steps on the same CPU differ in the last bits; a GPU more.
TOLERANCE = 0.01


def on_disk(tagger_id):
    tagger = TAGGERS[tagger_id]
    return all((TAGGER_CACHE / pinned.local_path).is_file() for pinned in (tagger, tagger.tag_list))


def tagger_params():
    return [
        pytest.param(
            tagger_id,
            marks=pytest.mark.skipif(
                not on_disk(tagger_id), reason=f"{tagger_id} is not in {TAGGER_CACHE} (it is never downloaded by tests)"
            ),
        )
        for tagger_id in GOLDEN
    ]


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def workers(tmp_path_factory):
    """A Worker per tagger, reading the model folder: each tagger's session made once for the module."""
    made = {}

    def worker(tagger_id):
        if tagger_id not in made:
            data = tmp_path_factory.mktemp("data")
            made[tagger_id] = TagImage({"host": {"dataDir": str(data)}, "modelDir": str(TAGGER_CACHE)})
        return made[tagger_id]

    yield worker
    for one in made.values():
        one.teardown()


@pytest.mark.parametrize("tagger_id", tagger_params())
def test_the_tag_list_is_the_catalogue_s(workers, tagger_id):
    tagger = TAGGERS[tagger_id]
    labels = workers(tagger_id).models.labels(tagger)

    counts = {code: labels.codes.count(code) for code in set(labels.codes)}
    if tagger_id == "wd-eva02-large":
        assert counts == {0: 8106, 4: 2751, 9: 4}
    assert set(counts) == {category.code for category in tagger.categories} | {tagger.rating_code}


@pytest.mark.parametrize("image", sorted(next(iter(GOLDEN.values()))))
@pytest.mark.parametrize("tagger_id", tagger_params())
def test_tags_as_the_reference_pipeline_does(workers, tagger_id, image):
    expected = GOLDEN[tagger_id][image]
    every_tag = {category.name: 0 for category in TAGGERS[tagger_id].categories}
    task = Task({"path": str(IMAGES / image)}, {"tagger": tagger_id, "thresholds": every_tag})

    result = run(workers(tagger_id), task)

    scores = {t["name"]: t["score"] for t in result["tags"]}
    assert {name: scores.get(name) for name in expected["tags"]} == pytest.approx(expected["tags"], abs=TOLERANCE)
    assert result["tags"][0]["name"] == next(iter(expected["tags"]))
    assert result["rating"]["scores"] == pytest.approx(expected["rating"], abs=TOLERANCE)
    assert result["rating"]["label"] == max(expected["rating"], key=expected["rating"].get)
