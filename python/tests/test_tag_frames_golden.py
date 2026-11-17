"""The real taggers on a frame sheet: each frame tagged as `TagImage` tags its cell cut out by hand, so the frames go
through the same preprocessing and model, batched or not.

As `test_tag_image_golden.py`, it runs only when the tagger is in `python/.cache/models/`, and is marked slow.
"""

import pytest
from PIL import Image

from image_tags.catalogue import TAGGERS
from image_tags.worker import TagFrames, TagImage
from tests.conftest import TAGGER_CACHE, Task, run
from tests.test_classify_frames import SAMPLE_TIMESTAMPS, SHEETS
from tests.test_tag_image_golden import TOLERANCE, tagger_params

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("tagger_id", tagger_params())
def test_each_frame_of_the_sample_sheet_is_tagged_as_tag_image_tags_its_cell(tagger_id, tmp_path):
    config = {"host": {"dataDir": str(tmp_path / "data")}, "modelDir": str(TAGGER_CACHE)}
    every_tag = {category.name: 0 for category in TAGGERS[tagger_id].categories}
    cells = []
    with Image.open(SHEETS / "sample.jpg") as image:
        for i in range(7):
            cell = tmp_path / f"cell{i}.png"
            image.crop((i % 3 * 160, i // 3 * 90, i % 3 * 160 + 160, i // 3 * 90 + 90)).save(cell)
            cells.append(cell)
    frames, alone = TagFrames(config), TagImage(config)
    try:
        settings = {"tagger": tagger_id, "thresholds": every_tag, "maxTags": 20}
        result = run(frames, Task({"path": str(SHEETS / "sample.jpg")}, settings), timeout=300)

        assert [frame["timestamp"] for frame in result["frames"]] == SAMPLE_TIMESTAMPS
        for frame, cell in zip(result["frames"], cells, strict=True):
            want = run(alone, Task({"path": str(cell)}, settings), timeout=300)
            assert [t["name"] for t in frame["tags"][:5]] == [t["name"] for t in want["tags"][:5]]
            got = {t["name"]: t["score"] for t in frame["tags"]}
            assert {t["name"]: got.get(t["name"]) for t in want["tags"][:5]} == pytest.approx(
                {t["name"]: t["score"] for t in want["tags"][:5]}, abs=TOLERANCE
            )
            assert frame["rating"]["scores"] == pytest.approx(want["rating"]["scores"], abs=TOLERANCE)
    finally:
        frames.teardown()
        alone.teardown()
