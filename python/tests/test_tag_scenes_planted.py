"""`TagScenes` on a planted video, through the real `wd-eva02-large`: three "scenes" made of the repo's own fixture
images (`anime-*.jpg`, AI-generated test data), a frame every 2 s, each frame a different crop, flip and brightness of
its scene's image. The middle scene cuts between two crops of one image, far apart, as dialogue does (A/B/A/B); black
frames lie between the first two scenes and inside the last.

Slow, and skipped unless the tagger is in `python/.cache/models/` (as `test_tag_image_golden.py`'s).
"""

import random

import pytest
from PIL import Image, ImageEnhance, ImageOps

from image_tags.worker import TagScenes
from tests.conftest import IMAGES, REACH, TAGGER_CACHE, Task, run
from tests.test_tag_image_golden import tagger_params

EVERY = 2.0
#: Each scene: its image, how many frames, and how its frames are cut from it: `None` for anywhere, or the shots it
#: cuts between, each a box (left, top, right, bottom) as shares of the image, two frames to a shot.
PLANTED = [
    ("anime-01-rainy-bus-stop.jpg", 10, None),
    ("anime-03-festival-night-market.jpg", 10, [(0.0, 0.1, 0.5, 0.9), (0.5, 0.1, 1.0, 0.9)]),
    ("anime-04-seaside-railway.jpg", 10, None),
]
#: Black frames: two between the first scene and the second, two in the middle of the last.
BLANKS_BETWEEN = 2
BLANKS_INSIDE = 2


def frame_of(image: Image.Image, rng: random.Random, box=None) -> Image.Image:
    """A frame cut from `image`: a crop of 75 to 90 % of its sides (within `box` when given), maybe flipped, its
    brightness from 0.85 to 1.15, so that no two frames are the same."""
    width, height = image.size
    left, top, right, bottom = box or (0, 0, 1, 1)
    area = (left * width, top * height, right * width, bottom * height)
    side = rng.uniform(0.75, 0.9)
    w, h = (area[2] - area[0]) * side, (area[3] - area[1]) * side
    x = area[0] + rng.uniform(0, area[2] - area[0] - w)
    y = area[1] + rng.uniform(0, area[3] - area[1] - h)
    frame = image.crop((round(x), round(y), round(x + w), round(y + h))).resize((640, 360))
    if rng.random() < 0.5:
        frame = ImageOps.mirror(frame)
    return ImageEnhance.Brightness(frame).enhance(rng.uniform(0.85, 1.15))


def planted(folder) -> tuple[list[dict], list[float], list[int], list[int]]:
    """The planted video, written to `folder`: its `frames` list, the times the second and third scenes start, the
    places of its black frames between scenes, and of those inside one."""
    rng = random.Random(23)
    folder.mkdir()
    frames, starts, between, inside = [], [], [], []

    def add(image: Image.Image) -> None:
        path = folder / f"{len(frames):03}.jpg"
        image.save(path, quality=90)
        frames.append({"path": str(path), "timestamp": len(frames) * EVERY})

    black = Image.new("RGB", (640, 360))
    for k, (name, count, shots) in enumerate(PLANTED):
        if k:
            starts.append(len(frames) * EVERY - EVERY / 2)
        image = Image.open(IMAGES / name).convert("RGB")
        for i in range(count):
            if k == 2 and i == count // 2:
                for _ in range(BLANKS_INSIDE):
                    inside.append(len(frames))
                    add(black)
            add(frame_of(image, rng, shots[i // 2 % len(shots)] if shots else None))
        if k == 0:
            for _ in range(BLANKS_BETWEEN):
                between.append(len(frames))
                add(black)
    return frames, starts, between, inside


@pytest.mark.slow
@pytest.mark.parametrize("tagger_id", tagger_params(["wd-eva02-large"]))
def test_the_planted_scenes_are_found_their_blanks_bridged_and_each_tagged(tmp_path, tagger_id):
    frames, starts, between, inside = planted(tmp_path / "video")
    worker = TagScenes({"host": {"dataDir": str(tmp_path / "data")}, "modelDir": str(TAGGER_CACHE)})

    try:
        result = run(worker, Task({"frames": frames}, {"tagger": tagger_id}, roots=REACH), timeout=600)
    finally:
        worker.teardown()

    for scene in result["scenes"]:
        print(scene["start"], scene["end"], [tag["name"] for tag in scene.get("tags", [])][:8])
    assert result["count"] == 3
    found = [scene["start"] for scene in result["scenes"][1:]]
    for at, want in zip(found, starts, strict=True):
        assert abs(at - want) <= EVERY, (found, starts)
    scene_of = {i: k for k, scene in enumerate(result["scenes"]) for i in scene["frames"]}
    # The black frames between two scenes end the first; those inside one stay in it.
    assert [scene_of[i] for i in between] == [0] * BLANKS_BETWEEN
    assert [scene_of[i] for i in inside] == [2] * BLANKS_INSIDE
    for scene in result["scenes"]:
        assert scene["tags"], scene
    assert result["summary"]["tags"]
