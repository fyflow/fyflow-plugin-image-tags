"""The frames a `ClassifyFrames` Task classifies: those of a frame sheet, the images of a folder, or one image.

A **frame sheet** is a JPEG made by the video-thumbnail Plugin's `thumbs-collage` service: a collage of a video's
frames, whose comment (COM) segment holds the service's JSON. Of that JSON this reads `frames` (per cell, null or
`{rect: {x, y, w, h}, timestamp_actual, …}`, the rect where the frame was drawn on the canvas), `video` (the video's
path, when the service was not told to leave it out) and `config.render_timestamps` (whether a timestamp was drawn on
each frame). Any other JPEG, or image, is not a sheet.

Nothing is read until the Task's reach allows it (`allowed_path`, the Host's check of its Allowed roots): the path
given, and each image of a folder, since a link inside an allowed folder may lead out of it. The sheet's `video` is
only said, never read, so it is not checked.
"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .preprocess import open_image, to_rgb

#: More frames than this in a folder fail the Task: a large set belongs in a Flow `map` over `ClassifyImage`.
MAX_FOLDER_IMAGES = 1000

TIMESTAMPS_DRAWN = (
    "The frame sheet has timestamps drawn on its frames, and the models see them: make it with timestamps off "
    "(VideoThumbnail's renderTimestamps)"
)
SHEET_AS_ONE_IMAGE = (
    "This is a frame sheet, scored here as one image (the whole collage): ClassifyFrames classifies its frames one by "
    "one"
)


@dataclass
class Frame:
    """One frame to classify: `load()` gives it in RGB, read when its batch comes. `timestamp` is where in the video
    it is, in seconds, when known."""

    load: Callable[[], Image.Image]
    timestamp: float | None = None


@dataclass
class Frames:
    """The frames of a sheet, folder or image, in order, and what the result says of where they came from."""

    frames: list[Frame]
    video: str | None = None
    warnings: list[str] = field(default_factory=list)


def sheet_of(image: Image.Image) -> dict | None:
    """The service's JSON, when `image` (as `open_image` read it) is a frame sheet: a JPEG whose comment is a JSON
    object with a `frames` list."""
    comment = image.info.get("comment") if image.format == "JPEG" else None
    if not comment:
        return None
    try:
        sheet = json.loads(comment)
    except (UnicodeDecodeError, ValueError):
        return None
    return sheet if isinstance(sheet, dict) and isinstance(sheet.get("frames"), list) else None


#: The Host's check of a path against the Task's reach, `ctx.allowed_path(path, access)`: where the path really leads,
#: or a `PermissionError` saying why it is refused.
AllowedPath = Callable[[str, str], str]


def read_frames(path: str, allowed_path: AllowedPath) -> Frames:
    """The frames at `path`: a folder's images, a frame sheet's frames, or the image itself as one frame. `path` is
    read where it really leads, once `allowed_path` allows it."""
    real = allowed_path(path, "read")
    if os.path.isdir(real):
        return _folder(real, allowed_path)
    image = open_image(real)
    sheet = sheet_of(image)
    rgb = to_rgb(image)
    if sheet is None:
        return Frames([Frame(lambda: rgb)])
    return _sheet(path, rgb, sheet)


def _sheet(path: str, image: Image.Image, sheet: dict) -> Frames:
    frames = []
    for cell in sheet["frames"]:
        box = _box(cell, image.size)
        if box is None:
            continue  # a frame the service could not decode, or a cell it had no timestamp for
        timestamp = cell.get("timestamp_actual")
        frames.append(
            Frame(
                lambda box=box: image.crop(box),
                float(timestamp) if isinstance(timestamp, int | float) and not isinstance(timestamp, bool) else None,
            )
        )
    if not frames:
        raise ValueError(f"The frame sheet has no frames: {path}")
    video = sheet.get("video")
    config = sheet.get("config")
    drawn = isinstance(config, dict) and config.get("render_timestamps") is True
    return Frames(frames, video if isinstance(video, str) else None, [TIMESTAMPS_DRAWN] if drawn else [])


def _box(cell, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """A cell's rect as a crop box within the canvas, or None for a null cell or one with nothing of it on it."""
    rect = cell.get("rect") if isinstance(cell, dict) else None
    if not isinstance(rect, dict):
        return None
    try:
        x, y, w, h = (int(rect[key]) for key in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError):
        return None
    left, top = max(x, 0), max(y, 0)
    right, bottom = min(x + w, size[0]), min(y + h, size[1])
    return (left, top, right, bottom) if right > left and bottom > top else None


def _folder(path: str, allowed_path: AllowedPath) -> Frames:
    """Every image directly inside the folder, sorted by name; a frame sheet among them is a plain image. An entry the
    reach refuses (a link leading out of it) is skipped with a warning, unless nothing else is left to read."""
    images = []
    refused: list[PermissionError] = []
    for name in sorted(os.listdir(path)):
        try:
            file = Path(allowed_path(os.path.join(path, name), "read"))
        except PermissionError as error:
            refused.append(error)
            continue
        if not file.is_file() or not _is_image(file):
            continue
        images.append(file)
        if len(images) > MAX_FOLDER_IMAGES:
            raise ValueError(
                f"The folder has more than {MAX_FOLDER_IMAGES} images: {path} (classify a large set with a map over "
                "ClassifyImage)"
            )
    if not images:
        if refused:
            raise refused[0]
        raise ValueError(f"The folder has no images: {path}")
    warnings = [f"Skipped: {error}" for error in refused]
    return Frames([Frame(lambda file=file: to_rgb(open_image(str(file)))) for file in images], warnings=warnings)


def _is_image(file: Path) -> bool:
    """Whether Pillow opens `file`: its header alone is read."""
    try:
        with Image.open(file):
            return True
    except (UnidentifiedImageError, OSError):
        return False
