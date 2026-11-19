"""The frames a `ClassifyFrames`, `TagFrames` or `TagScenes` Task reads: those of a frame sheet, the images of a
folder, a list of images, or one image.

A **frame sheet** is a JPEG made by the video-thumbnail Plugin's `thumbs-collage` service: a collage of a video's
frames, whose comment (COM) segment holds the service's JSON. Of that JSON this reads `frames` (per cell, null or
`{rect: {x, y, w, h}, timestamp_actual, …}`, the rect where the frame was drawn on the canvas), `video` (the video's
path, when the service was not told to leave it out) and `config.render_timestamps` (whether a timestamp was drawn on
each frame). Any other JPEG, or image, is not a sheet.

A **thumbnail** is a JPEG of one frame made by the same Plugin, whose comment holds a JSON object with the frame's
`timestamp` (seconds into the video), the video's `duration` and, when the service was not told to leave it out,
`video`. A thumbnail read as a plain image (alone, in a folder or in a list) gives its frame that timestamp, video and
duration.

Nothing is read until the Task's reach allows it (`allowed_path`, the Host's check of its Allowed roots): the path
given, each image of a folder, since a link inside an allowed folder may lead out of it, and each image of a list. The
`video` a sheet or thumbnail names is only said, never read, so it is not checked.
"""

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageStat, UnidentifiedImageError

from .preprocess import open_image, to_rgb

#: More frames than this in a folder fail the Task: a large set belongs in a Flow `map` over `ClassifyImage` or
#: `TagImage`.
MAX_FOLDER_IMAGES = 10_000
#: More frames than this in a `frames` list fail the Task, as a folder's images do.
MAX_LISTED_FRAMES = MAX_FOLDER_IMAGES

#: A frame whose grayscale standard deviation (out of 255) is below this is blank: black, white or a solid card.
BLANK_STD = 4
#: The side of the thumbnail a frame is measured on for `is_blank`, in pixels.
_BLANK_SIDE = 64

TIMESTAMPS_DRAWN = (
    "The frame sheet has timestamps drawn on its frames, and the models see them: make it with timestamps off "
    "(VideoThumbnail's renderTimestamps)"
)
SHEET_AS_ONE_IMAGE = (
    "This is a frame sheet, scored here as one image (the whole collage): ClassifyFrames classifies its frames one by "
    "one"
)
#: `TagImage`'s warning of a frame sheet, as `ClassifyImage`'s.
SHEET_TAGGED_AS_ONE_IMAGE = (
    "This is a frame sheet, tagged here as one image (the whole collage): TagFrames tags its frames one by one"
)


@dataclass
class Frame:
    """One frame to classify or tag: `load()` gives it in RGB, read when its batch comes. `timestamp` is where in the
    video it is, in seconds, when known; `path` is its own image, for a folder's or a list's frame; `video` and
    `duration` (seconds) are the video's, as a thumbnail says."""

    load: Callable[[], Image.Image]
    timestamp: float | None = None
    path: str | None = None
    video: str | None = None
    duration: float | None = None


@dataclass
class Frames:
    """The frames of a sheet, folder, list or image, in order, and what the result says of where they came from."""

    frames: list[Frame]
    video: str | None = None
    warnings: list[str] = field(default_factory=list)
    #: The files read, where they really are: what a result file's key covers.
    files: list[str] = field(default_factory=list)
    #: The video's duration in seconds, when the frames' thumbnails give one and agree on it; a sheet gives none.
    duration: float | None = None


def _comment_json(image: Image.Image):
    """The JSON of a JPEG's comment, or None when it has none, or none that parses."""
    comment = image.info.get("comment") if image.format == "JPEG" else None
    if not comment:
        return None
    try:
        return json.loads(comment)
    except (UnicodeDecodeError, ValueError):
        return None


def sheet_of(image: Image.Image) -> dict | None:
    """The service's JSON, when `image` (as `open_image` read it) is a frame sheet: a JPEG whose comment is a JSON
    object with a `frames` list."""
    sheet = _comment_json(image)
    return sheet if isinstance(sheet, dict) and isinstance(sheet.get("frames"), list) else None


def thumbnail_of(image: Image.Image) -> tuple[float | None, str | None]:
    """The timestamp and video a thumbnail's comment gives, each None when it gives none: `(None, None)` for any image
    that is not a thumbnail, a frame sheet included."""
    timestamp, video, _ = _thumbnail(image)
    return timestamp, video


def _thumbnail(image: Image.Image) -> tuple[float | None, str | None, float | None]:
    """The timestamp, video and video's duration a thumbnail's comment gives, each None when it gives none."""
    meta = _comment_json(image)
    if not isinstance(meta, dict) or isinstance(meta.get("frames"), list):
        return None, None, None
    timestamp, video, duration = meta.get("timestamp"), meta.get("video"), meta.get("duration")
    return (
        float(timestamp) if _is_seconds(timestamp) else None,
        video if isinstance(video, str) and video else None,
        float(duration) if _is_seconds(duration) and duration > 0 else None,
    )


def is_blank(image: Image.Image) -> bool:
    """Whether a frame is blank (black, white or a solid card): its grayscale standard deviation, measured on a
    thumbnail `_BLANK_SIDE` pixels across, is below `BLANK_STD`. `TagScenes` asks as each frame's batch loads."""
    small = image.copy()
    small.thumbnail((_BLANK_SIDE, _BLANK_SIDE))
    return ImageStat.Stat(small.convert("L")).stddev[0] < BLANK_STD


#: The Host's check of a path against the Task's reach, `ctx.allowed_path(path, access)`: where the path really leads,
#: or a `PermissionError` saying why it is refused.
AllowedPath = Callable[[str, str], str]


def read_frames(path: str, allowed_path: AllowedPath) -> Frames:
    """The frames at `path`: a folder's images, a frame sheet's frames, or the image itself as one frame. `path` is
    read where it really leads, once `allowed_path` allows it."""
    real = allowed_path(path, "read")
    if os.path.isdir(real):
        return _folder(path, real, allowed_path)
    image = open_image(real)
    sheet = sheet_of(image)
    rgb = to_rgb(image)
    if sheet is None:
        timestamp, video, duration = _thumbnail(image)
        frame = Frame(lambda: rgb, timestamp, video=video, duration=duration)
        return Frames([frame], video, files=[real], duration=duration)
    source = _sheet(path, rgb, sheet)
    source.files = [real]
    return source


def read_frame_list(items, allowed_path: AllowedPath) -> Frames:
    """The frames a `frames` input lists, in its order: each `{path, timestamp?}`, an image read as a plain one (a
    frame sheet too). Every entry is checked before any frame is read, so a bad one fails the Task before a model
    loads; a path the reach refuses fails it too, since the caller named it. A timestamp given wins over a
    thumbnail's own."""
    if not isinstance(items, list):
        raise ValueError(f"`frames` is not a list: {items!r}")
    if not items:
        raise ValueError("`frames` is empty: list at least one frame")
    if len(items) > MAX_LISTED_FRAMES:
        raise ValueError(
            f"`frames` has more than {MAX_LISTED_FRAMES} frames (a large set belongs in a map over ClassifyImage or "
            "TagImage)"
        )
    frames = []
    files = []
    for index, item in enumerate(items):
        path = item.get("path") if isinstance(item, dict) else None
        if not isinstance(path, str) or path == "":
            raise ValueError(f"Frame {index} is not an object with a `path`: {item!r}")
        timestamp = item.get("timestamp")
        if timestamp is not None and not _is_seconds(timestamp):
            raise ValueError(f"Frame {index}'s `timestamp` is not a number of 0 or more, in seconds: {timestamp!r}")
        file = Path(allowed_path(path, "read"))
        meta = _image_meta(file)
        if meta is None:
            raise ValueError(f"Frame {index} is not an image: {path}")
        own, video, duration = meta
        files.append(str(file))
        frames.append(
            Frame(
                lambda file=file: to_rgb(open_image(str(file))),
                float(timestamp) if timestamp is not None else own,
                path,
                video,
                duration,
            )
        )
    return Frames(frames, _one_video(frames), files=files, duration=_one_duration(frames))


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


def _folder(given: str, path: str, allowed_path: AllowedPath) -> Frames:
    """Every image directly inside the folder (`path`, where `given` really leads), sorted by name, each frame named
    as `given` joined with its file's name; a frame sheet among them is a plain image, a thumbnail gives its timestamp.
    An entry the reach refuses (a link leading out of it) is skipped with a warning, unless nothing else is left to
    read."""
    frames = []
    files = []
    refused: list[PermissionError] = []
    for name in sorted(os.listdir(path)):
        try:
            file = Path(allowed_path(os.path.join(path, name), "read"))
        except PermissionError as error:
            refused.append(error)
            continue
        meta = _image_meta(file)
        if meta is None:
            continue
        load = lambda file=file: to_rgb(open_image(str(file)))  # noqa: E731
        files.append(str(file))
        frames.append(Frame(load, meta[0], os.path.join(given, name), meta[1], meta[2]))
        if len(frames) > MAX_FOLDER_IMAGES:
            raise ValueError(
                f"The folder has more than {MAX_FOLDER_IMAGES} images: {path} (a large set belongs in a map over "
                "ClassifyImage or TagImage)"
            )
    if not frames:
        if refused:
            raise refused[0]
        raise ValueError(f"The folder has no images: {path}")
    warnings = [f"Skipped: {error}" for error in refused]
    return Frames(frames, _one_video(frames), warnings, files, _one_duration(frames))


def _image_meta(file: Path) -> tuple[float | None, str | None, float | None] | None:
    """`file`'s thumbnail timestamp, video and duration, or None when it is not a file Pillow opens: its header alone
    is read, which holds a JPEG's comment."""
    if not file.is_file():
        return None
    try:
        with Image.open(file) as image:
            return _thumbnail(image)
    except (UnidentifiedImageError, OSError):
        return None


def _one_video(frames: list[Frame]) -> str | None:
    """The video the frames' thumbnails name, when they name one and only one."""
    videos = {frame.video for frame in frames if frame.video is not None}
    return videos.pop() if len(videos) == 1 else None


def _one_duration(frames: list[Frame]) -> float | None:
    """The video's duration the frames' thumbnails give, when they give one and all that give one agree."""
    durations = {frame.duration for frame in frames if frame.duration is not None}
    return durations.pop() if len(durations) == 1 else None


def _is_seconds(value) -> bool:
    """Whether `value` is a timestamp: a finite number of 0 or more, not a bool."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
