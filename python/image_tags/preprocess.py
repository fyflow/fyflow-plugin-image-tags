"""Images as the models take them, each model's way as its catalogue entry describes it (`Preprocessing`).

Every image is first made RGB on a white background. Then, by model:

- the classifiers, imgutils' recipe (they ship no `preprocess.json`): a bilinear squash to the model's input (aspect
  ignored), scaled to [0, 1], then `(x - 0.5) / 0.5`, float32 RGB NCHW. The golden tests hold this to imgutils within
  0.02;
- the WD taggers, their Space's `app.py`: padded to a square on white (centred), a bicubic resize, float32 BGR NHWC
  from 0 to 255. The tagger's golden tests hold this to a copy of `app.py`'s steps.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

WHITE = (255, 255, 255)
RGB = "RGB"
BGR = "BGR"
#: Channels first (`[3, height, width]`), or last (`[height, width, 3]`).
NCHW = "NCHW"
NHWC = "NHWC"


@dataclass(frozen=True)
class Preprocessing:
    """How a model takes an RGB image, in the order it is done: padded to a square of `pad`'s colour (the image
    centred), unless `pad` is `None`; resized to `size` (width, height) by `resample`, so squashed when not padded;
    in `channels` order and `layout`; divided by `scale`; then `(x - mean) / std`. Float32."""

    size: tuple[int, int]
    pad: tuple[int, int, int] | None
    resample: Image.Resampling
    channels: str
    layout: str
    scale: float
    mean: float
    std: float


def load_image(path: str) -> Image.Image:
    """The image at `path` in RGB, what is see-through made white."""
    return to_rgb(open_image(path))


def open_image(path: str) -> Image.Image:
    """The image at `path` as it is in the file, read whole, with what Pillow says of it (`format`, `info`)."""
    try:
        with Image.open(path) as opened:
            opened.load()
            return opened
    except UnidentifiedImageError:
        raise ValueError(f"Not an image Pillow can read: {path}") from None
    except OSError as error:
        raise ValueError(f"Cannot read the image {path}: {error.strerror or error}") from None


def to_rgb(image: Image.Image) -> Image.Image:
    """`image` in RGB, what is see-through made white."""
    if has_alpha(image):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.alpha_composite(rgba)
        return white.convert("RGB")
    return image.convert("RGB")


def has_alpha(image: Image.Image) -> bool:
    """Whether anything in `image` may be see-through: as imgutils' `has_alpha_channel` judges it."""
    return image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info


def to_input(image: Image.Image, preprocessing: Preprocessing) -> np.ndarray:
    """RGB `image` as one input of a model that takes it `preprocessing`'s way."""
    if preprocessing.pad is not None:
        image = pad_to_square(image, preprocessing.pad)
    pixels = np.asarray(image.resize(preprocessing.size, preprocessing.resample), dtype=np.float32)
    if preprocessing.channels == BGR:
        pixels = pixels[:, :, ::-1]
    if preprocessing.layout == NCHW:
        pixels = pixels.transpose(2, 0, 1)
    if preprocessing.scale != 1:
        pixels = pixels / preprocessing.scale
    if preprocessing.mean != 0 or preprocessing.std != 1:
        pixels = (pixels - preprocessing.mean) / preprocessing.std
    return np.ascontiguousarray(pixels)


def pad_to_square(image: Image.Image, colour: tuple[int, int, int]) -> Image.Image:
    """RGB `image` in the middle of a square of `colour` as wide as its longer side; an odd pixel of padding goes
    after it (right or bottom)."""
    width, height = image.size
    side = max(width, height)
    if width == height:
        return image
    square = Image.new("RGB", (side, side), colour)
    square.paste(image, ((side - width) // 2, (side - height) // 2))
    return square
