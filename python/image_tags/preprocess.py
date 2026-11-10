"""Images as the models take them: imgutils' recipe for its classifiers, which ship no `preprocess.json`.

RGB on a white background, a bilinear squash to the model's input (aspect ignored), scaled to [0, 1], then
`(x - 0.5) / 0.5`, float32 NCHW. The golden tests hold this to imgutils within 0.02.
"""

import numpy as np
from PIL import Image, UnidentifiedImageError


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


def to_input(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """`image` as one model input: `[3, height, width]` float32 in [-1, 1]."""
    resized = image.resize(size, Image.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (pixels - 0.5) / 0.5
