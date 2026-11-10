# /// script
# requires-python = ">=3.12"
# dependencies = ["dghs-imgutils==0.19.0", "onnxruntime>=1.21"]
# ///
"""Makes the golden images and what imgutils scores them: `python/tests/fixtures/golden.json`.

Run by hand (`pnpm golden`) when a catalogue model changes; never by `pnpm check`. It runs in an environment of its
own (the inline metadata above), so imgutils and its `numpy<2` never reach the Plugin's lock. imgutils downloads
from each repo's `main`, so the file it used is checked against the catalogue's pinned sha256 first; then the
entry is checked against the model's own files: its labels against `meta.json`, its input size against the graph,
and that the graph takes any batch size and hands back probabilities.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from imgutils.generic import classify_predict_score
from PIL import Image, ImageDraw, ImageFilter

PYTHON = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PYTHON))
from image_tags.catalogue import MODELS  # noqa: E402
from image_tags.models import sha256_of  # noqa: E402

FIXTURES = PYTHON / "tests" / "fixtures"
IMAGES = FIXTURES / "images"


def cartoon(path):
    """Flat colours inside black outlines, wider than tall."""
    image = Image.new("RGB", (512, 384), (250, 236, 210))
    draw = ImageDraw.Draw(image)
    draw.ellipse((150, 60, 360, 300), fill=(255, 220, 190), outline=(0, 0, 0), width=6)
    draw.ellipse((200, 140, 240, 200), fill=(40, 90, 200), outline=(0, 0, 0), width=4)
    draw.ellipse((270, 140, 310, 200), fill=(40, 90, 200), outline=(0, 0, 0), width=4)
    draw.arc((215, 210, 295, 260), 20, 160, fill=(0, 0, 0), width=5)
    draw.polygon([(150, 140), (255, 20), (360, 140), (330, 70), (180, 70)], fill=(230, 80, 120), outline=(0, 0, 0))
    image.save(path)


def photo(path):
    """A smooth gradient under blur and grain, as a camera might see a wall in soft light."""
    rng = random.Random(7)
    image = Image.new("RGB", (640, 360))
    pixels = image.load()
    for y in range(360):
        for x in range(640):
            base = 90 + x * 80 // 640 + y * 40 // 360
            pixels[x, y] = tuple(max(0, min(255, base + tint + rng.randint(-18, 18))) for tint in (12, 0, -15))
    image.filter(ImageFilter.GaussianBlur(1.2)).save(path, quality=88)


def transparent(path):
    """A shape on a transparent background: what is see-through must become white."""
    image = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 60, 240, 240), fill=(30, 160, 80, 255), outline=(0, 0, 0, 255), width=5)
    image.save(path)


def keyed(path):
    """No alpha channel, but one colour keyed as see-through (a PNG `tRNS`): white as well."""
    image = Image.new("RGB", (240, 160), (0, 255, 0))
    ImageDraw.Draw(image).ellipse((40, 20, 200, 140), fill=(200, 40, 40), outline=(0, 0, 0), width=4)
    image.save(path, transparency=(0, 255, 0))


def keyed_grey(path):
    """Greyscale with a keyed see-through level."""
    image = Image.new("L", (200, 200), 0)
    ImageDraw.Draw(image).rectangle((50, 50, 150, 150), fill=180, outline=255, width=3)
    image.save(path, transparency=0)


def tiny(path):
    """Far smaller than the model's input: scaled up."""
    image = Image.new("RGB", (64, 48), (20, 20, 60))
    ImageDraw.Draw(image).ellipse((16, 8, 48, 40), fill=(240, 200, 40))
    image.save(path, quality=90)


MAKERS = {
    "cartoon.png": cartoon,
    "photo.jpg": photo,
    "transparent.png": transparent,
    "keyed.png": keyed,
    "keyed-grey.png": keyed_grey,
    "tiny.jpg": tiny,
}


def check_entry(model, onnx_file):
    """Exits naming what of `model`'s catalogue entry its own files contradict."""
    meta = json.loads(Path(hf_hub_download(model.repo, f"{model.name}/meta.json")).read_text(encoding="utf-8"))
    if tuple(meta["labels"]) != model.labels:
        sys.exit(f"{model.id}: meta.json's labels are {meta['labels']}, the catalogue's {list(model.labels)}")
    session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    (given,) = session.get_inputs()
    width, height = model.size
    if given.name != "input" or isinstance(given.shape[0], int) or list(given.shape[1:]) != [3, height, width]:
        sys.exit(f"{model.id}: the graph takes {given.name} {given.shape}, not input [batch, 3, {height}, {width}]")
    batch = np.random.default_rng(0).uniform(-1, 1, (2, 3, height, width)).astype(np.float32)
    (output,) = session.run(["output"], {"input": batch})
    if output.shape != (2, len(model.labels)) or not np.allclose(output.sum(axis=1), 1, atol=1e-4):
        sys.exit(f"{model.id}: the graph's output is not one probability per label for each image")


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    for name, make in MAKERS.items():
        make(IMAGES / name)

    golden = {}
    for model in MODELS.values():
        used = hf_hub_download(model.repo, f"{model.name}/{model.file}")
        if sha256_of(used) != model.sha256:
            sys.exit(f"{model.repo}'s main has a {model.name} other than the catalogue's pinned one: re-pin it first")
        check_entry(model, used)
        golden[model.id] = {
            name: classify_predict_score(str(IMAGES / name), model.repo, model.name) for name in MAKERS
        }

    out = FIXTURES / "golden.json"
    out.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
