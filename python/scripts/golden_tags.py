# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=0.25", "numpy>=2.1", "onnxruntime>=1.21", "pillow>=11"]
# ///
"""Writes what each catalogue tagger says of the test images, prepared by its reference pipeline rather than the
Plugin's code: `python/tests/fixtures/golden-tags.json`, which the taggers' golden tests hold the Plugin to.

- `wd-eva02-large`: the authors' Space (`SmilingWolf/wd-tagger`, `app.py`), its steps copied below: the image as
  RGBA (Gradio's `image_mode="RGBA"`), on a white canvas, padded to a square on white, resized bicubic to the model's
  input, RGB to BGR, float32 from 0 to 255.

PixAI's reference needs its authors' PyTorch code: `scripts/golden_pixai.py` writes it, to `golden-pixai.json`.

Run by hand (`pnpm golden:tags`) when a tagger of the catalogue changes; never by `pnpm check`. It runs in an
environment of its own (the inline metadata above). Each tagger's files are downloaded at its pinned commit, or taken
from `--from <folder>` (a Profile's `modelDir` layout, e.g. `python/.cache/models/`), and checked against the
catalogue's sha256 first; then the entry is checked against them: the graph's input and output, the tag list's length
against the output's, its categories, and that the output is already a probability per tag.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

PYTHON = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PYTHON))
from image_tags.catalogue import TAGGERS  # noqa: E402  (the pins and categories only, never the preprocessing)
from image_tags.models import sha256_of  # noqa: E402

FIXTURES = PYTHON / "tests" / "fixtures"
IMAGES = FIXTURES / "images"
#: The most tags kept per image, general and character together, highest first.
TOP = 20
#: The taggers whose reference is the Space's `app.py`.
WD_TAGGERS = ("wd-eva02-large",)


def prepare_image(image: Image.Image, target_size: int) -> np.ndarray:
    """`app.py`'s `Predictor.prepare_image`, as it is there."""
    canvas = Image.new("RGBA", image.size, (255, 255, 255))
    canvas.alpha_composite(image)
    image = canvas.convert("RGB")

    # Pad image to square
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # Resize
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

    # Convert to numpy array
    image_array = np.asarray(padded_image, dtype=np.float32)

    # Convert PIL-native RGB to BGR
    image_array = image_array[:, :, ::-1]

    return np.expand_dims(image_array, axis=0)


def images():
    """The test images: the reference frames and the synthetic ones `golden.py` makes."""
    return sorted(path.name for path in IMAGES.iterdir() if path.suffix in (".jpg", ".png"))


def file_of(pinned, source: Path | None) -> str:
    """The pinned file, from `source` (laid out as a model folder) or Hugging Face at its commit; exits unless it is
    the pinned one."""
    if source is not None:
        path = str(source / pinned.local_path)
    else:
        path = hf_hub_download(pinned.repo, pinned.file, subfolder=pinned.name or None, revision=pinned.commit)
    if sha256_of(path) != pinned.sha256:
        sys.exit(f"{path} is not the catalogue's pinned {pinned.file}")
    return path


def check_entry(tagger, session: ort.InferenceSession, rows: list[dict]) -> None:
    """Exits naming what of `tagger`'s catalogue entry its own files contradict."""
    (given,) = session.get_inputs()
    (output,) = session.get_outputs()
    width, height = tagger.preprocessing.size
    if given.name != "input" or isinstance(given.shape[0], int) or given.shape[1:] != [height, width, 3]:
        sys.exit(f"{tagger.id}: the graph takes {given.name} {given.shape}, not input [batch, {height}, {width}, 3]")
    if output.name != "output" or output.shape[1] != len(rows):
        sys.exit(f"{tagger.id}: the graph gives {output.name} {output.shape}, not output [batch, {len(rows)}]")
    codes = {int(row["category"]) for row in rows}
    named = {category.code for category in tagger.categories} | {tagger.rating_code}
    if not codes <= named:
        sys.exit(f"{tagger.id}: the tag list has categories {sorted(codes - named)} the catalogue does not name")
    batch = np.random.default_rng(0).uniform(0, 255, (2, height, width, 3)).astype(np.float32)
    (scores,) = session.run(["output"], {"input": batch})
    if tagger.sigmoided and not ((scores >= 0) & (scores <= 1)).all():
        sys.exit(f"{tagger.id}: the catalogue says its output is sigmoided, but it is not")


def golden_of(tagger, source: Path | None) -> dict:
    session = ort.InferenceSession(file_of(tagger, source), providers=["CPUExecutionProvider"])
    with open(file_of(tagger.tag_list, source), encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    check_entry(tagger, session, rows)
    counts = {}
    for row in rows:
        counts[int(row["category"])] = counts.get(int(row["category"]), 0) + 1
    print(f"{tagger.id}: {len(rows)} tags, by category {counts}")

    tagged = {category.code for category in tagger.categories}
    golden = {}
    for name in images():
        with Image.open(IMAGES / name) as opened:
            # Gradio hands `predict` the image as RGBA.
            image = opened.convert("RGBA")
        (output,) = session.run(["output"], {"input": prepare_image(image, tagger.preprocessing.size[0])})
        scored = [(row["name"], int(row["category"]), float(score)) for row, score in zip(rows, output[0], strict=True)]
        tags = sorted(((tag, score) for tag, code, score in scored if code in tagged), key=lambda p: p[1], reverse=True)
        rating = {tag: score for tag, code, score in scored if code == tagger.rating_code}
        golden[name] = {"tags": dict(tags[:TOP]), "rating": rating}
        print(f"  {name}: {', '.join(tag for tag, _ in tags[:5])}")
    return golden


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--from", dest="source", type=Path, help="a folder holding the taggers as a modelDir does")
    args = parser.parse_args()
    golden = {tagger_id: golden_of(TAGGERS[tagger_id], args.source) for tagger_id in WD_TAGGERS}
    out = FIXTURES / "golden-tags.json"
    out.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
