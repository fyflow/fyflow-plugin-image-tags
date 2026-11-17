# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface_hub>=0.25",
#   "numpy>=2.1",
#   "pillow>=11",
#   "safetensors>=0.4",
#   "timm>=1.0",
#   "torch==2.9.1",
#   "torchvision==0.24.1",
#   "transformers==4.57.6",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Writes what PixAI v1.0 says of the test images, from its authors' own pipeline in PyTorch rather than the Plugin's
code: `python/tests/fixtures/golden-pixai.json`, which the golden tests of `pixai` and `pixai-fp16` hold the Plugin
to.

The authors (`pixai-labs/pixai-tagger-v1.0`) ship safetensors and `tagger_pipeline.py` only, and running their model
needs their code: the pinned `tagger_pipeline.py` is imported by its path for its model (`ViTDetCls`) and its image
processor (`RescalePadProcessor`). The scores are `TaggerPipeline._forward`'s `model(x).sigmoid()`, without the
transformers `Pipeline` wrapper, which only applies thresholds. Each image is handed over as Pillow opens it: the
processor makes it RGB its own way (it ignores an RGB image's colour key, which the Plugin makes white, so
`keyed.png` differs and is left out of the tests).

Run by hand (`pnpm golden:pixai`) when PixAI's pins change; never by `pnpm check`. It runs in an environment of its
own (the inline metadata above: PyTorch for the CPU, never in the Plugin's), downloads about 2 GB at the authors'
pinned commit, checks each file's sha256 first, and takes some 9 minutes on a laptop's CPU. `--from <folder>` takes
the four files from a folder instead.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

PYTHON = Path(__file__).resolve().parent.parent
FIXTURES = PYTHON / "tests" / "fixtures"
IMAGES = FIXTURES / "images"
#: The most tags kept per image, every category but the ratings together, highest first.
TOP = 20

#: The authors' repo, at the commit ticket 15 checked Mexes's conversion against, and its files' sha256.
REPO = "pixai-labs/pixai-tagger-v1.0"
COMMIT = "9fe10addf9326e292da8a85a98ea74cd91b41771"
FILES = {
    "model.safetensors": "f29e475205cbcbc25b52a075840c7809d6215be15f45375113ba09c49bf90292",
    # Holds the tag list (`tags`, `tags_split`).
    "config.json": "f8a19b38661c37dc5fd519f2137be70f6f897972021b4ebc17b7d35a0e27b9dd",
    "tagger_pipeline.py": "3431279fdfff234f62664999fc1fe6bb20f1e8756f7435d0e165ebd4456f7fcd",
    "preprocessor_config.json": "20451575e627950993de7b00641d5b263eceb3e7436e4ad518fea2411b60e49d",
}


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_of(name: str, source: Path | None) -> Path:
    """The authors' pinned file, from `source` or Hugging Face at its commit; exits unless it is the pinned one."""
    path = source / name if source is not None else Path(hf_hub_download(REPO, name, revision=COMMIT))
    if sha256_of(str(path)) != FILES[name]:
        sys.exit(f"{path} is not {REPO}'s {name} at {COMMIT}")
    return path


def images():
    """The test images, as `golden_tags.py` takes them."""
    return sorted(path.name for path in IMAGES.iterdir() if path.suffix in (".jpg", ".png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--from", dest="source", type=Path, help="a folder holding the authors' four files")
    args = parser.parse_args()
    files = {name: file_of(name, args.source) for name in FILES}

    spec = importlib.util.spec_from_file_location("tagger_pipeline", files["tagger_pipeline.py"])
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)

    config = pipeline.ViTDetClsConfig(**json.loads(files["config.json"].read_text(encoding="utf-8")))
    model = pipeline.ViTDetCls(config)
    loaded = model.load_state_dict(load_file(str(files["model.safetensors"])), strict=False)
    # `freqs_cis` is a buffer made in `__init__`, never saved; nothing else may be missing or left over.
    if loaded.missing_keys or loaded.unexpected_keys:
        sys.exit(f"The weights do not fit: missing {loaded.missing_keys}, unexpected {loaded.unexpected_keys}")
    model.eval()
    size = json.loads(files["preprocessor_config.json"].read_text(encoding="utf-8"))["size"]
    processor = pipeline.RescalePadProcessor(size=size)

    tags = config.tags
    categories = [category for category, count in config.tags_split for _ in range(count)]
    if not len(tags) == len(categories) == config.num_classes:
        sys.exit(f"config.json names {len(tags)} tags in {len(categories)} categories for {config.num_classes} outputs")

    golden = {}
    for name in images():
        with Image.open(IMAGES / name) as opened:
            opened.load()
            image = opened.copy()
        x = processor(image)["pixel_values"].to(model.dtype)
        with torch.inference_mode():
            scores = model(x).sigmoid()[0].numpy()
        tagged = sorted(
            ((tags[i], float(scores[i])) for i in range(len(tags)) if categories[i] != "rating"),
            key=lambda pair: pair[1],
            reverse=True,
        )
        rating = {tags[i]: float(scores[i]) for i in range(len(tags)) if categories[i] == "rating"}
        golden[name] = {"tags": dict(tagged[:TOP]), "rating": rating}
        print(f"  {name}: {', '.join(tag for tag, _ in tagged[:5])} | {max(rating, key=rating.get)}", flush=True)

    out = FIXTURES / "golden-pixai.json"
    out.write_text(json.dumps({"pixai": golden}, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
