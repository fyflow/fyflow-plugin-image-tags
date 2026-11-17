"""PixAI Tagger v1.0 reference scores from the authors' own code (PyTorch, CPU).

Usage: python -I reference.py <upstream dir> <images dir> <out dir>

<upstream dir> holds the pinned files of pixai-labs/pixai-tagger-v1.0 @ 9fe10add: tagger_pipeline.py, config.json,
preprocessor_config.json, model.safetensors. tagger_pipeline.py (read in full beforehand) is imported by path for its
model (ViTDetCls, ViTDetClsConfig) and its image processor (RescalePadProcessor); the forward is TaggerPipeline._forward's
`self.model(x).sigmoid()`, without the transformers Pipeline wrapper (whose thresholding we redo ourselves).

Writes <out dir>/reference.npz (per image: the processor's input tensor and the 30,877 probabilities) and
<out dir>/golden-pixai.json in golden-tags.json's shape.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

TOP = 20


def main():
    upstream, images_dir, out = (Path(a) for a in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("tagger_pipeline", upstream / "tagger_pipeline.py")
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)

    config_dict = json.loads((upstream / "config.json").read_text(encoding="utf-8"))
    pre_dict = json.loads((upstream / "preprocessor_config.json").read_text(encoding="utf-8"))
    config = tp.ViTDetClsConfig(**config_dict)
    model = tp.ViTDetCls(config)
    state = load_file(str(upstream / "model.safetensors"))
    result = model.load_state_dict(state, strict=False)
    # freqs_cis is a non-persistent buffer computed in __init__; nothing else may be missing.
    print("missing:", result.missing_keys, "unexpected:", result.unexpected_keys)
    assert not result.missing_keys and not result.unexpected_keys
    model.eval()
    processor = tp.RescalePadProcessor(size=pre_dict["size"])
    print("processor size", processor.size, "params", sum(p.numel() for p in model.parameters()))

    tags = config.tags
    split = config.tags_split
    categories = [name for name, count in split for _ in range(count)]
    assert len(tags) == len(categories) == config.num_classes

    names = sorted(p.name for p in images_dir.iterdir() if p.suffix in (".jpg", ".png"))
    arrays = {}
    golden = {}
    for name in names:
        with Image.open(images_dir / name) as opened:
            opened.load()
            image = opened.copy()
        x = processor(image)["pixel_values"].to(model.dtype)
        with torch.inference_mode():
            logits = model(x)
            probs = logits.sigmoid()[0].numpy()
        arrays[f"input/{name}"] = x[0].numpy()
        arrays[f"logits/{name}"] = logits[0].numpy()
        arrays[f"probs/{name}"] = probs
        tagged = sorted(
            ((tags[i], float(probs[i])) for i in range(len(tags)) if categories[i] != "rating"),
            key=lambda pair: pair[1],
            reverse=True,
        )
        rating = {tags[i]: float(probs[i]) for i in range(len(tags)) if categories[i] == "rating"}
        golden[name] = {"tags": dict(tagged[:TOP]), "rating": rating}
        print(f"{name} {image.mode} {image.size}: {', '.join(t for t, _ in tagged[:6])} | {max(rating, key=rating.get)}")

    np.savez(out / "reference.npz", **arrays)
    (out / "golden-pixai.json").write_text(json.dumps({"pixai": golden}, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
