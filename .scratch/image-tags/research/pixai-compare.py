"""Compares Mexes's ONNX conversion of PixAI v1.0 with the upstream PyTorch reference (reference.py's npz).

Usage:
  python -I compare.py inputs <reference.npz> <images dir> <repo python dir>
      how close each candidate preprocessing's tensor is to upstream's RescalePadProcessor output (no model)
  python -I compare.py model <reference.npz> <images dir> <repo python dir> <config.json> <model.onnx> <provider>
      <variant,...> [out.json]
      runs the ONNX file on each image, fed each named variant ("upstream" is the reference's own input tensor), and
      reports the probability differences and the tags that flip at the default thresholds

The repo's `python/` folder is put on the path for the Plugin's own preprocessing (`image_tags.preprocess`), which is
trusted code; downloaded files are only read as data.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THRESHOLDS = {"general": 0.17, "character": 0.27, "copyright": 0.24, "style": 0.15, "meta": 0.17, "rating": 0.41}
SIZE = 1008


def resize_then_pad(image: Image.Image, resample, rounding) -> np.ndarray:
    """Long side to 1008 (upstream's int() truncation, or Mexes's round()), centred on black, (x/255 - 0.5)/0.5."""
    w, h = image.size
    r = min(SIZE / h, SIZE / w)
    nw, nh = (rounding(w * r), rounding(h * r))
    if (w, h) != (SIZE, SIZE):
        image = image.resize((nw, nh), resample)
    else:
        nw, nh = w, h
    canvas = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    canvas.paste(image, ((SIZE - nw) // 2, (SIZE - nh) // 2))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray((x - 0.5) / 0.5)


def resize_then_pad_float(image: Image.Image) -> np.ndarray:
    """As upstream's tensor path: to float 0-1 first, each channel resized as a PIL "F" image (bilinear, antialiased),
    truncated size, centred on black (0), then (x - 0.5) / 0.5."""
    w, h = image.size
    r = min(SIZE / h, SIZE / w)
    nw, nh = int(w * r), int(h * r)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    out = np.zeros((3, SIZE, SIZE), dtype=np.float32)
    left, top = (SIZE - nw) // 2, (SIZE - nh) // 2
    for c in range(3):
        channel = Image.fromarray(np.ascontiguousarray(pixels[:, :, c]), mode="F")
        if (w, h) != (SIZE, SIZE):
            channel = channel.resize((nw, nh), Image.Resampling.BILINEAR)
        out[c, top:top + nh, left:left + nw] = np.asarray(channel)
    return (out - 0.5) / 0.5


def variants(repo_python: Path):
    sys.path.insert(0, str(repo_python))
    from image_tags.preprocess import NCHW, RGB, Preprocessing, load_image, to_input

    def plugin(resample):
        pre = Preprocessing((SIZE, SIZE), (0, 0, 0), resample, RGB, NCHW, 255.0, 0.5, 0.5)
        return lambda image: to_input(image, pre)

    return load_image, {
        "bilinear-float": resize_then_pad_float,
        "bilinear": lambda im: resize_then_pad(im, Image.Resampling.BILINEAR, int),
        "bicubic": lambda im: resize_then_pad(im, Image.Resampling.BICUBIC, int),
        "bicubic-round": lambda im: resize_then_pad(im, Image.Resampling.BICUBIC, round),
        "pad-then-bilinear": plugin(Image.Resampling.BILINEAR),
        "pad-then-bicubic": plugin(Image.Resampling.BICUBIC),
    }


def inputs(npz, images, repo_python):
    load_image, made = variants(repo_python)
    names = sorted(k.split("/", 1)[1] for k in npz.files if k.startswith("input/"))
    for variant, make in made.items():
        maxes, means = [], []
        for name in names:
            d = np.abs(make(load_image(str(images / name))) - npz[f"input/{name}"])
            maxes.append(d.max())
            means.append(d.mean())
        print(f"{variant:18s} input |diff| max {max(maxes):.4f}  mean of means {np.mean(means):.5f}")


def model(npz, images, repo_python, config_path, onnx_path, provider, wanted, out):
    import onnxruntime as ort

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    tags = config["tags"]
    cats = np.array([c for c, n in config["tags_split"] for _ in range(n)])
    thresholds = np.array([THRESHOLDS[c] for c in cats], dtype=np.float32)

    load_image, made = variants(repo_python)
    providers = {"cuda": [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
                 "cpu": ["CPUExecutionProvider"]}[provider]
    if provider == "cuda":
        ort.preload_dlls()
    options = ort.SessionOptions()
    session = ort.InferenceSession(onnx_path, options, providers=providers)
    print(onnx_path, "providers", session.get_providers())
    names = sorted(k.split("/", 1)[1] for k in npz.files if k.startswith("probs/"))
    report = {}
    for variant in wanted:
        maxes, means, flips, near, per = [], [], {}, [], {}
        for name in names:
            x = npz[f"input/{name}"] if variant == "upstream" else made[variant](load_image(str(images / name)))
            (p,) = session.run(["prediction"], {"input": x[None].astype(np.float32)})
            p = p[0]
            ref = npz[f"probs/{name}"]
            d = np.abs(p - ref)
            maxes.append(float(d.max()))
            means.append(float(d.mean()))
            per[name] = [float(d.max()), float(d.mean())]
            on_ref, on = ref > thresholds, p > thresholds
            lost = [(tags[i], cats[i], round(float(ref[i]), 4), round(float(p[i]), 4)) for i in np.where(on_ref & ~on)[0]]
            added = [(tags[i], cats[i], round(float(ref[i]), 4), round(float(p[i]), 4)) for i in np.where(~on_ref & on)[0]]
            if lost or added:
                flips[name] = {"lost": lost, "added": added}
            near.append(int(on_ref.sum()))
        report[variant] = {"max": max(maxes), "mean": float(np.mean(means)), "flips": flips,
                           "tags over threshold (ref, total)": int(sum(near)), "per_image": per}
        nflip = sum(len(f["lost"]) + len(f["added"]) for f in flips.values())
        print(f"{variant:18s} max {max(maxes):.5f} mean {np.mean(means):.2e} flips {nflip} of {sum(near)} ref tags")
        for name, f in flips.items():
            print("   ", name, f)
    if out:
        Path(out).write_text(json.dumps(report, indent=1), encoding="utf-8")


def main():
    mode = sys.argv[1]
    npz = np.load(sys.argv[2])
    images, repo_python = Path(sys.argv[3]), Path(sys.argv[4])
    if mode == "inputs":
        inputs(npz, images, repo_python)
    else:
        model(npz, images, repo_python, sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8].split(","),
              sys.argv[9] if len(sys.argv) > 9 else None)


if __name__ == "__main__":
    main()
