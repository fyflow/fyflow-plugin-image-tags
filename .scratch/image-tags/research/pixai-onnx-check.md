# PixAI v1.0: Mexes's ONNX conversion checked against the authors' pipeline (ticket 15)

Done 2026-11-17 on the RTX 3050 Ti 4 GB laptop. **Verdict: pin Mexes** (`model.onnx` as `pixai`, `model_fp16.onnx` as
`pixai-fp16`), with upstream's default thresholds for both. **Preprocessing:** the long side is resized first, then
the image is padded. A bilinear resize done in float matches upstream best.

## Method

- **Downloads:** each repo's files were fetched at its pinned commit into a throwaway folder and handled as data. Both
  pinned commits are still each repo's `main`.
- **Throwaway environment** (uv, Python 3.12), never the Plugin's:
  - torch 2.9.1+cpu, torchvision 0.24.1, timm, transformers 4.57.6 (the version `config.json` names), safetensors;
  - onnxruntime-gpu 1.31.1 with the CUDA 13 and cuDNN wheels, the same build as the Plugin's `cuda` variant.
- **Reference:**
  - Upstream's `tagger_pipeline.py` was read in full, then imported by its path for its model (`ViTDetCls`) and its
    image processor (`RescalePadProcessor`). Running the model needs their code, so this could not be avoided.
  - The weights load with `strict=False`, and no key is missing or unexpected.
  - The scores are `TaggerPipeline._forward`'s `model(x).sigmoid()`. The transformers `Pipeline` wrapper only applies
    thresholds, so it was left out.
  - Run on the CPU on all 21 fixture images.
- **Comparison:**
  - Each ONNX file was fed two kinds of input:
    - upstream's own input tensor, to compare the models alone;
    - our candidate preprocessings, done with PIL and numpy, to compare end to end.
  - A tag "flips" when it is over its threshold on one side and not on the other; above, not equal, as in
    `_forward`.
  - Scores were compared on CUDA. The CPU check was functional only.
- **Scripts:** in the session scratchpad's `pixai/scripts/`:
  - `reference.py` makes the reference;
  - `tagmap.py` checks the tag map and the graphs;
  - `compare.py` runs the comparisons, and `summary.py` totals them;
  - `onecheck.py` runs one image on the CPU and on CUDA.

## Upstream facts, verified

- **Upstream repo:** `pixai-labs/pixai-tagger-v1.0` @ `9fe10addf9326e292da8a85a98ea74cd91b41771`.
  - Licence: the card's metadata says `license: apache-2.0`. The repo has no LICENSE file.
  - It ships safetensors only.

  | File | Bytes | sha256 |
  | :--- | ---: | :--- |
  | `model.safetensors` | 1,945,425,796 | `f29e475205cbcbc25b52a075840c7809d6215be15f45375113ba09c49bf90292` |
  | `config.json` (holds the tag map: `tags`, `tags_split`) | 693,918 | `f8a19b38661c37dc5fd519f2137be70f6f897972021b4ebc17b7d35a0e27b9dd` |
  | `tagger_pipeline.py` | 43,696 | `3431279fdfff234f62664999fc1fe6bb20f1e8756f7435d0e165ebd4456f7fcd` |
  | `preprocessor_config.json` (`size: 1008`) | 600 | `20451575e627950993de7b00641d5b263eceb3e7436e4ad518fea2411b60e49d` |

- **Model:** 486,346,909 parameters. It is ViTDet (SAM3-style): patch 14, depth 32, dimension 1024, 2-D RoPE, window
  24 with 4 global blocks, and an attention-pool head. The output is logits.
  - `img_size` is fixed at 1008, because the RoPE tables are built for it. So the input is always 1008×1008.
- **Tags:** 30,877, in `tags_split` order:

  | Category | Count | Indices |
  | :--- | ---: | :--- |
  | general | 15,043 | 0–15042 |
  | character | 8,308 | 15043–23350 |
  | copyright | 2,460 | 23351–25810 |
  | style | 4,917 | 25811–30727 |
  | meta | 145 | 30728–30872 |
  | rating | 4 | 30873–30876 |

  - The rating tags, in this order: `rating:s`, `rating:g`, `rating:q`, `rating:e`.
  - Every name is unique.
- **Thresholds:** `category_best_threshold` is general 0.17, character 0.27, style 0.15, copyright 0.24, meta 0.17,
  rating 0.41. `tags_best_threshold` is null, so each category's threshold applies to all its tags.
- **Upstream preprocessing** (`RescalePadProcessor`):
  1. Colour: a PIL image already in RGB mode is used as it is. Any other mode is made RGBA and composited on white.
     So an RGB image's tRNS colour key is ignored.
  2. `to_tensor`: float32 from 0 to 1.
  3. Size: `r = min(1008/h, 1008/w)`, then `new = int(side * r)` (truncated, not rounded).
  4. Resize: torchvision's tensor `resize` with its defaults, bilinear and antialiased (torchvision 0.17 and later).
  5. Padding: black (0), centred, with `left = (1008 - new_w) // 2` and the odd pixel on the right or bottom.
  6. Normalisation: `(x - 0.5) / 0.5`, laid out RGB NCHW.

## The conversion

- **Repo:** `Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8` @ `0800778563144a0e6fdf41ddadd84aae3cb0dbcf`.
  - It has three commits, all from 2026-09-28. The last one only edits the README.
  - int8 was ignored.

  | File | Bytes | sha256 |
  | :--- | ---: | :--- |
  | `model.onnx` (fp32) | 1,984,062,544 | `1cae6083f07be1e1757125802566c767005ae883e4504c00db4ee1ef93486dad` |
  | `model_fp16.onnx` | 992,916,002 | `cd668e6760bb58ec400ce1b0286e5c9133f83b543caac6a378406b7d56f8c303` |
  | `selected_tags.csv` | 804,671 | `a9455cbf0a910d4a3890739f2f70bd986278f4594285ef1049121a03896e2a6d` |
  | `preprocess.json` | 288 | `b6cc89f3dbeee8e97666934d812a6a6bcebe81ae4eff763da9f58f4c52cd6ed4` |

- **Licence:**
  - The README's metadata says `license: apache-2.0`. Its "License" section says "Apache-2.0, same as the original
    model".
  - There is no LICENSE or NOTICE file. Upstream has none either.
  - The conversion is a derivative of Apache-2.0 weights, and it is offered under the same licence. Nothing is more
    restrictive. Our README should credit both repos.
- **Graphs:**
  - Both are opset 17, IR 8, made by PyTorch 2.12.1. Each is a single file, with no external data.
  - Input `input`, float32 `[batch, 3, 1008, 1008]`. Output `prediction`, float32 `[batch, 30877]`.
  - The output is already sigmoided: the last node is a Sigmoid in both. fp16 casts its output back to float32.
  - The batch is dynamic.
  - Mexes's export script rewrites RoPE with real cos and sin. Its fp16 file was made by `onnxconverter_common` with
    `keep_io_types`.
- **`selected_tags.csv`** (`tag_id,name,category,count`): checked against `config.json`.
  - Its names are the same and in the same order.
  - Its category codes, Danbooru's, the same codes as WD's `selected_tags.csv`: general 0, style 1, copyright 3,
    character 4, meta 5, rating 9.
  - `count` is always 0.
- **Mexes's README is wrong in two places:**
  - It says "bicubic".
  - Its sizes are in GiB: 1.89 is 1.85 GiB, and 947 MB is MiB.

## Results

The scores are probabilities. "w/o keyed" leaves out `keyed.png` (see below). There are 650 reference tags over
their thresholds, across 21 images.

**The model alone** (upstream's input tensor):

| Run | Max abs diff | Mean abs diff | Flips |
| :--- | ---: | ---: | :--- |
| fp32, CPU | 2.0e-5 | 3.0e-8 | 0 |
| fp32, CUDA | 1.1e-3 | 5.0e-7 | 2: `tree_shade` 0.1698→0.1701, `tree` 0.1703→0.1699 |
| fp16, CUDA | 5.8e-3 | 1.7e-6 | 3, all within 0.0011 of the threshold |
| fp16, CPU | 2.3e-3 | 9.2e-7 | 2, within 0.0009 |

**Resize method.** First, each preprocessing's input tensor against upstream's. The models were then run with each
on fp32 (CPU for the first four rows, CUDA for the float row), w/o keyed:

| Our preprocessing | Input max diff | Score max diff | Flips |
| :--- | ---: | ---: | :--- |
| resize, then pad; bilinear, done in **float** (PIL `"F"` image per channel) | ≤ 6.9e-5 | 0.0023 (CUDA) | 1 (`tree` 0.1703→0.1696) |
| resize, then pad; bilinear, done in uint8 (PIL RGB) | 0.0078 (1 level) | 0.084 (`shield` on Sintel, 0.13→0.19) | 10 |
| resize, then pad; bicubic, done in uint8 (Mexes's README) | ~0.25 | larger | 78 |
| pad first, then resize; bilinear (the Plugin's current `pad_to_square`) | ~1.0 at the band edges | up to 0.95 | 125 |

- **Bilinear, resized in float, then padded, matches upstream.** Use it for the Plugin.
- Padding first blurs the edge between the picture and the black bands. The model reads that edge as `letterboxed`,
  which goes up to 0.95.
- Bicubic is wrong, whatever Mexes's README says.
- uint8 bilinear is close, but it moves borderline tags.

**End to end with float bilinear:**
- fp16 on CUDA: max 0.0049, mean 1.7e-6, 1 flip (`tree_shade` 0.1698→0.1709).
- fp16 on the CPU: max 0.0029, 1 flip.

**fp16 thresholds:** fp16 needs no thresholds of its own.
- Its differences are 3 to 6 thousandths at most.
- Every flip is a tag already within about 0.001 of its threshold.

**`keyed.png`** (RGB with a tRNS colour key):
- The Plugin's `to_rgb` makes the keyed green white. Upstream ignores the key and keeps the green, so this image's
  tags differ, whichever ONNX file is used: `pokemon` 0.62 against 0.05, `green_background` against
  `white_background`.
- `keyed-grey.png` (mode L) and `transparent.png` (RGBA) agree, because upstream composites them on white too.
- Ticket 16 should keep the Plugin's convention, and leave `keyed.png` out of the PixAI golden test, or give it a
  note.

## Runs on this machine (functional only)

- **CUDA, batch 1:**
  - Both fp32 and fp16 run, with CUDA as the session's provider. ONNX Runtime warns that it added 34 Memcpy nodes, so
    a few nodes, probably the shape ops, run on the CPU.
  - GPU memory peaked at 3953 MiB (fp32) and 3957 MiB (fp16) of the 4096. ONNX Runtime's arena grows to fill the card.
  - fp32 took 15 to 45 s per image on CUDA, no quicker than the CPU, so it is at the edge of this card.
  - Larger `tagBatchSize`s were not tried. Expect the out-of-GPU-memory message there.
- **CPU:**
  - fp32 runs: loaded in about 5 s, then about 19 s per image (one image, 2 runs).
  - CPU and CUDA agree to 4.7e-4 on that image.
  - fp16 ran all 21 images on the CPU (not timed).

## Catalogue facts for ticket 16

- **Repo:** `Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8`, commit `0800778563144a0e6fdf41ddadd84aae3cb0dbcf`.
  - `pixai` is `model.onnx` (sha256 above).
  - `pixai-fp16` is `model_fp16.onnx` (sha256 above).
  - The tag list is `selected_tags.csv` (sha256 above), with `name` and `category` columns, like WD's.
- **Input:** `input` `[b,3,1008,1008]`, float32, RGB, NCHW. Output: `prediction` `[b,30877]`, **sigmoided in the
  graph**.
- **Preprocessing:**
  1. The image as RGB on white, as the Plugin already does.
  2. The long side to 1008. `new = int(side * 1008 / max(w, h))` (truncated), with no resize when the image is already
     1008×1008.
  3. **Bilinear, resized as float from 0 to 1** (one PIL `"F"` image per channel); not uint8.
  4. Pasted centred on black (0): offsets `(1008 - new) // 2`.
  5. `(x - 0.5) / 0.5`.
  - The current `Preprocessing` order (pad first, then resize) must not be used for PixAI.
- **Categories** (CSV code: name, default threshold): 0 general 0.17, 4 character 0.27, 3 copyright 0.24, 1 style 0.15,
  5 meta 0.17.
- **Rating:** code 9, threshold 0.41. Its labels are `rating:g`, `rating:s`, `rating:q`, `rating:e`, in the output
  order s, g, q, e. The catalogue may map them to general, sensitive, questionable and explicit, as WD names them.
- **fp16** uses the same thresholds.
- **Golden results:** `python/tests/fixtures/golden-pixai.json`.
  - Its shape is `golden-tags.json`'s, under the key `pixai`: the top 20 tags (every category but the rating), and the
    4 rating scores, for each of the 21 fixture images.
  - It was made with upstream's PyTorch pipeline on the CPU, by `reference.py`.
  - It is the reference for both `pixai` and `pixai-fp16`.
  - Tolerances: fp32 within about 0.003 end to end, and fp16 about 0.006. Leave out `keyed.png`.
