# Image Tags: classify images and video frames with local models

Status: ready-for-agent

Raised by the user (2026-11-03): tell quickly and accurately whether a video is anime, CG or live action, using
deepghs/imgutils' models (or something better), with frames from the existing video-thumbnail Plugin, inside a Flow.
Decided with them: one Plugin, `fyflow-plugin-image-tags` (id `fyflow-ext.image-tags`), for classifiers now and
taggers later; no tagger in v1; frame extraction stays in the video-thumbnail Plugin (id `fyflow-ext.video-thumbnail`).

Refined in a grilling session (2026-11-10): the runtime is a **Python Worker** (fyflow-app ADR-0026, SDK 0.8.0+) with
no imgutils at runtime; Node Types are split by **input shape** (`ClassifyFrames`, `ClassifyImage`), not by output
vocabulary; the anime / CG / live-action answer is a **classifier preset** (`style`) chosen by a setting; GPU is in v1.

## Problem Statement

Sorting a video library by what kind of picture it is (anime, 3D CG, live action) takes a person opening each file.
fyflow-app can already list videos (`VideoFiles`) and pull frames out of them (`VideoThumbnail`), but nothing can
look at a frame and say what it is.

Small image classifiers do this well and fast. deepghs publishes them as ONNX on Hugging Face, with the
preprocessing in its Python library imgutils:

| Model (repo / name) | Labels | Size | Published accuracy |
| --- | --- | --- | --- |
| `deepghs/anime_real_cls` / `mobilenetv3_v1.4_dist` | anime, real | 0.63 GFLOPs, 4.2M params, ~17 MB | 98.78%, AUC 0.9968 |
| `deepghs/anime_classification` / `mobilenetv3_v1.3_dist` | 3d, bangumi, comic, illustration, not_painting | same | 96.41%, AUC 0.9973 |
| `deepghs/anime_classification` / `mobilenetv3_v1.5_dist` (imgutils' default) | same | same | 94.56%, AUC 0.9946 |

Their caformer_s36 versions cost about 35 times the compute for 0.3 to 1 point more; not worth it here. The
published figures are on deepghs' own image sets, not on video frames, and the `3d` class was learned from 3D-style
illustrations, not 3D animated films or game footage. So how well "CG" is told apart is unknown until it is measured
on the user's own videos.

## Solution

1. **Two Node Types, split by input shape**, because the shape decides the pool and how frames are batched:
   - **`ClassifyFrames`**: a frame sheet or a folder of images; the frames of one Task run as one batch and the
     result is aggregated over them.
   - **`ClassifyImage`**: one image per Task; many concurrent Tasks are batched together in one process.
2. **The classifier is a setting** (`classifier`), an enum over a **catalogue** of:
   - **models**: one ONNX classifier each, whose labels are the model's own (`anime_real`, `anime_classification`);
   - **presets**: a fixed label set computed from one or more models by a rule. `style` (anime / cg / live_action)
     is the default. A preset's vocabulary stays stable when the models under it change, so Flows that branch on
     `result.tag == "cg"` survive a model swap. A raw model's labels belong to that model; switching models means
     rewriting Edges.
3. **Frame sheets**: a JPEG made by `VideoThumbnail` carries its frames' places (`frames[].rect`) and timestamps in
   its metadata, so `ClassifyFrames` cuts the frames out by itself; the sheet's layout is never configured twice.
4. **Pinned models**: each catalogue model is pinned to a Hugging Face commit and a sha256, downloaded into the
   Plugin's data directory by the first Task that needs it. Nothing downloads at install, nothing is bundled, and
   nothing unpinned ever loads.

A Flow over a folder:

```
ListFolder → VideoFiles → map(entries) → VideoThumbnail (frame sheet) → ClassifyFrames (classifier: style) → Edges on outcome.result.tag
```

## Implementation Decisions

### Plugin

- Package `fyflow-plugin-image-tags`, id `fyflow-ext.image-tags`, `dataDir: true`, peer `fyflow-plugin-sdk ^0.11.0`
  (Python Workers from 0.8.0; `grants` for the GPU device from 0.9.0; `pickVariant`, holds per build and `host.pool`
  from 0.11.0, where a Host before 0.11 refuses `pool.variants`).
- Layout follows the SDK README's Python Worker layout and fyflow-app's `examples/plugins/word-counter`:
  `package.json`, `plugin.ts` (definition, schemas, Node Types, `status()`), `python/pyproject.toml`,
  `python/uv.lock`, `python/image_tags/…`. `pnpm plugin-folder` makes a folder the app installs, as in
  `fyflow-plugin-video-thumbnails`.
- Python dependencies: `onnxruntime` (per variant), `pillow`, `numpy`. **No imgutils at runtime**: its core pulls in
  opencv-contrib, scikit-learn, scipy, pandas, shapely, tokenizers and pins `numpy<2`; it pip-installs onnxruntime at
  import when missing; it downloads with no `revision`; it always uses CUDA device 0; it scores one image per call.
- `variants` (conflicting extras in `pyproject.toml`):
  - `cpu`: `onnxruntime`
  - `cuda`: `onnxruntime-gpu[cuda,cudnn]` (CUDA and cuDNN as pip packages; `onnxruntime.preload_dlls()` before the
    first session; the driver must support the wheel's CUDA major, 13.x today)
  - `directml`: `onnxruntime-directml`
- The Plugin picks the variant per machine (`python.pickVariant`, in `pick-build.ts`): `cuda` with an NVIDIA driver
  (Windows: `%SystemRoot%\System32\nvcuda.dll` of driver 580.88 or later (CUDA 13's), read from its `VS_FIXEDFILEINFO`, or any
  whose version can't be read; Linux: `/proc/driver/nvidia/version` of 580.65.06 or later, else `libcuda.so.1` in the
  usual places), else `directml` on Windows, else `cpu`; never an error, `cpu` on any. A Profile overrides it (the
  Default Profile's for every Node), so one Flow can run some Nodes on the GPU and others on the CPU. The number of
  GPUs is the built-in `gpu` group's units (1 by default), set on the Resource Groups page. The README explains both.
- Reads only what the Task's Allowed roots allow (fyflow-app's, ADR-0030; `roots: true`, SDK 0.11.0): each Node
  Type checks its input `path` with `ctx.allowed_path(path, "read")` before reading anything, and reads where it really
  leads. A refused path fails the Task with the check's message. `ClassifyFrames` also checks each image of a folder,
  and skips one refused (a link leading out of the reach) with a warning. A sheet's `video` is metadata, never read,
  so not checked; nor are `modelDir` (a Profile's own field) and the data directory.

### Input

- `path` in the Task input only (one Node classifies many images): an absolute path. `path` is also what
  `VideoThumbnail` hands on, so the two chain with no transform.
- Any image Pillow opens (JPEG, PNG, WebP, BMP, …). Converted to RGB on a white background.
- A **frame sheet** is a JPEG whose comment (COM) segment holds the thumbnail service's JSON with a `frames` array.
  Each non-null frame is cut out at its `rect` (`{x, y, w, h}`, the destination rect on the canvas); a null frame
  (not decoded, or a cell with no timestamp) is left out. The frames' `timestamp_actual` and the JSON's `video`
  reach the result.
- A sheet whose `config.render_timestamps` is true is still classified, with a `warnings` entry: the drawn
  timestamps sit on the frames the models see.
- **`ClassifyFrames`** takes a frame sheet or a folder:
  - A folder: every image directly inside it (not recursive), sorted by name, each one frame with
    `timestamp: null`. A frame sheet inside a folder is a plain image. More than 1000 images fails with
    `The folder has more than 1000 images:`; an empty one with `The folder has no images:`. Large sets belong in a
    Flow `map` over `ClassifyImage`.
  - Any other single image is one frame.
- **`ClassifyImage`** takes one image. A frame sheet given to it is scored as one image (the collage), with a
  `warnings` entry saying `ClassifyFrames` reads sheets.
- Recommended `VideoThumbnail` settings, given in the README: a `grid` of 9 to 16 cells, `fit: "stretch"` (what
  imgutils does: a squash to the model's size, no crop, no bars), `spacing` and `margin` 0, timestamps off,
  `skipBlankFrames` on, `startOffset`/`endOffset` past openings and credits. The cell size need not match the model:
  each frame is resized to the model's input anyway.

### Catalogue and preprocessing

- A **model** entry: `id` (e.g. `anime_real`), `kind: "model"`, title, the repo, commit and file names, each file's
  sha256, input size, labels (checked against the model's `meta.json` when the entry is written), and its default
  `minConfidence` (0: a raw model never answers `uncertain` unless asked).
- A **preset** entry: `id`, `kind: "preset"`, title, its labels, the models it needs, the rule (a Python function
  over those models' per-frame scores), and its default `minConfidence`.
- `kind: "tagger"` is reserved.
- v1 entries:
  - `anime_real`: `deepghs/anime_real_cls` @ `097a6c1e9e62866a435810c883a45a7cb2c7077d`, `mobilenetv3_v1.4_dist`.
  - `anime_classification`: `deepghs/anime_classification` @ `5ee62e06f5f4cd68a1c2f3bc5dc9805e827d37df`,
    `mobilenetv3_v1.5_dist` (imgutils' default; the eval set chose it over v1.3: it finds more anime).
  - `style` (preset, default classifier): labels `anime`, `cg`, `live_action`; default `minConfidence` 0.55 (settled
    by the eval set). Per frame, from `anime_real`'s `p_real` and `anime_classification`'s scores with `not_painting`
    dropped and the other four renormalised:
    - `live_action = p_real⁴` (only a very sure `p_real` counts: the eval's `real_pow4`)
    - `cg = (1 − p_real⁴) · p_3d`
    - `anime = (1 − p_real⁴) · (1 − p_3d)`, i.e. the share of `bangumi`, `illustration` and `comic`
    - When the four sum to less than 1e-6, `cg` and `anime` split `1 − p_real⁴` evenly.
- Preprocessing is our own, matching imgutils' fallback (`_img_encode`; neither model ships a `preprocess.json`):
  RGB on white, bilinear resize to the model's input W×H ignoring aspect (384×384 if the input is dynamic),
  scale to [0, 1], `(x − 0.5) / 0.5`, float32 NCHW, input tensor `input`, output tensor `output`. The exact input
  size, whether the batch dimension is dynamic, and whether `output` is already probabilities or needs a softmax are
  read from the files and settled by the golden test before the entries are written.

### Models

- Downloaded on first use, inside `run()`, not `setup()`, so the download reports progress and is cancelled through
  the Task.
- Standard-library HTTPS (`urllib`) from `https://huggingface.co/<repo>/resolve/<commit>/<name>/<file>`, no
  `huggingface_hub`. `HF_ENDPOINT` in the Host's environment names a mirror instead (the tests point it at a local
  server). One download per model per process, shared by Tasks that need it at once; progress through
  `ctx.progress`; Cancel stops it. Written to a temp file, checked against its sha256, then renamed into
  `<dataDir>/models/<repo>/<commit, first 12 characters>/<name>/`, so a half-written file is never loaded and
  processes of different pools or Profiles cannot collide. The commit is cut to 12 characters to stay under Windows'
  260-character paths when long paths are off; the catalogue and the download URL keep the full commit.
- A failed download or a sha256 mismatch fails that Task only, with a message starting
  `Could not get the model <id>:`; the next Task tries again.
- Profile start-up field `modelDir`: a folder holding the same layout, for an offline machine. When set, nothing is
  downloaded and a missing model fails the Task, naming the folder.
- `status()`: `ok` with the models present, or `info` "Models download on first use" with the missing ones listed.
- The deepghs repos are tagged OpenRAIL: fine to download for the user's own use; read before ever bundling the
  files into the Plugin folder.

### Settings

- `classifier`: an enum of the catalogue's models and presets, with titles; default `style`.
- `minConfidence`: optional, 0 to 1. Left empty, the chosen entry's own default applies. `tag` is the top label when
  its confidence reaches `minConfidence`, else `uncertain`. Low confidence is a result, never a failure.

### Results

- Both Node Types: `{ path, video?, classifier, labels, tag, confidence, scores: { <label>: number }, models: [ids],
  warnings? }`. `labels` is the classifier's label list; `tag` is one of them or `uncertain`.
- `ClassifyFrames` adds `agreement` (the share of frames whose own top label is `tag`) and `frames: [{ timestamp,
  tag, scores }]`. Its `scores` are the mean over frames; `confidence` is `tag`'s mean score. `frames` lets a Flow
  find mixed content (live action with anime inserts) without the Node guessing at it.
- `outputSchema`: `tag` is a string whose description lists every classifier's labels; `scores` is "numbers by
  label". The `examples` show `style` results, so the editor's sample and suggested paths (`result.tag`) are
  realistic.

### Running

- Both Node Types are Python Workers, so not `inline`. Pool on each Node Type:
  - `ClassifyFrames`: `maxThreads: 1`, `maxConcurrentTasks: 2` (one Task's decoding and cropping overlaps another's
    inference), `residentGroups: ["gpu"]`, `idleTimeout: 5_000`.
  - `ClassifyImage`: `maxThreads: 1`, `maxConcurrentTasks: 32`, `residentGroups: ["gpu"]`, `idleTimeout: 5_000`.
- Pools are per (Node Type, Profile), each its own Python process: each loads its own sessions (~17 MB per model)
  and holds its own `gpu` unit. With one unit, the two Node Types take turns on the GPU; the short `idleTimeout`
  keeps the wait short. The README says so.
- Under the `cpu` variant the Node Types hold no `gpu` unit: each pool's `variants: { cpu: { residentGroups:
  { cpu: 4 } } }` replaces its holds there.
- The Worker reads the pool options it runs under at `config["host"]["pool"]`: onnxruntime's threads are
  `residentGroups.cpu.cost` (else 4), `ClassifyImage`'s blocking thread pool is sized from `maxConcurrentTasks` (else
  32). `config["host"]["variant"]` is only said in the session's provider line on stderr; the providers are what the
  installed build offers.
- The device is `pool.grants["gpu"][0]`, passed as the provider's `device_id` (CUDA) or `device_id` (DirectML). The
  Host sets no `CUDA_VISIBLE_DEVICES`.
- One inference session per model per process, made on first use and kept until the instance retires.
- `ClassifyFrames`: frames are run in batches of 32, `ctx.progress` advancing and the cancel check running between
  batches.
- `ClassifyImage`: `async def run`. Each Task decodes and preprocesses its image with `asyncio.to_thread`, then hands
  the tensor to a per-model batcher, which runs a batch at 16 frames or 20 ms after the first arrived, whichever comes
  first, off the event loop. Each Task awaits its own result through `asyncio.shield`, so a cancelled Task (the shim
  fails it at once and cancels its coroutine) never cancels a shared batch: a queued frame of a cancelled Task is
  dropped, one in a running batch has its result discarded. Batch size and window are constants until the eval says
  otherwise.

### Failures

- Not an image, or unreadable: fails, naming the path.
- A sheet with no decodable frame: fails with `The frame sheet has no frames:`.
- A folder with no images, or too many: fails (above).
- A model that cannot be had: fails with `Could not get the model <id>:` (above).
- Low confidence is a result (`uncertain`), never a failure.

## Testing Decisions

- `pnpm check`: TypeScript typecheck and lint, a TypeScript test that the Plugin's declarations load and validate
  with the SDK, and `uv run pytest` in `python/`. uv is required for development, as it is on the Host.
- **Golden tests per catalogue model**: `python/scripts/golden.py` (`pnpm golden`) runs in an environment of its
  own from PEP 723 inline metadata, so imgutils and its `numpy<2` never reach the Plugin's lock. It checks each entry
  against the model's files (labels against `meta.json`, input size, a dynamic batch, probabilities out), runs
  imgutils on a few generated images and writes `python/tests/fixtures/golden.json`; pytest checks our preprocessing
  and sessions give the same top label and every score within 0.02. Regenerated by hand when an entry changes;
  imgutils is never needed for `pnpm check`.
- **Frame sheets**: a sheet made by the video-thumbnail Plugin from its `sample.mp4` is committed as a fixture; the
  right number of frames is cut at the right rects with their timestamps; a sheet with timestamps drawn gets its
  warning; a plain image is one frame; a folder gives its images in name order.
- **Models**: downloads from a local HTTP server: one download for concurrent Tasks, progress reported, Cancel stops
  it, a sha256 mismatch fails the Task and leaves no file, `modelDir` downloads nothing.
- **`style` and `uncertain`**: the per-frame rule, the aggregation and the threshold as unit tests on scores.
- **Batcher**: flushes at size and at the window; a cancelled Task leaves the other Tasks' results intact.
- **End to end**: one `ClassifyFrames` Task on the fixture sheet and one `ClassifyImage` Task, on the `cpu` variant.
- **Eval set** (by hand, not in `pnpm check`; part of v1, run before a release is tagged):
  `python/scripts/eval.py` takes a CSV of `path,label` (the user's videos labelled anime, cg, live_action; 30 to 50
  each) and the `thumbs-collage` binary's path, makes a sheet per video with the README's recommended settings, runs
  `style` and the raw models, and prints confusion matrices, the time per video and the `minConfidence` that best
  separates the classes. It decided (2026-11-10) the `anime_classification` version (v1.5), `style`'s rule (live
  action `p_real⁴`) and default `minConfidence` (0.55), and that `cg` needs another model (ticket 08); it keeps the
  configuration before that (`previous`: `p_real`, v1.3, 0.6) to compare a later change against.

## Dependencies

- **Frame-sheet contract in `fyflow-plugin-video-thumbnails`**: the `thumbs-collage` service writes `frames[].rect`,
  `video` and `config.render_timestamps` into the COM JSON, but that Plugin reads only the timestamps. Type those
  fields in its `Metadata` and test them on its fixture, so a change to the service breaks a test there first. Its
  own issue in that repo.

## Open Questions

1. **Does `3d` catch the user's CG?** The eval set answers it; if not, a zero-shot model (CLIP/SigLIP) is the next
   candidate, as a catalogue model feeding the `style` preset's rule (its labels stay the same).

## After v1: one Node Type per kind of answer

Planned 2026-11-12, after 0.1.0. Not yet ticketed.

A model fits the Plugin by the shape of its answer, not by what it is. One shared layer loads catalogue models and
runs batched forward passes: model fetching, devices, builds, pools, frame reading and Allowed roots, most of which
v1 has. On it sits one Node Type pair (an image and its frames) per kind of answer:

| Answer | Node Types | Models |
| --- | --- | --- |
| one label | `ClassifyImage` / `ClassifyFrames` (v1) | `anime_real`, `anime_classification`; CLIP/SigLIP zero-shot |
| many tags | `TagImage` / `TagFrames` | WD v3 taggers (SmilingWolf), JoyTag |
| a vector | `EmbedImage` / `EmbedFrames` | OpenCLIP, SigLIP |
| text | not this Plugin | vision-language models such as Gemma |

- **Each model's preprocessing becomes its own,** in its catalogue entry: v1's single path (384×384) doesn't serve
  WD's (448×448, padded square on white, BGR, NHWC, 0–255) or JoyTag's (448×448, CLIP mean/std, NCHW).
- **Taggers** (`kind: "tagger"`, reserved in v1): an ONNX file, its tag list (WD's `selected_tags.csv`), its
  preprocessing and its categories (WD: general, character, rating; JoyTag: general only). Each tag is scored on its
  own (sigmoid), so the result is the tags over a threshold, `[{ name, score, category }]`, never every tag's score.
  Node settings: `tagger`, a threshold per category (WD's usual: general about 0.35, character about 0.85), perhaps
  `maxTags`. They act after the forward pass, so Tasks with different thresholds still share a batch. `TagFrames`
  gives each frame its tags and a summary over the frames: each tag's share of frames, or its mean score. Ratings
  may be their own small output, for routing in a Flow. JoyTag ships as PyTorch: export it to ONNX, or pin a
  conversion. The models are about 380 MB (WD ViT, SwinV2) to 1.2 GB (WD EVA02-large), so slow per frame on a CPU.
- **Zero-shot classification** (CLIP/SigLIP) is a `classifier` of `ClassifyImage`, no new Node Type: with prompts
  fixed in the catalogue, the text side's vectors are computed once; with labels a Node sets, the text encoder
  runs in the Worker too, and the settings depend on the chosen classifier (see "Settings that depend on the chosen
  model" in `todo.md`). It is the candidate for CG vs live action (Open Question 1).
- **Embeddings:** a vector per frame (512 to 1152 numbers) and one pooled over the frames, for similarity,
  duplicates and clustering. A Flow needs somewhere to keep them, which fyflow has nothing for yet.
- **Vision-language models** generate text: a tokenizer, a decoding loop, several GB, an engine other than one
  batched forward pass (onnxruntime-genai, llama.cpp, a local server such as Ollama). They take seconds per frame and
  vary run to run. Their own Plugin, with a `prompt` setting, e.g. calling a local model server.
- **Licences to check per model before picking it:** the WD v3 taggers' and JoyTag's model cards (believed
  Apache-2.0), each OpenCLIP checkpoint's (the code is MIT, the weights vary), SigLIP's (Apache-2.0), and Gemma's
  own terms. The README's "Licence and credits" names each one shipped.

## Out of Scope

For v1; "After v1" above plans taggers, zero-shot models and embeddings.

- Taggers and any multi-label output (the catalogue reserves `kind: "tagger"`).
- Models a user adds by hand beyond `modelDir`'s copies of catalogue entries.
- Zero-shot models, detection, embeddings.
- Pulling frames from video, and scene detection: the video-thumbnail Plugin's (and its service's) job.
- Frame sheets inside a folder (read as plain images).
