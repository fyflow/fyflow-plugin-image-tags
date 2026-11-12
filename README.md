# fyflow-plugin-image-tags

A fyflow-app Plugin (`fyflow-ext.image-tags`) that tells what an image is with small local ONNX classifiers: anime,
CG or live action, of one image or of a whole video from its frames (see `.scratch/image-tags/spec.md`).

Needs a Host with `fyflow-plugin-sdk` 0.11.0 or later and Python Workers (uv). The Host makes the Plugin's Python
environment from `python/uv.lock` on install; nothing else is set up by hand.

## Node Types

### `ClassifyImage`

One image per Task (`path` in its input: anything Pillow opens), through one classifier (the `classifier` setting):
a **preset**, whose labels are computed by a rule from the models under it and stay the same when those models
change, or a **model** as it is, whose labels are the model's own (switching models means rewriting Edges).

| Classifier | Labels | From | Default `minConfidence` |
| --- | --- | --- | --- |
| `style` (default, preset) | anime, cg, live_action | `live_action` = anime_real's `real`⁴ (only a very sure `real` counts); what is not live action is shared out by anime_classification, `not_painting` dropped and the other four made to sum to 1: `cg` gets `3d`'s share, `anime` that of `bangumi`, `illustration` and `comic` | 0.55 |
| `anime_real` (model) | anime, real | `deepghs/anime_real_cls` / `mobilenetv3_v1.4_dist` | 0 |
| `anime_classification` (model) | 3d, bangumi, comic, illustration, not_painting | `deepghs/anime_classification` / `mobilenetv3_v1.5_dist` | 0 |

`style` is made for anime first: measured on a home library (see [Eval](#eval)), it finds every anime video and tags
nothing else anime. Telling CG from live action is weaker: anime_classification's `3d` comes from 3D-style
illustrations and it calls live-action footage `3d` too, so a CG film anime_real takes for real comes out live action,
and about half of live action comes out `uncertain`. If anime_classification is all but sure an image is not a
painting at all (the four sum to less than 1e-6), what is not live action is split evenly between `anime` and `cg`.

It hands on `{ path, classifier, labels, tag, confidence, scores, models, warnings? }`: `scores` holds each label's
score, `confidence` the top one's, and `models` the catalogue models the classifier ran (both, for `style`). `tag` is the top
label when its score reaches **`minConfidence`** (a setting from 0 to 1; left empty, the classifier's own default
above), else **`uncertain`**: a result, never a failure. So a Flow routes on `outcome.result.tag` with plain Edge
conditions, `uncertain` included. A raw model's default is 0: it answers `uncertain` only when a Node asks for a
`minConfidence`.

Given a frame sheet, it scores the whole collage as one image and adds a `warnings` entry: `ClassifyFrames` reads
sheets.

It is made for a Flow's `map` over many images. One process per Profile runs up to 32 Tasks at once, and the images
of the Tasks running together go through each model in shared batches: a batch runs once it holds `batchSize` images
(16 unless the Profile says otherwise), or 20 ms after its first one arrived, whichever comes first, and one batch of
a model runs at a time (images arriving meanwhile wait for the next). Tasks with different classifiers share the batches of the models they have in common. Each Task
reads its own image, so one that cannot be read fails alone; a model that cannot be had fails the Tasks that need it.
A cancelled Task leaves its batch at once, and the other Tasks' results are untouched.

### `ClassifyFrames`

A whole video per Task, from its frames: the same `classifier` and `minConfidence` settings, and `path` in its input
being one of

- a **frame sheet** made by the video-thumbnail Plugin's `VideoThumbnail` (`fyflow-ext.video-thumbnail`): a JPEG
  whose comment holds where each frame was drawn and its timestamp. Each frame is cut out at its place and
  classified on its own; a cell with no frame is left out, and a sheet with none fails with
  `The frame sheet has no frames:`. A sheet with timestamps drawn on it is still classified, with a `warnings`
  entry: the models see the digits.
- a **folder**: every image directly inside it (not in its subfolders), in name order, each one frame. A frame sheet
  in a folder is a plain image. More than 1000 images fail with `The folder has more than 1000 images:` (a large
  set belongs in a Flow `map` over `ClassifyImage`), none with `The folder has no images:`.
- any other **image**: one frame.

So `VideoThumbnail → ClassifyFrames → Edges on outcome.result.tag` needs no Transform: `VideoThumbnail` hands on the
sheet's `path`.

It hands on what `ClassifyImage` does, over all the frames, and more:

- `scores`: each label's mean over the frames; `tag` the top one (or `uncertain` below `minConfidence`), and
  `confidence` its mean score;
- `agreement`: the share of frames whose own top label is the tag (the top label, when the tag is `uncertain`), from 0
  to 1;
- `frames`: each frame's `{ timestamp, tag, scores }`, in the sheet's or folder's order, `timestamp` (seconds into
  the video) null for a folder or an image. A Flow can find mixed content in it (live action with anime inserts);
- `video`: the video's path, as the sheet has it.

Frames go through the models `batchSize` at a time (16 unless the Profile says otherwise); the Task's progress moves
on, and a Cancel is looked at, after each batch.
One process per Profile runs two Tasks at once, so one Task's decoding and cutting overlaps the other's inference.

#### Recommended `VideoThumbnail` settings

| Setting | Value | Why |
| --- | --- | --- |
| `collage` | `{ mode: "grid", rows: 3, cols: 3 }` to `{ rows: 4, cols: 4 }`, `spacing: 0`, `margin: 0` | 9 to 16 frames: enough to outvote an odd one, few enough to stay quick. Each frame is resized to the model's input anyway, so the cell size need not match it. |
| `selection` | `{ mode: "uniform", frames: <rows × cols>, startOffset, endOffset }`, the offsets past the openings and credits | Openings and credits are often drawn differently from the rest. |
| `fit` | `"stretch"` | What imgutils does: the whole frame squashed into the cell, no crop, no bars. |
| `renderTimestamps` | `false` | Drawn timestamps sit on the frames the models see (a warning says so). |
| `skipBlankFrames` | `true` | A black or white frame says nothing. |

## Resources

Both Node Types hold what they use per process, not per Task: while its process lives (and for 5 s after its last
Task), each pool holds a unit of the Host's `gpu` group and 4 of its `cpu` group, and a Task holds nothing of its own.
Tasks waiting on a shared forward pass use no processor, and a forward pass runs on 4 threads. On the `cpu` build a
pool holds the 4 `cpu` units and no `gpu` unit. With one GPU unit, the two Node Types' GPU pools, and each Profile's,
take turns: a pool waits for the other to idle out, up to about 10 s (its 5 s idle time and the Host's 5 s more).
That suits the intended use, many Tasks of one Node Type and then many of the other, rather than Tasks alternating
between them. A Profile can change the idle time (`idleTimeout`) for either Node Type.

A Profile can also change what a pool holds (`residentGroups`, e.g. `{ cpu: 8 }`) and how many Tasks it runs at once
(`maxConcurrentTasks`). The Worker reads what fyflow booked for its process: a forward pass runs on as many threads
as the `cpu` units it holds, and `ClassifyImage` makes room for as many Tasks at once as its pool runs.

### On a GPU

The Plugin picks its build on each machine, at the Host's start and at each Reload: `cuda` with an NVIDIA driver
(Windows: `nvcuda.dll` of driver 580.88 or later; Linux: driver 580.65.06 or later, CUDA 13's, or `libcuda.so.1` where no version
file says), else `directml` on Windows, else `cpu`. The Host then makes the Plugin's Python environment with that
build of ONNX Runtime (`pyproject.toml`'s extras):

| Build | Installs | Needs |
| --- | --- | --- |
| `cpu` | `onnxruntime` | nothing |
| `cuda` | `onnxruntime-gpu` with CUDA and cuDNN as pip packages (about 2 GB installed) | an NVIDIA driver that supports CUDA 13; no CUDA install |
| `directml` | `onnxruntime-directml` (about 100 MB installed) | Windows with a DirectX 12 GPU |

To run another build, set it on a Profile, in this Plugin's section on the Profiles page: on the Default Profile for
every Node, or on a Profile of its own for some Nodes, so that one Flow can run some Nodes on the GPU and others on
the CPU. The Host makes an environment of each build a Profile uses, and takes the change at the next Reload.

The Plugins page shows the picked build on the Plugin's Python line once its environment is Ready. Each Worker
process says on its "Python output" where each model's session runs, e.g. `image-tags: the model anime_real runs on
CUDAExecutionProvider (gpu unit 0: device_id=0); the cuda build, onnxruntime 1.31.1, 4 threads`. A CUDA build whose
driver is broken or too old falls back to the CPU, and the line says so.

The GPU a process uses is the unit of the Host's `gpu` group it holds. That group has 1 unit unless its number is set
on the Resource Groups page, which a Host with more GPUs needs: unit `n` is CUDA's device `n`. DirectML numbers
adapters otherwise (a laptop's integrated GPU first), so with `directml` unit 0 is the fastest GPU Windows reports,
and unit `n` above 0 is DirectML's adapter `n`.

With one `gpu` unit, `ClassifyImage` and `ClassifyFrames` take turns on the GPU, as above; on the `cpu` build they
hold no `gpu` unit, so they run beside each other and beside GPU pools. Set [`batchSize`](#batchsize) for the GPU's
memory.

### `batchSize`

A Profile start-up field (read when a Worker starts, so it takes effect at the next Reload): the most frames in one
forward pass, 16 if left empty. It is for the GPU, where the best size depends on the GPU's memory. Raising it on
`ClassifyImage` helps only if that Profile's `maxConcurrentTasks` for `ClassifyImage` is at least as high, since a
batch holds one image per Task running.

## Models

Each model is pinned to a Hugging Face commit and its file's sha256, and downloaded into the Plugin's data
directory by the first Task that needs it (about 17 MB each); nothing downloads at install, and nothing else ever
loads. A mirror of Hugging Face can be named with `HF_ENDPOINT` in the Host's environment. `anime_real_cls` is
OpenRAIL-licensed, `anime_classification` MIT.

A download shows as the Task's progress ("Downloading the model anime_real"). Tasks of one Node Type and Profile that
need a model at once share one download; a cancelled Task stops waiting at once, and the download itself stops when
every Task waiting on it is cancelled. A failed download, or a file that is not the pinned one, fails the Tasks that
waited on it with `Could not get the model <id>:` and leaves nothing behind; the next Task tries again.

The Plugin's status on the Plugins page says which models are in its data directory: `ok` once all are, else
"Models download on first use" with the missing ones. It looks only in the data directory, not in a Profile's model
folder (a Plugin's status is not handed the Profile).

### Without internet: `modelDir`

On a machine that cannot reach Hugging Face, set **`modelDir`** on the Plugin's section of a Profile to an absolute
folder holding the models, and Reload (it is read when a Worker starts). Then nothing is ever downloaded: a model
missing from the folder fails the Task with `Could not get the model <id>:`, naming the folder and the path it
expected. Each file is checked against its pinned sha256 before it loads.

The folder has the layout of the data directory's `models` folder:
`<repo>/<commit, first 12 characters>/<name>/model.onnx`, each file downloaded on another machine from
`https://huggingface.co/<repo>/resolve/<commit>/<name>/model.onnx`:

| Model | Put at (below `modelDir`) | Download from |
| --- | --- | --- |
| `anime_real` | `deepghs/anime_real_cls/097a6c1e9e62/mobilenetv3_v1.4_dist/model.onnx` | https://huggingface.co/deepghs/anime_real_cls/resolve/097a6c1e9e62866a435810c883a45a7cb2c7077d/mobilenetv3_v1.4_dist/model.onnx |
| `anime_classification` | `deepghs/anime_classification/5ee62e06f5f4/mobilenetv3_v1.5_dist/model.onnx` | https://huggingface.co/deepghs/anime_classification/resolve/5ee62e06f5f4cd68a1c2f3bc5dc9805e827d37df/mobilenetv3_v1.5_dist/model.onnx |

A machine that has run the Plugin online already holds this folder: its data directory's `models` folder can be
copied whole.

## Development

Needs Node 26, pnpm and [uv](https://docs.astral.sh/uv/).

- `pnpm check`: typecheck, lint, the declarations test (vitest) and the Python tests (pytest, through uv). The first
  run fetches the pinned models into `python/.cache/`.
- `pnpm plugin-folder [dir]`: the folder to install from the app's Plugins page ("Choose folder…"), by default
  `dist/image-tags/`.
- `pnpm sheets`: remakes the frame-sheet fixtures (`python/tests/fixtures/sheets/`) with the video-thumbnail Plugin's
  `thumbs-collage` service and test clip, from `../fyflow-plugin-video-thumbnails` (or `-- --thumbnails <folder>`).
- `pnpm golden`: remakes the synthetic test images (the small and see-through ones) and writes imgutils' scores for
  them and the reference frames (`python/tests/fixtures/golden.json`), in an environment of its own; run it by hand
  when a catalogue model changes.
- `pnpm eval`: measures the classifiers on labelled videos and images; run by hand before a release is tagged (see
  [Eval](#eval)).

### Eval

`python/scripts/eval.py` measures `style` as shipped (its rule over `anime_classification` v1.5, at its default
`minConfidence`) beside what it replaced (`previous`: live action `real` itself, v1.3, 0.6) and a candidate rule, so a
run shows what a change to the catalogue would change. It needs no Host.

```sh
pnpm eval --references                                   # the reference frames alone
pnpm eval -- eval.csv --collage <thumbs-collage>         # your videos
pnpm eval -- eval.csv --references --collage <thumbs-collage>
```

The CSV holds `path,label` rows (a header is optional, relative paths are from the CSV's folder), labels `anime`, `cg`
or `live_action`. A path may be a video, an image or a folder of images. A video's frame sheet is made by the
video-thumbnail Plugin's `thumbs-collage` service (`--collage`, e.g.
`../fyflow-plugin-video-thumbnails/bin/win32-x64/thumbs-collage.exe`) with the recommended settings above: a 3x3 grid
(`--grid 4` for 4x4), cells 384x216, `fit: "stretch"`, no timestamps, blank frames skipped, and 0.5 s skipped at each
end (`--start-offset` and `--end-offset`, in seconds: set them past the openings and credits). The sheets go to
`python/.cache/eval/sheets/`, the models to `python/.cache/eval/models/`. `--references` adds the reference frames,
labelled by their names.

Each item goes through `ClassifyFrames` itself (the cpu build), and its frames through every model, `anime_classification`
in both versions (v1.5, the catalogue's, and v1.3, the one before). It prints:

- `style`'s confusion matrix at its default `minConfidence`, `uncertain` a column of its own, and the previous
  configuration's at 0.6;
- each raw model's, with its accuracy in `style`'s terms: `anime_real`'s `real` as live action (it has no `cg`: its
  CG row shows where CG goes, but is left out of its accuracy), `anime_classification`'s `3d` as CG, `bangumi`,
  `comic` and `illustration` as anime, `not_painting` as live action;
- `style`'s rule as shipped (live action `real⁴`) beside `previous` (live action `real`) and `3d_first` (a strong
  `3d` overrides `real`: `cg` is `3d`'s score, the rest split by `anime_real`), each with either
  `anime_classification` version;
- a table of thresholds against accuracy on what is answered and coverage (the share not `uncertain`), with a pick
  per rule: the best accuracy among the thresholds that answer at least `--min-coverage` (90%) of the items, the
  lowest of equals;
- the time per item, making the sheet and classifying apart;
- anime against everything else per rule and threshold: anime items found, and items that are not anime tagged
  anime (anime matters most: `cg` and `live_action` are good to have, never at anime's cost);
- the items any rule gets wrong, with each rule's tag and confidence.

The reference frames are too few to decide anything. The decisions want your own videos, 30 to 50 per label, in the
kinds you mean to sort; set the offsets shorter than your shortest video (a sheet of a video shorter than both offsets
fails with `invalid offsets`).

What it decided (2026-11-10, 132 videos of a home library, 56 anime, 48 CG, 45 live action, plus the reference
frames; `--grid 4 --start-offset 120 --end-offset 420`), all now in the catalogue:

- **`anime_classification` v1.5** over v1.3: it finds more anime (the photographic backgrounds of Shinkai's films).
- **`style`'s rule became `real_pow4`** (live action `real⁴`: the script's `shipped` now; the old rule is its
  `previous`), and its default `minConfidence` **0.55**. Anime against the rest: every
  anime video found and none of the others tagged anime, from 0.4 to 0.6. CG: 36 of 48 right at 0.6, where the old
  rule had 21 (most of the rest `uncertain`). Live action: never wrong, but about half of it `uncertain`.
- **Telling CG from live action needs another model:** `anime_classification` calls every live-action video `3d`, and
  `anime_real`'s `real` overlaps between CG films (0.4 to 0.73) and live action (0.6 to 0.95), so no rule over these
  two models separates them well.
- **"anime" is 2D drawn animation:** Western cartoons (Rick and Morty, The Little Mermaid, Ernest & Celestine) come
  out anime, Chinese donghua and 3D CG anime too; stop-motion and puppets come out live action.

### Test images

`python/tests/fixtures/images/` holds real reference frames (`<category>-NN-<what>.jpg`: anime, CG and live action,
resized to 640 px) besides the synthetic images. They are not under this repository's MIT licence but each under its
own: MIT (the AI-generated anime frames), CC BY 2.5, 3.0 and 4.0, and CC BY-SA 4.0 (shared under the same licence).
Their authors, sources, licences and changes are in that folder's `CREDITS.md`, which goes with them.
