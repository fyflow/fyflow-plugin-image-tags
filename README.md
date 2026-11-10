# fyflow-plugin-image-tags

A fyflow-app Plugin (`fyflow-ext.image-tags`) that tells what an image is with small local ONNX classifiers: anime,
CG or live action, of one image or of a whole video from its frames (see `.scratch/image-tags/spec.md`).

Needs a Host with `fyflow-plugin-sdk` 0.9.0 or later and Python Workers (uv). The Host makes the Plugin's Python
environment from `python/uv.lock` on install; nothing else is set up by hand.

## Node Types

### `ClassifyImage`

One image per Task (`path` in its input: anything Pillow opens), through one classifier (the `classifier` setting):
a **preset**, whose labels are computed by a rule from the models under it and stay the same when those models
change, or a **model** as it is, whose labels are the model's own (switching models means rewriting Edges).

| Classifier | Labels | From | Default `minConfidence` |
| --- | --- | --- | --- |
| `style` (default, preset) | anime, cg, live_action | `live_action` = anime_real's `real`; what is not real is shared out by anime_classification, `not_painting` dropped and the other four made to sum to 1: `cg` gets `3d`'s share, `anime` that of `bangumi`, `illustration` and `comic` | 0.6 |
| `anime_real` (model) | anime, real | `deepghs/anime_real_cls` / `mobilenetv3_v1.4_dist` | 0 |
| `anime_classification` (model) | 3d, bangumi, comic, illustration, not_painting | `deepghs/anime_classification` / `mobilenetv3_v1.3_dist` | 0 |

`style`'s `3d` comes from 3D-style illustrations, so how well it tells CG footage apart is still to be measured. If
anime_classification is all but sure an image is not a painting at all (the four sum to less than 1e-6), what is not
real is split evenly between `anime` and `cg`. Its default threshold, and the anime_classification version, are
provisional until they are measured on real videos.

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
Tasks waiting on a shared forward pass use no processor, and a forward pass runs on 4 threads. The `gpu` unit is held
on the `cpu` variant too (a pool cannot change by variant), so with one GPU unit the two Node Types, and each Profile's
pools, take turns: a pool waits for the other to idle out, up to about 10 s (its 5 s idle time and the Host's 5 s
more). That suits the intended use, many Tasks of one Node Type and then many of the other, rather than Tasks
alternating between them. A Profile can change the idle time (`idleTimeout`) for either Node Type.

The 4 is the Plugin's: fyflow-app does not yet let a Profile change what a pool holds while it lives, nor tell the
Worker how many `cpu` units it holds. When it does (`pool.holds`), the Worker takes its thread count from there.

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
| `anime_classification` | `deepghs/anime_classification/5ee62e06f5f4/mobilenetv3_v1.3_dist/model.onnx` | https://huggingface.co/deepghs/anime_classification/resolve/5ee62e06f5f4cd68a1c2f3bc5dc9805e827d37df/mobilenetv3_v1.3_dist/model.onnx |

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

### Test images

`python/tests/fixtures/images/` holds real reference frames (`<category>-NN-<what>.jpg`: anime, CG and live action,
resized to 640 px) besides the synthetic images. They are not under this repository's MIT licence but each under its
own: MIT (the AI-generated anime frames), CC BY 2.5, 3.0 and 4.0, and CC BY-SA 4.0 (shared under the same licence).
Their authors, sources, licences and changes are in that folder's `CREDITS.md`, which goes with them.
