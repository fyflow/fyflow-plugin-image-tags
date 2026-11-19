# fyflow-plugin-image-tags

A fyflow-app Plugin (`fyflow-ext.image-tags`) that tells what an image is with small local ONNX classifiers: anime,
CG or live action, of one image or of a whole video from its frames; and tags an image, or each frame of a video,
with a local tagger: characters, general tags and a rating (see `.scratch/image-tags/spec.md`).

Needs a Host with `fyflow-plugin-sdk` 0.11.0 or later and Python Workers (uv); 0.12.0 or later adds the
[Data](#data) link. Install it from **Browse** on the
app's Plugins page (the official Plugin repository), or upload the `.tgz` of a
[Release](https://github.com/fyflow/fyflow-plugin-image-tags/releases). The Host makes the Plugin's Python environment
from `python/uv.lock` on install; nothing else is set up by hand. Before the first Task, give a Profile a read
[Allowed root](#allowed-roots) over the images it should see: the Plugin refuses every path until then.

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
and about half of live action comes out `uncertain`. "Anime" means 2D drawn animation: Western cartoons come out
`anime`, and stop-motion comes out `live_action`. If anime_classification is all but sure an image is not a
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

A whole video per Task, from its frames: the same `classifier` and `minConfidence` settings, and in its input either
`path` or `frames` (one of them; neither or both fails the Task). `path` is one of

- a **frame sheet** made by the video-thumbnail Plugin's `VideoThumbnail` (`fyflow-ext.video-thumbnail`): a JPEG
  whose comment holds where each frame was drawn and its timestamp. Each frame is cut out at its place and
  classified on its own; a cell with no frame is left out, and a sheet with none fails with
  `The frame sheet has no frames:`. A sheet with timestamps drawn on it is still classified, with a `warnings`
  entry: the models see the digits.
- a **folder**: every image directly inside it (not in its subfolders), in name order, each one frame. A frame sheet
  in a folder is a plain image. More than 10000 images fail with `The folder has more than 10000 images:` (a large
  set belongs in a Flow `map` over `ClassifyImage`), none with `The folder has no images:`.
- any other **image**: one frame.

`frames` lists the frames instead, each `{ path, timestamp? }`: an image (a frame sheet too is one image), read in the
list's order, with its timestamp in seconds when you know it. 1 to 10000 of them; an entry that is not an image, a
timestamp below 0, or a path outside the Allowed roots fails the Task, naming the frame by its index.

A **thumbnail** made by `VideoThumbnail` (a JPEG of one frame, whose comment holds its `timestamp` and `video`) gives
its frame that timestamp, alone, in a folder or in a list; a timestamp listed in `frames` wins over it.

So `VideoThumbnail → ClassifyFrames → Edges on outcome.result.tag` needs no Transform: `VideoThumbnail` hands on the
sheet's `path`.

It hands on what `ClassifyImage` does, over all the frames, and more:

- `scores`: each label's mean over the frames; `tag` the top one (or `uncertain` below `minConfidence`), and
  `confidence` its mean score;
- `agreement`: the share of frames whose own top label is the tag (the top label, when the tag is `uncertain`), from 0
  to 1;
- `frames`: each frame's `{ path?, timestamp, tag, scores }`, in the sheet's, folder's or list's order. `timestamp`
  is seconds into the video: a sheet's, the one listed, or a thumbnail's own, else null. `path` is the frame's own
  image, for a folder's frame (the folder joined with its file name) or a listed one. A Flow can find mixed content in
  it (live action with anime inserts);
- `path`: the `path` given; none for a `frames` list;
- `video`: the video's path, as the sheet has it, or the one video the frames' thumbnails name.

With [`lists: "file"`](#answers-in-a-file), `frames` goes to a file instead.

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

### `TagImage`

One image per Task (`path` in its input), through one **tagger** (the `tagger` setting; see [Taggers](#taggers)),
which scores every tag of its tag list. It hands on `{ path, video?, tagger, tags, rating, warnings? }`:

- `tags`: `[{ name, score, category }]`, each tag whose score is over its category's threshold, the highest first.
  Names are the tagger's own (`long_hair`, `hatsune_miku`, underscores kept); categories are the tagger's
  (`general`, `character` for `wd-eva02-large`; also `copyright`, `style` and `meta` for PixAI).
- `rating`: `{ label, scores }`: every rating's score (`general`, `sensitive`, `questionable`, `explicit`, whatever
  the tagger's own names for them) and the highest as the `label`; `null` for a tagger that rates nothing.
- `video`: a frame sheet's video, as the sheet has it (as `ClassifyImage` hands it on).
- `warnings`: says so when the image is a frame sheet, tagged as one image, and when the tagger ran on the CPU, where
  it is many times slower than on a GPU (and, for `pixai-fp16`, that fp16 on the CPU is slow too: no quicker than
  fp32).

Settings, all applied to the tagger's output, so Tasks with different settings still share forward passes:

- **`thresholds`**: an object of category: threshold (0 to 1), **merged over the tagger's defaults**: a category
  it sets changes only that one, so `{ "character": 0.5 }` keeps `general` at its default. A category the tagger
  lacks fails the Task, naming the ones it has. Defaults for `wd-eva02-large`: general 0.35, character 0.85 (its
  authors' Space's); for `pixai` and `pixai-fp16`: general 0.17, character 0.27, copyright 0.24, style 0.15, meta
  0.17 (its authors').
- **`maxTags`**: the most tags handed on, the highest first, a whole number of 1 or more (anything else fails the
  Task); left empty, every tag over its threshold.

Like `ClassifyImage`, it is made for a Flow's `map`: one process per Profile runs up to 16 Tasks at once, and their
images go through the tagger in shared batches of up to [`tagBatchSize`](#tagbatchsize) (4 unless the Profile says
otherwise), a batch waiting at most 20 ms for more. A frame sheet is tagged as one image, with a warning: `TagFrames` tags
its frames one by one.

### `TagFrames`

A whole video per Task, from its frames: `TagImage`'s settings (`tagger`, `thresholds`, `maxTags`), and `path` or
`frames` in its input read as [`ClassifyFrames`](#classifyframes) reads them: a frame sheet (each frame cut out, with
its timestamp), a folder (its images in name order, at most 10000), one image, or a list of images with their
timestamps; a thumbnail gives its own timestamp. It hands on `{ path?, video?, tagger, frames, warnings? }`:

- `frames`: each frame's `{ path?, timestamp, tags, rating }`, in the sheet's, folder's or list's order, `path` and
  `timestamp` as `ClassifyFrames` gives them, `tags` and `rating` as
  `TagImage` gives them for an image, with the same settings applied to every frame;
- `video` and `warnings` as `ClassifyFrames` gives them (timestamps drawn on a sheet, a folder's image outside the
  Allowed roots), and the warning of a tagger run on the CPU.

With [`lists: "file"`](#answers-in-a-file), `frames` goes to a file instead.

It hands on each frame's own answer only: summing the frames up (the tags most frames have, say) is left to the Flow.
Frames go through the tagger [`tagBatchSize`](#tagbatchsize) at a time; the Task's progress ("Tagged 8 of 16
frames") moves on, and a Cancel is looked at, after each batch. One process per Profile runs two Tasks at once, as
`ClassifyFrames`' does. It has a pool of its own, apart from `TagImage`'s (fyflow gives each Node Type one), holding
the same `gpu` unit: so the two take turns on the GPU, and a GPU never holds two copies of the tagger.

### Answers in a file

A Task over thousands of frames has a large answer: 10000 frames tagged can be tens of MB, more than a Task may hand
on (16 MiB), and too much to keep in a Flow. `ClassifyFrames` and `TagFrames` can write it to a file instead, as
VideoThumbnail's `ExtractFrames` does with its lists. Three Node settings:

| Setting | What it is |
| --- | --- |
| `lists` | `result` (the default): `frames` is in the Task's result. `file`: the whole result, `frames` included, is written to a JSON file, and the Task hands on the rest, with `file` (its path), `count` (its frames) and `reused`. |
| `output` | With `file`: a `.json` file to write, or a folder to name one in. An absolute path in an Allowed root with `write` access. |
| `overwrite` | With `file`: run the models again even when a file made before is there. |

**Where the file goes.** First `output`. Without it, the Profile's `outputDir` (a per-Task field, which a Flow or
Node may set too; also needing `write` access). Without that, the Plugin's own `results/` folder in its data
directory, which the [Data](#data) link lists and nothing clears by itself.

**Reuse.** In a folder, the file is named `<name>.<key>.json`, after the video (or the sheet, folder, or a list's
first frame's folder). The key covers the Node Type, the input, each frame file's path, size and last change, the
settings and the models' commits. So a Task on unchanged frames with the same settings hands on the file it made
before (`reused: true`) without running a model, unless `overwrite` is set. A `.json` path given is written every
time. A file is written beside its place and renamed into it, so a file there is always whole.

A Flow reads the file with fyflow.fs's `ReadFile` (`as: "json"`), which needs a `read` root on its folder. With
`lists: "result"`, a result too large to hand on fails the Task saying to set `lists` to `file`.

## Allowed roots

The Plugin reads only the folders a Profile allows it, its Allowed roots: it takes the Host's (`roots: true`), so
until a Profile has a root with `read` access covering a path, every Task refuses it. Add one on the Profiles page:
on the Profile's own roots, which every Plugin that takes them shares, or on this Plugin's own in its section, which
add to the Profile's (and may leave the Profile's out). A Profile other than the Default one has the Default
Profile's roots unless it sets its own.

Each Task checks its `path` before reading anything, where the path really leads (links followed). A refused path
fails the Task with why: `… has no Allowed roots for it yet. Add one with read access on the Profiles page`, or
`… is outside every Allowed root with read access on this Profile`. A folder given to `ClassifyFrames` is checked,
and so is each image read from it: a link in it that leads out of every root is skipped, with a `warnings` entry
naming it (and if nothing else is left, the Task fails with why). A frame sheet's `video` is only handed on, never
read, so it is not checked. The result's `path` is the path as given.

The Plugin's data directory, where the models download and its own `results/` folder is, needs no root, nor does a
Profile's `modelDir`: a Profile sets it itself. An `output` or `outputDir` it writes the frames' answers to needs a
root with `write` access.

## Resources

Every Node Type holds what it uses per process, not per Task: while its process lives (and for 5 s after its last
Task), each pool holds a unit of the Host's `gpu` group and 4 of its `cpu` group (every one, on a machine with fewer than 4
logical processors), and a Task holds nothing of its own. Tasks waiting on a shared forward pass use no processor, and
a forward pass runs on as many threads as the `cpu` units held. On the `cpu` build a pool holds the same `cpu` units
and no `gpu` unit. With one GPU unit, the Node Types' GPU pools (`TagImage` and `TagFrames` have pools of their own too), and each
Profile's, take turns: a pool waits for the other to idle out, up to about 10 s (its 5 s idle time and the Host's 5 s more).
That suits the intended use, many Tasks of one Node Type and then many of the other, rather than Tasks alternating
between them. A Profile can change the idle time (`idleTimeout`) for each Node Type.

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
on the Resource Groups page, which a Host with more GPUs needs: unit `n` is CUDA's device `n`.

On `directml`:

- **Unit 0 is the fastest GPU Windows reports.** On a machine whose only GPU is integrated, that is the integrated
  GPU, and the Plugin runs on it.
- **To keep the Plugin off an integrated GPU,** set this Plugin's build to `cpu` on a Profile (the Default Profile for
  every Node). The `cpu` build holds no `gpu` unit either.
- **More than one `gpu` unit isn't supported.** Units above 0 are DirectML's adapter of that number, in Windows'
  order, not by speed: unit 1 may be an integrated GPU, or the same GPU as unit 0. Keep the `gpu` group at 1 unit
  on a `directml` Host.

With one `gpu` unit, `ClassifyImage`, `ClassifyFrames`, `TagImage` and `TagFrames` take turns on the GPU, as above; on the `cpu` build they
hold no `gpu` unit, so they run beside each other and beside GPU pools. Set [`batchSize`](#batchsize) for the GPU's
memory.

### `batchSize`

A Profile start-up field (read when a Worker starts, so it takes effect at the next Reload): the most frames in one
forward pass, 16 if left empty. It is for the GPU, where the best size depends on the GPU's memory. Raising it on
`ClassifyImage` helps only if that Profile's `maxConcurrentTasks` for `ClassifyImage` is at least as high, since a
batch holds one image per Task running.

### `tagBatchSize`

A Profile start-up field, apart from `batchSize`: the most images in one forward pass of a tagger, 4 if left empty:
the images of `TagImage`'s Tasks running at once, or the frames of one `TagFrames` Task.
A tagger is over 1 GB and each image of a batch needs GPU memory on top of it. A tagger that does not fit in the GPU's
memory fails the Task with `<tagger> needs more GPU memory than this GPU has:`, then what to do: a smaller tagger,
a smaller `tagBatchSize`, or the `cpu` build. It never falls back to the CPU by itself. PixAI takes a 1008×1008
image: on a 4 GB GPU, `pixai` and `pixai-fp16` need a `tagBatchSize` of 1.

## Models

Each model is pinned to a Hugging Face commit and its file's sha256, and downloaded into the Plugin's data
directory by the first Task that needs it (about 17 MB each for the classifiers, over 1 GB for a tagger, whose tag
list is pinned and downloaded beside it); nothing downloads at install, and nothing else ever loads. A mirror of Hugging Face can be named with `HF_ENDPOINT` in the Host's environment. The models are not part
of this repository and keep their own licences: see [Licence and credits](#licence-and-credits).

A download shows as the Task's progress ("Downloading the model anime_real"; a tagger's tag list first, "Downloading
the tag list of wd-eva02-large"). Tasks of one Node Type and Profile that
need a model at once share one download; a cancelled Task stops waiting at once, and the download itself stops when
every Task waiting on it is cancelled. A failed download, or a file that is not the pinned one, fails the Tasks that
waited on it with `Could not get the model <id>:` and leaves nothing behind; the next Task tries again.

The Plugin's status on the Plugins page says which models, and taggers' tag lists, are in its data directory: `ok`
once all are, else "Models download on first use" with the missing ones. It looks only in the data directory, not in a Profile's model
folder (a Plugin's status is not handed the Profile).

### Data

On a Host with `fyflow-plugin-sdk` 0.12.0 or later, the Plugin's card on the Plugins page has a **Data** link. It
lists each model version in the data directory's `models` folder with its size: a catalogue model by its id and source
(`anime_real: deepghs/anime_real_cls at 097a6c1e9e62`), and a model at a commit this version no longer pins, left by
an earlier version, as not used by this version. Any of them can be deleted there, or the whole folder. Deleting a
model only costs a download: the Host stops the Plugin's Workers first, and the next Task that needs it downloads it
again. After an update, a model marked not used by this version can be cleared there. A Profile's `modelDir` is
outside the data directory, so it is never listed or deleted. The `results/` folder, where
[answers in a file](#answers-in-a-file) go when nothing else is named, is listed too once there is one: deleting it
only means the next Task runs its models again.

### Without internet: `modelDir`

On a machine that cannot reach Hugging Face, set **`modelDir`** on the Plugin's section of a Profile to an absolute
folder holding the models, and Reload (it is read when a Worker starts). Then nothing is ever downloaded: a model
missing from the folder fails the Task with `Could not get the model <id>:`, naming the folder and the path it
expected. Each file is checked against its pinned sha256 before it loads.

The folder has the layout of the data directory's `models` folder:
`<repo>/<commit, first 12 characters>/<name>/model.onnx`, each file downloaded on another machine from
`https://huggingface.co/<repo>/resolve/<commit>/<name>/model.onnx` (a tagger's files are at the top of its repo, with
no `<name>`):

| Model | Put at (below `modelDir`) | Download from |
| --- | --- | --- |
| `anime_real` | `deepghs/anime_real_cls/097a6c1e9e62/mobilenetv3_v1.4_dist/model.onnx` | https://huggingface.co/deepghs/anime_real_cls/resolve/097a6c1e9e62866a435810c883a45a7cb2c7077d/mobilenetv3_v1.4_dist/model.onnx |
| `anime_classification` | `deepghs/anime_classification/5ee62e06f5f4/mobilenetv3_v1.5_dist/model.onnx` | https://huggingface.co/deepghs/anime_classification/resolve/5ee62e06f5f4cd68a1c2f3bc5dc9805e827d37df/mobilenetv3_v1.5_dist/model.onnx |
| `wd-eva02-large` | `SmilingWolf/wd-eva02-large-tagger-v3/c5303bb71394/model.onnx` | https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/c5303bb7139430db980e4c680a778fe79d72b541/model.onnx |
| `wd-eva02-large`'s tag list | `SmilingWolf/wd-eva02-large-tagger-v3/c5303bb71394/selected_tags.csv` | https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/c5303bb7139430db980e4c680a778fe79d72b541/selected_tags.csv |
| `pixai` | `Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8/080077856314/model.onnx` | https://huggingface.co/Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8/resolve/0800778563144a0e6fdf41ddadd84aae3cb0dbcf/model.onnx |
| `pixai-fp16` | `Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8/080077856314/model_fp16.onnx` | https://huggingface.co/Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8/resolve/0800778563144a0e6fdf41ddadd84aae3cb0dbcf/model_fp16.onnx |
| `pixai`'s and `pixai-fp16`'s tag list | `Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8/080077856314/selected_tags.csv` | https://huggingface.co/Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8/resolve/0800778563144a0e6fdf41ddadd84aae3cb0dbcf/selected_tags.csv |

A machine that has run the Plugin online already holds this folder: its data directory's `models` folder can be
copied whole.

## Taggers

What `TagImage` and `TagFrames` may run (their `tagger` setting):

| Tagger | Model | Size | Categories (default thresholds) | Rating |
| --- | --- | --- | --- | --- |
| `wd-eva02-large` (default) | [`SmilingWolf/wd-eva02-large-tagger-v3`](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) at `c5303bb7139430db980e4c680a778fe79d72b541` (tag `v1.0`), its `model.onnx` and `selected_tags.csv` | about 1.26 GB, downloaded on first use | general (0.35), character (0.85): 8,106 general tags, 2,751 characters | general, sensitive, questionable, explicit |
| `pixai` | PixAI tagger v1.0, as [`Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8`](https://huggingface.co/Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8) at `0800778563144a0e6fdf41ddadd84aae3cb0dbcf`, its `model.onnx` (fp32) and `selected_tags.csv` | about 1.98 GB, downloaded on first use | general (0.17), character (0.27), copyright (0.24), style (0.15), meta (0.17): 15,043 general tags, 8,308 characters, 2,460 copyrights, 4,917 styles, 145 meta | general, sensitive, questionable, explicit |
| `pixai-fp16` | the same, its `model_fp16.onnx` (fp16) | about 0.99 GB, downloaded on first use | as `pixai` | as `pixai` |

`wd-eva02-large` is by SmilingWolf, under the Apache-2.0 licence, trained on Danbooru's tags. The Plugin prepares an
image as the authors' [Space](https://huggingface.co/spaces/SmilingWolf/wd-tagger) (`app.py`) does: what is
see-through made white, padded to a square on white, resized bicubic to 448×448, BGR, 0 to 255; its golden tests hold
its scores to a copy of those steps (`pnpm golden:tags`).

PixAI tagger v1.0 is by [PixAI](https://huggingface.co/pixai-labs), published as
[`pixai-labs/pixai-tagger-v1.0`](https://huggingface.co/pixai-labs/pixai-tagger-v1.0) (pinned at
`9fe10addf9326e292da8a85a98ea74cd91b41771`) under the Apache-2.0 licence, which its model card's metadata declares
(the repo has no LICENSE file). Its authors ship PyTorch weights only, so the Plugin runs the community's ONNX
conversion by Mexes, offered under the same licence (declared the same way): `pixai` is its fp32 file, `pixai-fp16`
its fp16 one. The Plugin prepares an image as the authors' `RescalePadProcessor` does (bilinear, not bicubic as the
conversion's README says): what is see-through made white, the long side resized to 1008 (in float, not in whole
levels), centred on black, RGB, `(x - 0.5) / 0.5`. Checked against the authors' own pipeline on the test images,
`pixai` is within about 0.002 of its scores and `pixai-fp16` within about 0.005, so both keep the authors'
thresholds; the golden tests hold both to it. Their reference scores (`python/tests/fixtures/golden-pixai.json`) were
made by the research script `.scratch/image-tags/research/pixai-reference.py`, which ran the authors' PyTorch pipeline
(see `.scratch/image-tags/research/pixai-onnx-check.md`); `pnpm golden:pixai` is a port of it that has not been run
yet. One difference is kept on purpose: the authors'
pipeline ignores an RGB image's colour key, which the Plugin makes white, as for every model. `pixai-fp16` is half
the size, for a GPU; on the CPU it is no quicker than `pixai`. The authors' rating threshold (0.41) is not used: the
rating's `label` is always the highest, as for `wd-eva02-large`, and the ratings (`rating:g` and so on in the tag
list) are named as `wd-eva02-large`'s, so a Flow that routes on the rating works with either tagger.

**Thresholds merge** over the chosen tagger's defaults, category by category: `{ "character": 0.5 }` lowers the
character threshold and keeps the general one at 0.35. A tag is handed on when its score is over (not at) its
category's threshold. A category the tagger lacks fails the Task, naming the tagger's categories.

## Development

Needs Node 26, pnpm and [uv](https://docs.astral.sh/uv/).

- `pnpm check`: typecheck, lint, the declarations test (vitest) and the Python tests (pytest, through uv). The first
  run fetches the classifiers' pinned models into `python/.cache/`. The taggers are never downloaded by the tests:
  the tagging tests run a small stand-in tagger, and a tagger's golden tests (marked `slow`) run only when its files
  are in `python/.cache/models/`, laid out as a [`modelDir`](#without-internet-modeldir) (skipped otherwise; `-m "not
  slow"` leaves them out).
- `pnpm plugin-folder [dir]`: the folder to install from the app's Plugins page ("Choose folder…"), by default
  `dist/image-tags/`.
- `pnpm sheets`: remakes the frame-sheet fixtures (`python/tests/fixtures/sheets/`) with the video-thumbnail Plugin's
  `thumbs-collage` service and test clip, from `../fyflow-plugin-video-thumbnails` (or `-- --thumbnails <folder>`).
- `pnpm golden`: remakes the synthetic test images (the small and see-through ones) and writes imgutils' scores for
  them and the reference frames (`python/tests/fixtures/golden.json`), in an environment of its own; run it by hand
  when a catalogue model changes.
- `pnpm golden:tags [-- --from python/.cache/models]`: writes each tagger's top tags and ratings for the test images,
  prepared by its reference pipeline, not the Plugin's code (`python/tests/fixtures/golden-tags.json`); run it by hand
  when a catalogue tagger changes. Without `--from`, it downloads each tagger at its pinned commit.
- `pnpm golden:pixai [-- --from <folder>]`: writes PixAI's top tags and ratings for the test images from its
  authors' own pipeline (`python/tests/fixtures/golden-pixai.json`), in an environment of its own with PyTorch for
  the CPU; it downloads about 2 GB of the authors' weights at their pinned commit (or takes them from `--from`) and
  should take some 9 minutes. Run it by hand when PixAI's pins change. It is a port of
  `.scratch/image-tags/research/pixai-reference.py`, which made the file there now, and has not been run itself.
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

## Licence and credits

The Plugin's code is under the [MIT licence](LICENSE). The models it downloads are not part of it, and each keeps its
own licence, which applies to whoever runs them:

- [`deepghs/anime_real_cls`](https://huggingface.co/deepghs/anime_real_cls): OpenRAIL. Its use restrictions bind
  anyone using the model, through this Plugin too.
- [`deepghs/anime_classification`](https://huggingface.co/deepghs/anime_classification): MIT.
- [`SmilingWolf/wd-eva02-large-tagger-v3`](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3): Apache-2.0,
  by SmilingWolf. The Plugin prepares images as its authors' Space does (see [Taggers](#taggers)).
- [`pixai-labs/pixai-tagger-v1.0`](https://huggingface.co/pixai-labs/pixai-tagger-v1.0): Apache-2.0 (declared in its
  model card's metadata), by PixAI; run as
  [`Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8`](https://huggingface.co/Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8),
  Mexes's ONNX conversion of it, under the same licence (declared the same way). The Plugin prepares images as
  PixAI's own pipeline does (see [Taggers](#taggers)).

The classifiers are by [DeepGHS](https://huggingface.co/deepghs). The Plugin prepares images as their
[imgutils](https://github.com/deepghs/imgutils) does, and its tests hold its scores to imgutils' (`pnpm golden`).

The test images keep their own licences, listed in `python/tests/fixtures/images/CREDITS.md` (see
[Test images](#test-images)).
