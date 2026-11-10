# fyflow-plugin-image-tags

A fyflow-app Plugin (`fyflow-ext.image-tags`) that tells what an image is with small local ONNX classifiers: anime,
CG or live action today, and for whole videos next (see `.scratch/image-tags/spec.md`).

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

It hands on `{ path, classifier, labels, tag, confidence, scores, models }`: `scores` holds each label's score,
`confidence` the top one's, and `models` the catalogue models the classifier ran (both, for `style`). `tag` is the top
label when its score reaches **`minConfidence`** (a setting from 0 to 1; left empty, the classifier's own default
above), else **`uncertain`**: a result, never a failure. So a Flow routes on `outcome.result.tag` with plain Edge
conditions, `uncertain` included. A raw model's default is 0: it answers `uncertain` only when a Node asks for a
`minConfidence`.

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
- `pnpm golden`: remakes the golden images and imgutils' scores for them (`python/tests/fixtures/`), in an
  environment of its own; run it by hand when a catalogue model changes.
