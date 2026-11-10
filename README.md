# fyflow-plugin-image-tags

A fyflow-app Plugin (`fyflow-ext.image-tags`) that tells what an image is with small local ONNX classifiers: anime
or real today, and anime / CG / live action for whole videos next (see `.scratch/image-tags/spec.md`).

Needs a Host with `fyflow-plugin-sdk` 0.9.0 or later and Python Workers (uv). The Host makes the Plugin's Python
environment from `python/uv.lock` on install; nothing else is set up by hand.

## Node Types

### `ClassifyImage`

One image per Task (`path` in its input: anything Pillow opens), through one classifier (the `classifier` setting):

| Classifier | Labels | Model |
| --- | --- | --- |
| `anime_real` (default) | anime, real | `deepghs/anime_real_cls` / `mobilenetv3_v1.4_dist` |

It hands on `{ path, classifier, labels, tag, confidence, scores, models }`: `tag` is the label with the highest
score, so a Flow routes on `outcome.result.tag` with plain Edge conditions.

## Models

Each model is pinned to a Hugging Face commit and its file's sha256, and downloaded into the Plugin's data
directory by the first Task that needs it (about 17 MB each); nothing downloads at install, and nothing else ever
loads. A mirror of Hugging Face can be named with `HF_ENDPOINT` in the Host's environment. The deepghs models are
OpenRAIL-licensed.

## Development

Needs Node 26, pnpm and [uv](https://docs.astral.sh/uv/).

- `pnpm check`: typecheck, lint, the declarations test (vitest) and the Python tests (pytest, through uv). The first
  run fetches the pinned models into `python/.cache/`.
- `pnpm plugin-folder [dir]`: the folder to install from the app's Plugins page ("Choose folder…"), by default
  `dist/image-tags/`.
- `pnpm golden`: remakes the golden images and imgutils' scores for them (`python/tests/fixtures/`), in an
  environment of its own; run it by hand when a catalogue model changes.
