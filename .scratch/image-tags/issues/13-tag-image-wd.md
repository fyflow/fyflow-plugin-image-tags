# 13: TagImage with WD EVA02-large, end to end

**What to build:** a `TagImage` Node Type that tags one image with `wd-eva02-large` (SmilingWolf's
`wd-eva02-large-tagger-v3`), as decided in the spec's "Taggers" section. It lays the ground every later tagger uses:
the `kind: "tagger"` catalogue entry, per-model preprocessing, thresholds merged over defaults, and `tagBatchSize`.

**Blocked by:** none

**Status:** resolved

- [x] **Catalogue entry `wd-eva02-large`** (`kind: "tagger"`):
  - repo `SmilingWolf/wd-eva02-large-tagger-v3` at `c5303bb7139430db980e4c680a778fe79d72b541`;
  - `model.onnx`: 1,260,435,999 B, sha256 `9e768793060c7939b277ccb382783e8670e8a042d29d77aa736be0c8cc898bfc`;
  - `selected_tags.csv`, pinned by sha256 too;
  - categories from the csv's `category` column (0 general, 4 character, 9 rating), with defaults general 0.35 and
    character 0.85;
  - the output is already sigmoided.

  Fetched by the same hardened fetching as the classifiers; listed by `data()` and `status()`; `cache` downloads it.
- [x] **Per-model preprocessing** in the catalogue entry: size 448, padded to a square on white, bicubic, BGR, NHWC,
  0-255 float32. The classifiers' existing path moves into the same description with no change to their results:
  their golden tests still pass.
- [x] **Settings:**
  - `tagger`: an enum, default `wd-eva02-large`.
  - `thresholds`: an open object (`additionalProperties`: a number from 0 to 1), merged over the model's defaults.
    A key the model lacks fails the Task, naming the model's categories. The description lists each model's defaults.
  - `maxTags`: an optional integer, at least 1.
- [x] **Result:**
  - `tags: [{ name, score, category }]`: the tags over their threshold, highest first, names unchanged;
  - `rating: { label, scores }`: all four labels' scores, with the highest as the label; `null` for a model without
    ratings.

  Output schema and an example in `plugin.ts`.
- [x] **Workers and pools:** taggers have their own Worker and pool, taking turns on the `gpu` unit with the
  classifiers. Tasks share batches through `batcher.py`; thresholds and `maxTags` apply after the forward pass.
- [x] **`tagBatchSize`:** a Profile start-up field, default 4, separate from `batchSize`.
- [x] **CPU warning** when the tagger runs on the CPU, in the result's warnings as the classifiers do it.
- [x] **Out of GPU memory:** an allocation failure fails the Task with "<model> needs more GPU memory than this GPU
  has: use a smaller tagger, a smaller `tagBatchSize`, or the `cpu` build". No fallback to the CPU.
- [x] **Progress** as in ticket 11: the download up to 1.0, then "Loading the model wd-eva02-large on <device>", then
  "Tagging".
- [x] **Teardown** as in ticket 12: sessions are dropped.
- [x] **Tests:**
  - **Unit:**
    - threshold merging: one key overrides only that category, and an unknown key fails;
    - `maxTags` and sorting;
    - the rating;
    - preprocessing on a small synthetic image: padding colour, channel order and layout.
  - **Golden:** compare against the fixture images' top tags and scores, computed once with the Space's `app.py`
    preprocessing, within a tolerance. Marked slow; skipped without the model on disk.
  - **vitest:** the settings and output schemas, and the Profile field.
- [x] **README:** a "Taggers" section with the authors, repo, pinned commit, Apache-2.0 licence and size (about
  1.26 GB, downloaded on first use), and how thresholds merge.
- [x] **Verified by hand on a Host:**
  - a TagImage Run on a fixture image completes on CUDA;
  - two Tasks with different thresholds share a batch;
  - the Data link lists the model;
  - nvidia-smi shows its memory freed after the Worker ends.

## Changed while building it (2026-11-17)

- **`thresholds` lists its keys:**
  - fyflow-orchestration (0.14.0, also 0.15.0) admits only `true` or `false` for `additionalProperties`, never a schema (FD-V33), so the decided open object with a number schema failed every Node's validation.
  - `thresholds` is now `additionalProperties: true`, with each category any tagger has as its own property (a number from 0 to 1), which the form shows as named fields.
  - A key the chosen tagger lacks still reaches the Worker, which fails the Task naming the tagger's categories.
- **The tag list downloads first,** with the same hardened fetching (sha256 `298633d9…41217`) and its own progress message.
- **A tag is kept only when its score is above the threshold,** not equal to it, as in `app.py`.
- **The result also has `path` and `tagger`.**
- **A frame sheet is tagged as one image,** without ClassifyImage's sheet warning.
- **Pool:** TagImage's pool runs up to 16 Tasks at once (`TAG_TASKS_AT_ONCE`).
- **Shared code:** the batching machinery is shared with ClassifyImage.
- **The out-of-GPU-memory message** also covers the classifiers, worded for them ("a smaller `batchSize`, or the `cpu` build"). Real CUDA out-of-memory errors weren't reproduced: detection matches ONNX Runtime's known error texts and is tested with a faked GPU session.
- **Tests:**
  - A tiny stand-in tagger (`tests/stand_in.py`, needs `onnx` in the test dependency group) keeps the unit tests from downloading 1.26 GB.
  - The golden tests (`golden-tags.json`, made by `pnpm golden:tags` from a copy of `app.py`'s steps) look for the model in the gitignored `python/.cache/models/`. They differ from the reference by 0.
  - 259 pytest tests pass, 22 of them slow golden ones; 66 vitest tests pass.
- **Verified by hand on a fresh Host** (fyflow-app, `cuda` build, RTX 3050 Ti 4 GB) with the packed `.tgz`:
  - **First Run:** the model downloaded in about 18 s, then the Task reported "Loading the model wd-eva02-large on CUDA", then "Tagging", and completed (yakitori stall: food, food_stand, night, paper_lantern…; rating general 0.99).
  - **Three Flows at once** with different thresholds:
    - `{general: 0.95}` gave only tags at 0.95 and above;
    - `{general: 0.2, character: 0.3}` gave the longer lists;
    - `{style: 0.5}` failed with "has no category `style`: its categories are general, character".
  - Batch sharing across Tasks is covered by the tests, not seen from outside.
  - **Data link:** it listed `wd-eva02-large` at 1,260,744,467 B.
  - **GPU memory:** it was 2.1 GB while the Worker lived and 0 once it idled out, with no Worker processes left.
