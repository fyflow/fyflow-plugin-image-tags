# 23: The TagScenes Node Type

**What to build:** `TagScenes`: reads frames as TagFrames does, tags and embeds them in batches, groups and pools them
with `scenes.py`, and hands on scenes and a summary, or writes them to a file. See `TagScenes-spec.md` §3, §4 and §6.

**Blocked by:** 21, 22

**Status:** ready-for-human

- [x] `frames.py`:
  - blank detection (`BLANK_STD`, `ImageStat` on a 64-px thumbnail) at batch load time
  - `duration` from thumbnail comments onto `Frames` when the frames agree on one
  - ClassifyFrames' and TagFrames' output unchanged
- [x] `results.py`: `summary` / `reused` take the list fields to move. `count` comes from the caller. ClassifyFrames
  and TagFrames behave as before.
- [x] `worker.py`: `class TagScenes(_Tag)`.
  - The input is `path` or `frames`.
  - Every frame needs a timestamp; the Task fails before any model if not, and the message says how to get them.
  - Tag and embed via `scores_and_embeddings`, keeping only §4.4. Progress 0 → 0.9, then grouping to 1.0. Cancel is
    checked between batches and stages.
  - The result follows §3.4. `includeFrames` adds `frames` with `blank` and `scene`.
- [x] `plugin.ts`:
  - the worker and Node Type (pool as TagFrames)
  - settings §3.3, with `fileSettingsProperties`
  - output schema with `examples`, reusing `TAGGED_FRAME`; nothing required that `lists: "file"` removes
- [x] README:
  - the Node Type and the Flows of §2
  - the settings
  - "sample about every 2 s, and give DetectScenes a `maxFrames` above the frame count"
- [x] **Tests:**
  - pytest per §7.2: the worker on the stand-in, and the `slow` planted video from the fixture images
  - vitest for the Node Type, settings and schemas
  - `pnpm check` passes
- [ ] On a Host: §7.4.

## Changed while building it (2026-11-19)

- `frames.py`: `is_blank(image)` is public. TagScenes calls it on each frame as that frame's batch loads; ClassifyFrames
  and TagFrames never call it. `Frame` and `Frames` gain `duration`: a thumbnail's `duration` (above 0), taken when
  every frame that gives one gives the same. `thumbnail_of` keeps its two-value return; a private `_thumbnail` reads all
  three.
- `results.py`: the signatures are `summary(result, path, was_reused, count, lists=("frames",))` and
  `reused(path, lists=("frames",))`. A reused file needs the first list field; any other one must be a list where it
  appears, since `frames` is absent without `includeFrames`. `_Worker._hand_on` takes `lists` and counts the first
  field. A reused TagScenes file reports "Reused the answers of N scenes".
- `worker.py`: `_frame_by_frame` now wraps `_batch_by_batch(..., share)`, which returns whatever each batch's function
  returns and takes the progress from 0 to `share`.
- The worker keeps each frame's L2-normalised embedding, a sparse copy of its scores with the rating columns zeroed
  (the ratings are kept whole in their own array), its blank flag and its timestamp.
- Rating labels come from the tag list's order (WD) or from `rating_names` (PixAI), least to most severe, as in `tag()`.
- The warning check asks about the embedding session (`runs_on_cpu(..., embedding=True)`), so no plain session is ever
  made in TagScenes' process.
- A Cancel between stages fails with "Cancelled after tagging N frames" or "Cancelled after grouping N frames". The
  progress messages are "Grouping" at 0.9, "Pooling" at 0.95, then "Found S scenes in N frames" at 1.0.
- The missing-timestamp message reads: "K of N frames have no timestamp, and TagScenes groups frames by their time:
  give it frames that carry one (a frame sheet or thumbnails from VideoThumbnail, the frames of DetectScenes or
  ExtractFrames), or a `frames` list with each frame's `timestamp`".
- **The result's `duration`** is the last scene's `end`. That is the thumbnails' duration when they give one, otherwise
  the last frame's time plus half the median gap.
- **No per-frame `video`** with `includeFrames`. §3.4 asks for one, but TagFrames' per-frame shape (and
  `TAGGED_FRAME`) has none, so I followed TagFrames.
- **`ignoreTags` also filters each frame's own tags** with `includeFrames`. The settings table says "everywhere".
- **The reuse key covers the `scenes.py` constants and `BLANK_STD`** besides the effective settings, so a file made
  under other constants (for example by the review script) is not reused.
- `plugin.ts`: `TAGGED_FRAME` did not exist before, so I extracted it from TagFrames' output schema; both now use it.
  `SCENE_MAX_TAGS` and `SCENE_DEFAULTS` are exported, and a vitest holds them to the Worker's `SCENE_MAX_TAGS` and
  `scenes.Settings()`.
- `plugin.ts` wording: the Profile's `outputDir` and `tagBatchSize` descriptions, and the Data item's label, now name
  TagScenes too.
- `plugin.ts` icon: TagScenes uses `tag` as the spec says, while TagFrames' icon is `image`.
- The output schema requires `tagger`, `frameCount`, `duration`, `threshold` (`number|null`), `summary` and `count`.
- README: a TagScenes section (the Flows of §2, the advice to sample about every 2 s and set DetectScenes' `maxFrames`,
  the settings table, the output). Answers in a file, Resources and `tagBatchSize` now name TagScenes, and the intro line
  mentions scenes.
- Tests: `tests/test_tag_scenes.py` (31, on the stand-in, including `is_blank` and the duration checks) and
  `tests/test_tag_scenes_planted.py` (slow, real `wd-eva02-large`, about 4.5 min on the CPU).
- The planted video is three scenes of 36, 36 and 32 s at 2 s. The middle one cuts between the left and right halves of
  one image, two frames to a shot. Two black frames lie between scenes 1 and 2, and two in the middle of scene 3.
- The planted test passed untuned. Seeds 1, 2, 3 and 23 gave the exact planted boundaries (39.0 s and 75.0 s), with
  every blank where it should be. Links across the cuts were about −1.2 z against a threshold of about 0.95. No
  `scenes.py` constant was touched.
