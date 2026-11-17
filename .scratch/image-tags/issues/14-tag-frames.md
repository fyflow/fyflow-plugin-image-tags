# 14: TagFrames

**What to build:** a `TagFrames` Node Type that tags each frame of a frame sheet, a folder or one image, with the
taggers of ticket 13. It gives per-frame tags only; summaries come later from `SummarizeTags`.

**Blocked by:** 13

**Status:** resolved

- [x] **Input:** read with the same reader as `ClassifyFrames` (`frames.read_frames`), with Allowed roots checked the
  same way:
  - a frame sheet, with timestamps;
  - a folder, sorted by name, without timestamps, up to 1000 images;
  - one image.
- [x] **Settings:** as for `TagImage`: `tagger`, `thresholds`, `maxTags`.
- [x] **Result:** `frames: [{ timestamp, tags, rating }]` in frame order, plus `video` and `warnings` as
  `ClassifyFrames` gives them, including the timestamps-drawn warning.
- [x] **Batches** of `tagBatchSize` frames; progress reads "Tagging n of N frames".
- [x] **Tests:**
  - a sheet, a folder and one image each give the expected frames and timestamps;
  - the settings apply per frame;
  - progress comes in the right order.
- [x] **Verified by hand on a Host:** a TagFrames Run on a real 64-frame sheet completes, and the time per sheet is
  noted.

## Changed while building it (2026-11-17)

- **Its own Worker and pool:** the SDK gives each Node Type its own pool per Profile, so TagFrames can't share TagImage's process.
  - The pool holds the same `gpu` unit as TagImage's, so the two take turns and never hold two copies of the tagger at once.
  - Up to 2 Tasks at once with `maxThreads: 1`, as ClassifyFrames has.
- **Shared code:**
  - `TagImage` and `TagFrames` share a `_Tag` base: tagger lookup, threshold merging, `maxTags`, `tagBatchSize`, getting ready, and the CPU warning.
  - `ClassifyFrames` and `TagFrames` share the frame-by-frame loop (batches, Cancel checks, progress) and the result's `path`/`video`/`warnings`.
- **Progress:** "Tagging 0 of N frames", then "Tagged n of N frames"; a Cancel says "Cancelled after n of N frames".
- **Wording:** the over-1000-images message now points to "a map over ClassifyImage or TagImage".
- **Tests:**
  - `test_tag_frames.py`: 21 tests with the stand-in tagger.
  - A slow golden test holds each frame of the sample sheet to TagImage's result on the hand-cut cell, within 0.01. It takes about 90 s.
  - Allowed-roots tests cover both tagging Node Types.
  - 296 pytest and 71 vitest tests pass.
- **Verified by hand on a fresh Host** (`cuda` build, RTX 3050 Ti 4 GB, model pre-seeded in the data folder) on a 64-frame sheet made from the fixtures (2560×1440, 320×180 cells, timestamps 0-630 s):
  - completed in about 21 s from the Run's start, about 10 s of it tagging (about 6 frames/s at `tagBatchSize` 4);
  - progress went by 4 frames;
  - each frame's timestamp and tags matched its source image; repeated images got the same tags.
