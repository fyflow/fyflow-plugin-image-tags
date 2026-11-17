# 14: TagFrames

**What to build:** a `TagFrames` Node Type that tags each frame of a frame sheet, a folder or one image, with the
taggers of ticket 13. It gives per-frame tags only; summaries come later from `SummarizeTags`.

**Blocked by:** 13

**Status:** ready-for-agent

- [ ] **Input:** read with the same reader as `ClassifyFrames` (`frames.read_frames`), with Allowed roots checked the
  same way:
  - a frame sheet, with timestamps;
  - a folder, sorted by name, without timestamps, up to 1000 images;
  - one image.
- [ ] **Settings:** as for `TagImage`: `tagger`, `thresholds`, `maxTags`.
- [ ] **Result:** `frames: [{ timestamp, tags, rating }]` in frame order, plus `video` and `warnings` as
  `ClassifyFrames` gives them, including the timestamps-drawn warning.
- [ ] **Batches** of `tagBatchSize` frames; progress reads "Tagging n of N frames".
- [ ] **Tests:**
  - a sheet, a folder and one image each give the expected frames and timestamps;
  - the settings apply per frame;
  - progress comes in the right order.
- [ ] **Verified by hand on a Host:** a TagFrames Run on a real 64-frame sheet completes, and the time per sheet is
  noted.
