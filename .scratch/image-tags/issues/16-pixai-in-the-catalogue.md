# 16: PixAI in the catalogue: `pixai` and `pixai-fp16`

**What to build:** two tagger entries for PixAI v1.0, from the ONNX that ticket 15 settled on.

**Blocked by:** 13, 15

**Status:** resolved

- [x] **Entries** `pixai` (fp32, about 1.98 GB) and `pixai-fp16` (about 0.99 GB). Each has:
  - its own file and sha256;
  - its own default thresholds: general 0.17, character 0.27, copyright 0.24, style 0.15, meta 0.17 and rating 0.41,
    or fp16's own from ticket 15;
  - the pinned tag map;
  - whether its output is sigmoided.
- [x] **Preprocessing:** 1008, long side resized (by the method ticket 15 settled), padded black, RGB, NCHW, mean/std
  0.5.
- [x] **CPU warning** names fp16 on the CPU as slow.
- [x] **Tests:** golden results from ticket 15, marked slow, skipped without the model.
- [x] **README:** PixAI's authors, repo, commit, Apache-2.0 licence, the conversion and its licence, and the sizes.
- [x] **Verified by hand on a Host:**
  - TagImage and TagFrames run with `pixai` and with `pixai-fp16`;
  - on the 4 GB GPU, the out-of-memory message shows if the model doesn't fit.

## Changed while building it (2026-11-17)

- **Rating:**
  - The rating has no threshold: the label is the highest score, as for WD, and a `rating` key in `thresholds` fails the Task. PixAI's 0.41 is unused.
  - PixAI's `rating:g/s/q/e` are renamed to WD's `general/sensitive/questionable/explicit`, in WD's order, so a Flow routes on the rating the same way with either tagger.
- **Preprocessing:** a new `long_side_first` option. The resize works on 0-255 values rather than 0-1; bilinear is linear, so it stays within 1e-5 of the check's float resize, which a test holds.
- **Catalogue:**
  - `Tagger` gained `output` (the graph's output name), `precision` and `rating_names`.
  - Both PixAI files share one commit folder, so the Data link lists them as one item: "pixai, pixai-fp16: Mexes/… at 080077856314".
- **CPU warning for `pixai-fp16`:** fp16 on the CPU was no quicker than fp32 here (about 20 s against 18 s per image), so the warning says so and points to fp32.
- **Golden tests:**
  - 3 images per model on the CPU: the tall yakitori frame, `cg-02-sintel.jpg` and `transparent.png`; `keyed.png` is left out.
  - Tolerances are fp32 0.004 and fp16 0.008. Measured: fp32 within 1e-5, fp16 within 0.0019.
  - They take about 130 s.
- **`python/scripts/golden_pixai.py`** (`pnpm golden:pixai`) ports the reference script with CPU PyTorch in a uv inline-script environment. It hasn't been run.
- **Tests:** 326 pytest (slow included) and 73 vitest pass.
- **Verified by hand on a fresh Host** (`cuda` build, RTX 3050 Ti 4 GB, the models pre-seeded in the data folder):
  - TagImage with `pixai-fp16` loaded on CUDA and tagged in about 5 s; the result included a `copyright` tag (`original`).
  - The Data link listed WD (1.26 GB) and "pixai, pixai-fp16" (2.98 GB).
  - **No out-of-memory failure at batch 4:** TagFrames with `pixai` (fp32) on a folder of 4 images completed in about 74 s. The Windows driver most likely spilled into shared system memory, so a 4 GB card is slow rather than refused. The clear out-of-memory message stays tested with a faked failure only.
  - With `tagBatchSize` 1 set in the Profile, TagFrames with `pixai-fp16` went one frame at a time and completed, at about 25 s per frame with the GPU full (3.96 GB).
