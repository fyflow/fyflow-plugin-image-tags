# 16: PixAI in the catalogue: `pixai` and `pixai-fp16`

**What to build:** two tagger entries for PixAI v1.0, from the ONNX that ticket 15 settled on.

**Blocked by:** 13, 15

**Status:** ready-for-agent

- [ ] **Entries** `pixai` (fp32, about 1.98 GB) and `pixai-fp16` (about 0.99 GB). Each has:
  - its own file and sha256;
  - its own default thresholds: general 0.17, character 0.27, copyright 0.24, style 0.15, meta 0.17 and rating 0.41,
    or fp16's own from ticket 15;
  - the pinned tag map;
  - whether its output is sigmoided.
- [ ] **Preprocessing:** 1008, long side resized (by the method ticket 15 settled), padded black, RGB, NCHW, mean/std
  0.5.
- [ ] **CPU warning** names fp16 on the CPU as slow.
- [ ] **Tests:** golden results from ticket 15, marked slow, skipped without the model.
- [ ] **README:** PixAI's authors, repo, commit, Apache-2.0 licence, the conversion and its licence, and the sizes.
- [ ] **Verified by hand on a Host:**
  - TagImage and TagFrames run with `pixai` and with `pixai-fp16`;
  - on the 4 GB GPU, the out-of-memory message shows if the model doesn't fit.
