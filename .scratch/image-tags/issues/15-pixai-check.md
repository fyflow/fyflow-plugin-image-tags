# 15: Check the PixAI ONNX conversion against the authors' pipeline

**What to build:** a research note, with no Plugin code.
- `pixai-labs/pixai-tagger-v1.0` (at `9fe10addf9326e292da8a85a98ea74cd91b41771`) ships only safetensors.
- Check that the community conversion `Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8` (at
  `0800778563144a0e6fdf41ddadd84aae3cb0dbcf`) gives the authors' scores, so that ticket 16 can pin it.

**Blocked by:** none; it can run alongside 13.

**Status:** resolved

- [x] **Reference scores:** in a throwaway environment outside the Plugin, with PyTorch, run the authors'
  `tagger_pipeline.py` on the fixture images and keep its scores.
- [x] **Compare** the fp32 and fp16 ONNX files against those scores, using our preprocessing in ONNX Runtime:
  - the maximum and mean absolute difference;
  - which tags flip at the default thresholds;
  - the resize method: upstream uses bilinear, while Mexes's README says bicubic.
- [x] **fp16:** whether it needs its own default thresholds.
- [x] **Runs on this machine** (RTX 3050 Ti 4 GB and its CPU): whether fp32 and fp16 run on CUDA at
  `tagBatchSize` 1, or fail for want of GPU memory, and that the CPU runs them. A functional check only: no
  performance figures are needed (decided 2026-11-17), and no run on another machine.
- [x] **Licence** of the conversion repo, and the sha256 of each file.
- [x] **Verdict:** pin Mexes, or make our own export, published in a repo with its script.
- [x] **Where it goes:** the note goes in `.scratch/image-tags/research/`, and the stored scores become ticket 16's
  golden results.

## Changed while building it (2026-11-17)

The full note is `.scratch/image-tags/research/pixai-onnx-check.md`. The scripts that made it are kept beside it: `pixai-reference.py` (the upstream PyTorch reference) and `pixai-compare.py`.

- **Verdict: pin Mexes's conversion.**
  - Fed upstream's own input: fp32 is within 2e-5 on the CPU and 1.1e-3 on CUDA; fp16 is within 5.8e-3.
  - The only flips are tags within about 0.001 of their threshold, so fp16 needs no thresholds of its own.
- **Resize:** what matches upstream is to resize the long side first (size truncated with `int`), bilinear, in float (each channel as a PIL "F" image), then pad black, centred. End to end: fp32 within 0.0023 and fp16 within 0.0049, 1 borderline flip each. The other orders flip far more tags:
  - uint8 bilinear: 10 flips;
  - bicubic, as Mexes's README says: 78;
  - the Plugin's current order (pad, then resize): 125.

  Ticket 16 needs this resize as a new preprocessing option.
- **`keyed.png`:** upstream ignores an RGB image's colour key, where the Plugin makes it white. Keep the Plugin's behaviour and leave this image out of PixAI's golden test.
- **Licence:** both repos say Apache-2.0 only in their README metadata. Neither has a LICENSE or NOTICE file.
- **This machine:**
  - fp32 and fp16 both run on CUDA at batch 1, peaking near 3.95 GB of the 4 GB card. fp32 there is no faster than the CPU.
  - On the CPU, fp32 takes about 19 s per image and agrees with CUDA's result to 4.7e-4.
  - On a 4 GB card, PixAI needs `tagBatchSize` 1; larger batches should hit the clear out-of-memory message.
- **Reference scores:** `python/tests/fixtures/golden-pixai.json` holds all 21 fixtures, each with its top 20 tags and the four rating scores. It is the reference for both `pixai` and `pixai-fp16`.
- **Not done:** no `golden_pixai.py` in the repo; ticket 16 ports `pixai-reference.py`. Batch sizes above 1 weren't tried.
