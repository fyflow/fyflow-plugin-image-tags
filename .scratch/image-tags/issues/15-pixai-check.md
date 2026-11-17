# 15: Check the PixAI ONNX conversion against the authors' pipeline

**What to build:** a research note, with no Plugin code.
- `pixai-labs/pixai-tagger-v1.0` (at `9fe10addf9326e292da8a85a98ea74cd91b41771`) ships only safetensors.
- Check that the community conversion `Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8` (at
  `0800778563144a0e6fdf41ddadd84aae3cb0dbcf`) gives the authors' scores, so that ticket 16 can pin it.

**Blocked by:** none; it can run alongside 13.

**Status:** ready-for-agent

- [ ] **Reference scores:** in a throwaway environment outside the Plugin, with PyTorch, run the authors'
  `tagger_pipeline.py` on the fixture images and keep its scores.
- [ ] **Compare** the fp32 and fp16 ONNX files against those scores, using our preprocessing in ONNX Runtime:
  - the maximum and mean absolute difference;
  - which tags flip at the default thresholds;
  - the resize method: upstream uses bilinear, while Mexes's README says bicubic.
- [ ] **fp16:** whether it needs its own default thresholds.
- [ ] **Speed and memory** at `tagBatchSize` 1, 2 and 4, measured on:
  - the RTX 4070 Ti Super;
  - the RTX 3050 Ti 4 GB, in fp32 and fp16;
  - the CPU.
- [ ] **Licence** of the conversion repo, and the sha256 of each file.
- [ ] **Verdict:** pin Mexes, or make our own export, published in a repo with its script.
- [ ] **Where it goes:** the note goes in `.scratch/image-tags/research/`, and the stored scores become ticket 16's
  golden results.
