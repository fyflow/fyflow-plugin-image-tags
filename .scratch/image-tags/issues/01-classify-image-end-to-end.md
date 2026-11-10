# 01: ClassifyImage end to end on the CPU with one pinned model

**What to build:** the thinnest complete path through the Plugin. The Plugin (`fyflow-ext.image-tags`, peer `fyflow-plugin-sdk ^0.9.0`, `dataDir: true`) installs from a folder made by `pnpm plugin-folder`; the Host syncs its Python environment (cpu variant: `onnxruntime`, `pillow`, `numpy`; no imgutils at runtime) and runs a `ClassifyImage` Task on one image with `classifier: anime_real`, returning the result shape from the spec. The model is fetched on first use, inside `run()`, from its pinned Hugging Face commit over standard-library HTTPS into the data directory, checked against its sha256 and renamed into place. Preprocessing is our own and matches imgutils. See the spec's Plugin, Input, Catalogue and preprocessing, Results and Running sections (`.scratch/image-tags/spec.md`).

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Plugin layout as in the SDK README's Python Worker layout and fyflow-app's word-counter example; `ClassifyImage` is a Python Worker, not `inline`, with `maxThreads: 1`.
- [ ] Catalogue holds the `anime_real` model entry pinned to `deepghs/anime_real_cls` @ `097a6c1e9e62866a435810c883a45a7cb2c7077d`, `mobilenetv3_v1.4_dist`, with sha256s and labels checked against its `meta.json`; input size, dynamic batch dimension and softmax-or-not are read from the file before the entry is written.
- [ ] Any image Pillow opens is accepted (RGB on white); not an image or unreadable fails, naming the path.
- [ ] Result: `{ path, classifier, labels, tag, confidence, scores, models, warnings? }`; `outputSchema` with descriptions and an `examples` entry.
- [ ] A failed download or sha256 mismatch fails the Task with `Could not get the model <id>:`; a half-written file is never loaded.
- [ ] A golden script (imgutils as a dev-only extra) writes `golden.json` for a few committed images; pytest matches the top label and every score within 0.02.
- [ ] `pnpm check` runs typecheck, lint, a test that the Plugin's declarations load and validate with the SDK, and `uv run pytest`.
- [ ] One end-to-end `ClassifyImage` Task on the cpu variant passes.
