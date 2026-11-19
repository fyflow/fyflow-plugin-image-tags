# 21: The tagger's embedding as a second output

**What to build:** a tagger session that gives each frame's image embedding beside its scores, in one forward pass, for
TagScenes' grouping. See `TagScenes-spec.md` §4.2.

**Blocked by:** none

**Status:** ready-for-agent

- [ ] `Tagger.head_node` in `catalogue.py`: `/core_model/head/Gemm` for `wd-eva02-large`, `/head/Gemm` for `pixai` and
  `pixai-fp16`.
- [ ] `Models.session(tagger, ctx, embedding=True)`:
  - find the head by `head_node`, checked against a walk-back from the graph output through `Sigmoid`, `Add` and
    `Cast` to the first `Gemm` or `MatMul`
  - a rank-2 head input only; anything else, or a missing or disagreeing node, fails saying the tagger's graph
    changed
  - a float16 input is cast to float32
  - output `embedding`, `[batch, d]` float32
  - built in memory from `SerializeToString()`, with nothing written to disk, and kept like the plain session
- [ ] `Models.scores_and_embeddings(tagger, batch, ctx) -> (scores, emb)`: scores handled as in `scores()`.
- [ ] Out of GPU memory, download and Cancel follow the same paths as `session()` / `scores()`.
- [ ] `onnx` moves from the `test` group to `dependencies`; run `uv lock`.
- [ ] **Tests:**
  - the stand-in tagger gains a `Gemm` head
  - found by name; a wrong name fails
  - the width is the head's input width
  - scores are unchanged
  - fp16 comes out as float32
  - `slow`: `wd-eva02-large` and `pixai-fp16` give 1024-wide embeddings, and their scores match `scores()` to 1e-5
