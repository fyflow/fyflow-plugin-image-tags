# 21: The tagger's embedding as a second output

**What to build:** a tagger session that gives each frame's image embedding beside its scores, in one forward pass, for
TagScenes' grouping. See `TagScenes-spec.md` §4.2.

**Blocked by:** none

**Status:** ready-for-human

- [x] `Tagger.head_node` in `catalogue.py`: `/core_model/head/Gemm` for `wd-eva02-large`, `/head/Gemm` for `pixai` and
  `pixai-fp16`.
- [x] `Models.session(tagger, ctx, embedding=True)`:
  - find the head by `head_node`, checked against a walk-back from the graph output through `Sigmoid`, `Add` and
    `Cast` to the first `Gemm` or `MatMul`
  - a rank-2 head input only; anything else, or a missing or disagreeing node, fails saying the tagger's graph
    changed
  - a float16 input is cast to float32
  - output `embedding`, `[batch, d]` float32
  - built in memory from `SerializeToString()`, with nothing written to disk, and kept like the plain session
- [x] `Models.scores_and_embeddings(tagger, batch, ctx) -> (scores, emb)`: scores handled as in `scores()`.
- [x] Out of GPU memory, download and Cancel follow the same paths as `session()` / `scores()`.
- [x] `onnx` moves from the `test` group to `dependencies`; run `uv lock`.
- [x] **Tests:**
  - the stand-in tagger gains a `Gemm` head
  - found by name; a wrong name fails
  - the width is the head's input width
  - scores are unchanged
  - fp16 comes out as float32
  - `slow`: `wd-eva02-large` and `pixai-fp16` give 1024-wide embeddings, and their scores match `scores()` to 1e-5

## Changed while building it (2026-11-19)

- The head input's width and type are read from the head's weight (an initializer): `pixai`'s graph has no shape for
  `/head_pool/If_output_0`, and `pixai-fp16`'s only its type. A `Gemm`'s input is rank 2 by the operator; a shape in
  the graph's `value_info` is checked too when there is one. A `MatMul` head with no shape in the graph fails.
- `scores_and_embeddings` also fails as `GraphChanged` if the embedding that comes out is not `[n, d]`.
- `models.py` gains `with_embedding(tagger, path) -> bytes` (the graph edit), `GraphChanged` and `EMBEDDING`;
  `runs_on_cpu(model, embedding=False)` can ask about the embedding session, so a caller need not make the plain one.
- The stand-in's head is now a feature `MatMul` (`/features/MatMul`, 8 wide: the colour, then mixes of it) and a `Gemm`
  named as `wd-eva02-large`'s that weighs only the colour, so its scores are as before. `stand_in.make(fp16=True)` runs
  the head in float16 as `pixai-fp16` does. That stand-in's scores from the embedding session differ from `scores()` by
  about 1e-4 (ONNX Runtime rounds to float16 elsewhere with the extra output), so that test allows 1e-3; the real
  `pixai-fp16` and `wd-eva02-large` match to 1e-5.
- More tests than listed: a `head_node` that is another node, a rank-4 head input, download, Cancel while downloading,
  out of GPU memory (making and running), `close()`, and nothing written to disk.
