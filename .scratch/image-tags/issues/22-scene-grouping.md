# 22: Scene grouping and pooling (`scenes.py`)

**What to build:** the pure-numpy core of TagScenes: frames with timestamps, embeddings and sparse scores in, then
scenes with tags, highlights, rating, keyframe and a video summary out. See `TagScenes-spec.md` §5. No ONNX and no I/O.

**Blocked by:** none (synthetic inputs; the embedding is only an array here)

**Status:** ready-for-agent

- [ ] Frame spans (§5.1): halfway to each neighbour, from 0 to `duration` or to the last frame plus half the median
  gap. Duplicate timestamps share their time.
- [ ] Time order: a stable sort by timestamp. Output indices refer to the input order.
- [ ] Tag vectors (§5.2):
  - general and character only
  - `SHOT_SPECIFIC_TAGS` (the reference's 43) and `ignoreTags` left out
  - idf over non-blank frames, characters × `CHARACTER_WEIGHT`
- [ ] Similarity (§5.3):
  - centred embedding cosine
  - tag cosine
  - each z-scored over the candidates, mixed by `embeddingWeight`
  - `w = 1` when no tags are left
- [ ] Linking (§5.4):
  - candidates within `LINK_WINDOW` seconds and `MAX_LINK_LAG` non-blank frames
  - linked at the `LINK_PERCENTILE`, with an `S_emb ≥ 0` floor
  - boundaries where no link crosses
  - blank bridging
  - short scenes merged into the more similar neighbour
  - `link` kept per boundary
- [ ] Pooling (§5.5):
  - smoothing within each scene and hysteresis with `OFF_GAP` per category
  - tags by coverage
  - highlights from raw scores
  - rating with `peaks`
  - keyframe closest to the scene's mean
  - the summary
  - `maxTags` caps every list; `includeTags: false` leaves tags and highlights out
- [ ] Constants as in §5.6, module-level.
- [ ] **Tests:** the cases in §7.1.

## Notes

- Grouping takes the tag list's names and categories, and the thresholds, as plain arrays, so the tests don't need a
  tagger.
