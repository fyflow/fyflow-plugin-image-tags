# 22: Scene grouping and pooling (`scenes.py`)

**What to build:** the pure-numpy core of TagScenes: frames with timestamps, embeddings and sparse scores in, then
scenes with tags, highlights, rating, keyframe and a video summary out. See `TagScenes-spec.md` §5. No ONNX and no I/O.

**Blocked by:** none (synthetic inputs; the embedding is only an array here)

**Status:** ready-for-human

- [x] Frame spans (§5.1): halfway to each neighbour, from 0 to `duration` or to the last frame plus half the median
  gap. Duplicate timestamps share their time.
- [x] Time order: a stable sort by timestamp. Output indices refer to the input order.
- [x] Tag vectors (§5.2):
  - general and character only
  - `SHOT_SPECIFIC_TAGS` (the reference's 43) and `ignoreTags` left out
  - idf over non-blank frames, characters × `CHARACTER_WEIGHT`
- [x] Similarity (§5.3):
  - centred embedding cosine
  - tag cosine
  - each z-scored over the candidates, mixed by `embeddingWeight`
  - `w = 1` when no tags are left
- [x] Linking (§5.4):
  - candidates within `LINK_WINDOW` seconds and `MAX_LINK_LAG` non-blank frames
  - linked at the `LINK_PERCENTILE`, with an `S_emb ≥ 0` floor
  - boundaries where no link crosses
  - blank bridging
  - short scenes merged into the more similar neighbour
  - `link` kept per boundary
- [x] Pooling (§5.5):
  - smoothing within each scene and hysteresis with `OFF_GAP` per category
  - tags by coverage
  - highlights from raw scores
  - rating with `peaks`
  - keyframe closest to the scene's mean
  - the summary
  - `maxTags` caps every list; `includeTags: false` leaves tags and highlights out
- [x] Constants as in §5.6, module-level.
- [x] **Tests:** the cases in §7.1.

## Notes

- Grouping takes the tag list's names and categories, and the thresholds, as plain arrays, so the tests don't need a
  tagger.

## Changed while building it (2026-11-19)

**Interface** (`python/image_tags/scenes.py`):

- `Scores(rows, tags, values)`: sparse scores, CSR-style (`rows` is `[n + 1]` offsets, `tags` places in the tag
  list). `Scores.from_dense(batch, floor=SCORE_FLOOR)` per batch, `Scores.stack(parts)` to join the batches,
  `scores.entries(frames) -> (row, tag, value)`. Scenes densify only their own tags, 512 at a time.
- `TaggedFrames(timestamps, embeddings, scores, ratings, blank, paths, names, categories, rating_labels, thresholds,
  duration=None)`, all in input order. `categories` is each tag's category name (`None` or anything else for tags
  not pooled, such as the ratings), so the worker maps `Labels.codes` through `tagger.categories`. `rating_labels`
  run from least to most severe (the result's names). A frame's `frame` in the output is its place in these arrays.
- `Settings(embedding_weight=0.6, min_scene_duration=15, ignore_tags=frozenset(), min_coverage=0.3, max_tags=40,
  include_tags=True)`.
- `group(frames, settings) -> Grouping(scenes, links, scene_of, threshold, …)`: `scenes[k]` the input places in time
  order, `scene_of[f]` each input frame's scene (for `includeFrames`), `threshold` None when no pair was compared.
- `pool(frames, grouping, settings) -> {"threshold", "summary", "count", "scenes"}` in §3.4's shapes. The worker adds
  `path`, `video`, `tagger`, `frameCount`, `duration`, `frames` and `warnings`.
- `frame_spans(times_in_order, duration)` is public, for the tests.

**Where the spec was read one way of several:**

- A scene's `frames` are in time order.
- "The frames either side link to each other" (blank bridging) is read as "no boundary falls between them", i.e. some
  link crosses the blanks. Blanks before the first non-blank frame join the first scene, blanks after the last join
  the last; a blank run longer than `minSceneDuration` is its own scene wherever it is, even inside linked frames.
- `coverage` is over the scene's whole time, blank frames included (they count towards time, and are never on).
- A tag's `score` is the mean raw score over the time it is on (not the smoothed one).
- On and off use `≥` ("turns on at", "reaches"), where `catalogue.tag` uses `>`; it only differs at equality.
- Scene `tags` include shot-specific tags and `ignoreTags`: §5.5 leaves them out of highlights only, and the
  settings table says `ignoreTags` is for similarity and highlights. Worth a look in review.
- Highlights skip the tags in the scene's listed `tags` (after the `maxTags` cap). A highlight's `frames` counts
  frames at `PEAK_REPEAT` for characters too, so a PixAI character (threshold 0.27) can show `frames: 0`.
- Smoothing weights each other frame by its span times the Gaussian, leaves out frames more than
  `SMOOTHING_REACH` (4) σ away, and gives a frame's own score `max(the others' weight, its span)`. Hysteresis starts
  off at each scene's start.
- Keyframe: the highest cosine between a frame's unit centred embedding and the span-weighted mean of them.
- Merging compares a short scene with each neighbour by the mean centred cosine over all their non-blank frame pairs
  (from per-scene sums, never N × N). A neighbour with no non-blank frame is the less alike; a tie goes to the earlier.
- An all-blank scene's rating comes from its blank frames (a scene needs one); its tags are empty. The summary's
  rating scores are over the non-blank frames.
- Spans: the median gap is over distinct times; a `duration` before the last timestamp is ignored.
- Rounding: seconds to 3 places, scores, coverage and z-values to 4.
- New module constants beside §5.6: `DF_SCORE` 0.2 and `IDF_FLOOR` 0.1 (the idf's numbers), `SMOOTHING_REACH` 4,
  `CATEGORIES`. `BLANK_STD` stays for `frames.py` (ticket 23).

**Seen while testing:** the link threshold is a percentile of the video's pairs, so a short video of only two long,
uniform scenes is often cut inside one of them (about 1 seed in 4 for 60 s + 40 s at the tests' noise); with three or
more scenes it holds. The tests plant three or more where they count boundaries. Worth watching in the review (§7.3).
At 10000 frames with a 13000-tag list and 60 random tags a frame, `group` takes about 1.5 s and `pool` about 4 s.

- **Decided (Q36):** `ignoreTags` is left out of scene and summary tags too; shot-specific tags stay listed there.
