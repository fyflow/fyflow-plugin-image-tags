# TagScenes: spec

A new Node Type in the image-tags Plugin (`fyflow-ext.image-tags`). It tags a video's frames and groups them into
scenes in one Task, using each frame's **image embedding** as well as its tags. The point is **harvesting tags**: each
scene's tags, highlights and rating, and the whole video's. Exact scene boundaries matter less.

Status: agreed · 2026-11-19 (design review; replaces the 2026-11-19 draft)
Ships in: 0.4.0, with tickets 18 and 19 (not yet released)
Tickets: 21–24
Reference: `C:\Users\matij\Documents\dev\video-classification-experiments\scene_tagger.py` (`group_shots`,
`pool_scenes`, `with_embedding_output`, `SHOT_SPECIFIC_TAGS`, `write_report`). It is a starting point, not a target:
its results were acceptable but not ideal.

---

## 1. Goals and non-goals

**Goals**
- Per video:
  - a list of scenes, each with its time span, tags, short-lived highlights (each with the frame where it peaks), a
    rating with its evidence, and a keyframe
  - a summary of the whole video's tags and rating
- Grouping that holds up on real content:
  - dialogue cutting (A/B/A/B) stays in one scene
  - a change of location or cast starts a new scene
  - a few blank or title frames don't break a scene
- The same input as TagFrames: frames with timestamps.
- No per-frame data in the output unless asked for.

**Non-goals**
- **No shots.** TagScenes groups a flat list of timestamped frames. It reads no `scenes.json` or `frames.json`, and
  no manifest at all. A boundary falls halfway between two frames, up to about half a sampling interval off, which
  doesn't matter for harvesting tags.
- **No cache of scores or embeddings.** They live in memory for the Task only. A finished result can be reused as a
  file (§3.5), the same way as with the other frames Node Types.
- **No derived model file.** The embedding session is built in memory (§4.2).
- No change to the behaviour or output of ClassifyFrames, TagFrames or TagImage.
- Not in this version:
  - novelty splitting
  - long-take splitting
  - `maxSceneDuration`
  - distance decay in the link score

  Add one only if reviews (§7.3) show the need. For example, if two scenes chain together through shared tags, novelty
  splitting is the candidate fix.

---

## 2. Where it sits in a Flow

```
VideoFiles → DetectScenes (frames: every 2 s) → TagScenes(path = DetectScenes.folder) → …
VideoFiles → ExtractFrames (every 2 s)        → TagScenes(path = ExtractFrames.folder) → …
VideoFiles → VideoThumbnail (sheet)           → TagScenes(path = sheet) → …
```

Each of these gives frames with timestamps: DetectScenes and ExtractFrames JPEGs in their COM comment, and a sheet in
its JSON. The README recommends sampling **about every 2 s**, and with DetectScenes, a `maxFrames` above the expected
frame count. The real DetectScenes runs on this machine used `max_frames` 400, and one 2629 s video lost 67 shots'
frames to that cap.

---

## 3. Interface

### 3.1 Node Type

| | |
|---|---|
| Name | `TagScenes` |
| Worker | `image_tags.worker:TagScenes`, a subclass of `_Tag` with a sync `run`, as `TagFrames` |
| Pool | as `TagFrames`: `maxConcurrentTasks: 2`, the same `RESIDENT_GROUPS` (holds `gpu`), the same `POOL_VARIANTS` |
| Icon | `tag` |
| Description | "Tags a video's frames and groups them into scenes: each scene's tags, highlights and rating, and the video's." |
| `roots` | yes, as TagFrames: through `_Worker._frames` and `read_frames` / `read_frame_list` |

### 3.2 Input

The input is `framesInputSchema`, unchanged: exactly one of `path` or `frames`.

| Field | |
|---|---|
| `path` | a folder of frames, a VideoThumbnail frame sheet, or one image |
| `frames` | a list of `{path, timestamp?}`, up to 10000 entries |

The frames are read by `read_frames` / `read_frame_list` (`frames.py`), unchanged. Timestamps come from:
- the list's `timestamp`
- otherwise a thumbnail's COM comment (`timestamp`, `video`, `duration`)
- or a sheet's `timestamp_actual`

The limit is the shared `MAX_FOLDER_IMAGES` / `MAX_LISTED_FRAMES` of 10000.

**Every frame needs a timestamp.** If any frame has none, the Task fails before any model runs. The message names how
many frames lack one and says how to get them: frames from VideoThumbnail, DetectScenes or ExtractFrames, or a
`frames` list with `timestamp`. Grouping runs on time (the link window, the minimum scene length, coverage), so a
guessed spacing would give made-up scene times that look real.

**Duration:** `frames.py` also reads the thumbnail comment's `duration` (in seconds) onto `Frames`, when the frames
agree on one; a sheet gives none. This is the only change to `frames.py` besides blank detection (§4.3), and it
doesn't change ClassifyFrames' or TagFrames' output.

### 3.3 Settings

| Setting | Default | |
|---|---|---|
| `tagger` | `wd-eva02-large` | as TagFrames |
| `thresholds` | the tagger's | as TagFrames: when a tag counts as **on** (§5.5). Also the bar for character highlights |
| `maxTags` | 40 | the most entries in any one list: a scene's tags, a scene's highlights, the summary's tags, and, with `includeFrames`, a frame's tags |
| `embeddingWeight` | 0.6 | how much the image embedding counts against the tags when comparing frames (§5.3). 0 = tags only, 1 = image only |
| `minSceneDuration` | 15 | seconds. Shorter scenes merge into their more similar neighbour |
| `ignoreTags` | [] | more tags to leave out of similarity and highlights, beside the built-in shot-specific list |
| `minCoverage` | 0.3 | a scene lists a tag when it is on for this share of the scene's time |
| `includeTags` | true | hand on each scene's tags and highlights. With false, a scene has only times, frames, rating and keyframe |
| `includeFrames` | false | also hand on each frame's own tags and rating (§3.4). When off, no per-frame tags appear anywhere: not in the result, not in the file |
| `lists`, `output`, `overwrite` | `result`, none, false | as the other frames Node Types (`fileSettingsProperties`), see §3.5 |

Everything else is a named constant in `scenes.py` (§5.6). A constant becomes a setting only when reviews show it needs
to vary between videos.

### 3.4 Output

```jsonc
{
  "path": "C:/…/scenes/ep01.1c8fcf3f1311",   // as given; absent for a `frames` list
  "video": "C:/…/ep01.avi",                   // when the frames name one
  "tagger": "wd-eva02-large",
  "frameCount": 278,
  "duration": 1510.26,
  "threshold": 1.12,                          // the link threshold used, in z-units (§5.4)
  "summary": {
    "tags": [ { "name": "…", "category": "general", "coverage": 0.64, "score": 0.78 } ],
    "rating": { "label": "general", "scores": { "general": 0.9, "sensitive": 0.08, "questionable": 0.02, "explicit": 0 } }
  },
  "count": 31,                                // scenes
  "scenes": [
    {
      "index": 0,
      "start": 0, "end": 74.3, "duration": 74.3,
      "frames": [0, 1, 2],                    // places in the input frame list
      "boundary": { "link": 0.41 },           // absent on scene 0: the strongest link across the cut (z), null if none
      "keyframe": { "frame": 1, "timestamp": 12.4, "path": "C:/…/frame_0005.jpg" },
      "rating": {
        "label": "general",
        "scores": { "general": 0.88, "sensitive": 0.1, "questionable": 0.01, "explicit": 0 },
        "peaks": [ { "label": "sensitive", "score": 0.62, "frame": 2, "timestamp": 30.1, "path": "…" } ]
      },
      "tags": [ { "name": "…", "category": "general", "coverage": 0.92, "score": 0.81 } ],
      "highlights": [ { "name": "…", "category": "general", "score": 0.83, "frames": 1,
                        "frame": 2, "timestamp": 21.5, "path": "…" } ]
    }
  ],
  "frames": [ … ],                            // only with includeFrames
  "warnings": [ … ]
}
```

- `frame` everywhere is the place in the **input** frame list: the folder's order by name, or the list's order. `path`
  is that frame's image, as `_named` gives it, or absent for a sheet's cell.
- With `includeFrames`, each frame is TagFrames' per-frame shape (`timestamp`, `path`, `tags`, `rating`, and `video`
  when its thumbnail names one), plus `blank` (bool) and `scene` (its scene's `index`).
- `peaks` lists only ratings more severe than `label`, each at the frame where it scores highest.
- `highlights[].frames` is how many frames reached `PEAK_REPEAT` (§5.5).
- With `includeTags: false`, scenes and the summary have no `tags` or `highlights`.
- The schemas go in `plugin.ts` beside TagFrames' (`TAGGED_FRAME` is reused for `frames`), with `examples`. No field is
  required that `lists: "file"` takes away.

### 3.5 Answers in a file

This works as in ticket 19, through `results.py` and `_Worker._hand_on`:
- With `lists: "file"`, **`scenes` and `frames`** go to the file, which holds the whole result.
- The Task hands on the rest: `summary`, the counts, `threshold`, `warnings`, plus `file` and `reused`.
- `count` is the number of scenes, in both cases.
- Reuse works as before: the key covers the Node Type, the input, each frame file's stat, the effective settings and
  the tagger's commit.

`results.summary` and `reused` take the names of the list fields to move: `("frames",)` for the other frames Node
Types, `("scenes", "frames")` here. `count` comes from the caller.

With `lists: "result"`, the 15 MiB check (`results.fits`) applies as before.

---

## 4. Running the tagger

### 4.1 Order of work

1. Check the settings and file options, then read the frames (§3.2). Fail on a missing timestamp.
2. Work out the file target and look for reuse, as `_hand_on` does. A reused file hands on with no model loaded.
3. Tag and embed in batches (§4.2). Keep only what §4.4 lists.
4. Group (§5), pool (§5.5), and hand on.

Frames are **sorted by timestamp** (a stable sort, so ties keep input order) for grouping and pooling, while every
`frame` index in the output refers to the input order.

### 4.2 The embedding output (`models.py`, `catalogue.py`)

The embedding is the tagger's pooled features, the data input of its final Gemm. All three catalogue taggers have one,
1024 wide:

| Tagger | `head_node` | Head input | Notes |
|---|---|---|---|
| `wd-eva02-large` | `/core_model/head/Gemm` | `/core_model/fc_norm/LayerNormalization_output_0`, `[N, 1024]` float32 | output `output`, after `/final_act/Sigmoid` |
| `pixai` | `/head/Gemm` | `/head_pool/If_output_0`, float32 | output `prediction`, after `/Sigmoid`. 1.98 GB, under the 2 GB protobuf limit |
| `pixai-fp16` | `/head/Gemm` | `/head_pool/If_output_0`, **float16** | output `prediction` (float32), through `Cast` `/Sigmoid_cast_to_prediction` → `Sigmoid` → Gemm |

- `Tagger` gains `head_node: str` (`catalogue.py`), set for all three.
- `Models.session(tagger, ctx, embedding=True)` gives a second session whose graph has a second output, `embedding`,
  built at first use and kept like the first:
  1. `onnx.load(path)` and find the node named `head_node`. As a check, walk back from the graph output through
     `Sigmoid`, `Add` and `Cast` to the first `Gemm` or `MatMul`. If they disagree, or the node is missing, fail
     saying the tagger's graph changed.
  2. Take the head's data input. A `Gemm` input is rank 2 (`[batch, d]`); anything else fails clearly. There is no
     ReduceMean.
  3. If it is float16, add a `Cast` to float32. Add an `Identity` named `embedding` and append it as a graph output,
     typed float32 with shape `[batch, d]`.
  4. `ort.InferenceSession(model.SerializeToString(), …)` with the same providers and options as the plain session.
     **Nothing is written to disk.**
- `Models.scores_and_embeddings(tagger, batch, ctx) -> (scores [n, tags], emb [n, d])` runs both outputs in one
  forward pass. Scores are handled as `scores()` handles them, sigmoided or not per `Tagger.sigmoided`.
- Out of GPU memory, model download, Cancel and the `gpu` unit all behave as for TagFrames, through the same `Models`
  paths. TagScenes' pool is its own, so the embedding session never shares a process with TagFrames' plain one.
- `onnx` moves from the `test` group to `dependencies` in `pyproject.toml`; run `uv lock`.

### 4.3 Blank frames (`frames.py`)

When a frame is loaded for its batch, compute its grayscale standard deviation with `ImageStat` on a 64-px
thumbnail. A frame with std < `BLANK_STD` (4, out of 255) is **blank**: black, white or a solid card. Blank frames are
tagged as usual, but marked, and grouping bridges them (§5.4).

### 4.4 What is kept per frame

To keep memory flat at 10000 frames:
- `emb`: float32 `[d]`, L2-normalised. At d = 1024 that is 4 KB per frame, 40 MB at 10000.
- `scores`: sparse, only tags at or above `SCORE_FLOOR` (0.05), as (index, score) pairs. Ratings are kept whole.
- `blank` and `timestamp`.
- With `includeFrames`, also the frame's own `tags` and `rating`, from `catalogue.tag(...)` as TagFrames gives them.

Each batch's full score row is dropped after the batch.

### 4.5 Progress and Cancel

- Tagging reports progress from 0 to 0.9, per batch ("Tagging n of N frames", "Tagged …"), through `_frame_by_frame`
  or a version of it that also returns embeddings.
- Grouping and pooling report 0.9 to 1.0 ("Grouping", "Pooling").
- Cancel is checked between batches, as `_frame_by_frame` does, and between grouping stages.

---

## 5. Grouping and pooling (`scenes.py`)

This is pure numpy in a new module `python/image_tags/scenes.py`, with no ONNX and no I/O, so it can be tested on its
own (§7.1). Input: per frame, in time order, `timestamp`, `emb`, sparse `scores`, `blank`, and `path`/`index` for the
output; plus the tag list's names and categories, the thresholds, `duration` (optional) and the settings.

### 5.1 Frame spans

Each frame is the unit that gets linked.

- A frame spans from halfway to the previous frame's time to halfway to the next one's.
- The first frame's span starts at 0.
- The last frame's span ends at `duration` when known. Otherwise it ends at its own time plus half the median gap
  between frames (or plus 1 s for a single frame).
- Frames with the same timestamp share the time between them equally.
- A frame's **weight** in every duration-weighted mean is its span's length.

### 5.2 Tag vectors

- Only the `general` and `character` categories count.
- Leave out `ignoreTags` and the built-in `SHOT_SPECIFIC_TAGS`: framing, camera angle, gaze, expression, head counts,
  screencap and border artefacts. These change with every cut inside one scene. Copy the 43 entries from the
  reference's `SHOT_SPECIFIC_TAGS` into `scenes.py`. They name Danbooru tags, which both taggers use.
- A tag's value is the frame's score. Scores below `SCORE_FLOOR` count as 0. Using raw scores rather than on/off tags
  lets near-threshold evidence count.
- Weight each tag by `idf = log((N + 1) / (df + 1)) + 0.1` over the video's non-blank frames, where `df` is the number
  of frames scoring 0.2 or more on it. Tags present all through the video (the main character, the art style) then
  count little. Multiply character tags by `CHARACTER_WEIGHT` (1.5).

### 5.3 Similarity

Similarity is computed for candidate pairs only (§5.4), never the full N × N.

- `S_emb(i, j)` is the cosine of the **centred** embeddings: subtract the video's mean embedding (non-blank frames),
  then L2-normalise. Without centring, every frame of one show sits near cosine 0.8 (same art style), and the
  differences that matter drown.
- `S_tag(i, j)` is the cosine of the tag vectors, or 0 when either is empty.
- **Calibrate before mixing.** The two scales differ and shift from video to video, so each is z-scored over the
  video's candidate pairs: `z = (S − mean) / std`, with std floored at 1e-6.
- `S(i, j) = w · z_emb + (1 − w) · z_tag`, with `w = embeddingWeight`. When no frame has any tag left after §5.2,
  `w = 1`.

### 5.4 Linking

This follows the reference's `group_shots`, run on frames, with these changes:

1. **Candidates:** pairs `i < j` of non-blank frames with `j − i ≤ MAX_LINK_LAG` (15, counting non-blank frames only)
   and `t_j − t_i ≤ LINK_WINDOW` (30 s).
2. **Threshold:** the `LINK_PERCENTILE` (85th) of `S` over the candidates. The result's `threshold` is this value.
   **Floor:** a pair also needs `S_emb ≥ 0` (centred, so at least as similar as the video's average) to link.
3. **Link** `i → j` when `S ≥ threshold`. `reach[i]` is the largest `j` linked from `i`.
4. **Boundaries:** a cut before frame `k` is a boundary when no link crosses it (`max reach[0..k−1] < k`).
5. **Blank frames** between two non-blank frames:
   - If the frames either side link to each other, the blanks join that scene.
   - Otherwise the boundary goes after the blanks, so blank frames end a scene rather than start the next.
   - A run of blanks longer than `minSceneDuration` is a scene of its own.
6. **Merge short scenes:** while the shortest scene is under `minSceneDuration`, merge it into the neighbour with the
   higher mean `S_emb` between their frames (computed for these pairs only). The merged cut's boundary goes away.

For each remaining boundary, keep `link`: the strongest `S` among candidate pairs crossing it, or null if there are
none. With 0 non-blank frames, the whole video is one scene. With 0 frames, the input check (§3.2) has already failed.

### 5.5 Pooling (per scene, and for the summary)

All means are duration-weighted over the scene's frames (§5.1). Blank frames count towards time but not towards tags
or rating.

- **Tag on/off:**
  - Smooth each frame's scores with a Gaussian over frame times (σ = `SMOOTHING`, 4 s), **within the scene only**, so
    nothing leaks across a boundary. Each frame keeps at least half of its own weight.
  - Then apply hysteresis per category: a tag turns on at its category's threshold (`thresholds`) and stays on down
    to the threshold minus `OFF_GAP`, which is 0.10 for general and 0.15 for character.
- **Tags:**
  - `coverage` is the share of the scene's time during which the tag is on.
  - `score` is the mean score while it is on.
  - List the tags with `coverage ≥ minCoverage`, general and character only, ordered by coverage then score, at most
    `maxTags`.
- **Highlights** come from raw, unsmoothed scores:
  - A general tag qualifies when it reaches `PEAK_STRONG` (0.7) in one frame, or `PEAK_REPEAT` (0.5) in two frames.
  - A character qualifies when it reaches the character threshold in two frames, since single-frame misfires are
    common.
  - Skip shot-specific tags, `ignoreTags`, and tags already in `tags`.
  - Each highlight gives `score` (its peak), `frames` (how many frames reached `PEAK_REPEAT`), and the peak frame's
    `frame`, `timestamp` and `path`.
  - Order by score; at most `maxTags`.
- **Rating:**
  - `label` is the most severe rating reaching `PEAK_REPEAT` in two frames, or `RATING_STRONG` (0.9) in one.
    Otherwise it is the highest of the mean scores.
  - `scores` are the mean scores.
  - `peaks` are the ratings more severe than `label` that reach `PEAK_REPEAT` in any frame, each at its peak frame.
- **Keyframe:** the non-blank frame whose centred embedding is closest to the scene's mean centred embedding. For an
  all-blank scene, the middle frame.
- **Summary:**
  - `tags`: each tag's coverage over the whole video, computed scene by scene as above (so smoothing never crosses a
    boundary) and weighted by scene time. `score` is the mean score while on. `minCoverage` and `maxTags` apply.
  - `rating.label`: the most severe `label` of any scene. `rating.scores`: the mean scores over the video.

### 5.6 Constants

`SCORE_FLOOR` 0.05, `BLANK_STD` 4, `LINK_WINDOW` 30, `MAX_LINK_LAG` 15, `LINK_PERCENTILE` 85, `CHARACTER_WEIGHT` 1.5,
`SMOOTHING` 4, `OFF_GAP` {general 0.10, character 0.15}, `PEAK_STRONG` 0.7, `PEAK_REPEAT` 0.5, `RATING_STRONG` 0.9,
and `SHOT_SPECIFIC_TAGS` (43). All are module-level in `scenes.py` (`BLANK_STD` in `frames.py`), so the review script
(§7.3) can vary them without a setting.

---

## 6. Changes, file by file

| File | Change | Ticket |
|---|---|---|
| `python/image_tags/catalogue.py` | `Tagger.head_node` for all three taggers | 21 |
| `python/image_tags/models.py` | the embedding session and `scores_and_embeddings` (§4.2) | 21 |
| `python/pyproject.toml`, `uv.lock` | `onnx` into `dependencies` | 21 |
| `python/image_tags/scenes.py` | **new**: §5, pure numpy | 22 |
| `python/image_tags/frames.py` | blank detection (§4.3), `duration` from thumbnail comments (§3.2) | 23 |
| `python/image_tags/results.py` | `summary` / `reused` take the list fields to move, and `count` from the caller (§3.5) | 23 |
| `python/image_tags/worker.py` | `class TagScenes(_Tag)`: read, check timestamps, tag and embed in batches, `scenes.group`, `scenes.pool`, hand on | 23 |
| `plugin.ts` | worker and Node Type `TagScenes`, settings, input and output schemas with examples | 23 |
| `README.md` | the Node Type, the Flows of §2, the settings, "about every 2 s" | 23 |
| `python/scripts/review_scenes.py` | **new**: the review page (§7.3) | 24 |

---

## 7. Testing

**Privacy rule for the agent:** tests, logs and scripts print **times, counts and metrics only**, never tag names or
scores from real videos. The one exception is a review file the user saves (§7.3). The agent reads the tag names
marked in it, because acting on a "wrong tag" needs its name, but never echoes them into tracked files.

### 7.1 Unit tests for `scenes.py` (no model)

No images are involved: `scenes.py` takes arrays, so a test builds those arrays directly with a seeded
`numpy.random.default_rng`.
- **Embeddings:** each planted scene (or each "camera" of a scene, for A/B cutting) gets a random unit centre in 64
  dimensions. Its frames are that centre plus Gaussian noise, then normalised. The noise level sets how hard the case
  is: the A and B cameras of one scene are closer to each other than to another scene.
- **Tags:** a made-up tag list of about 30 names in `general` and `character`. Each scene has a few "location" tags
  scored around 0.8 with noise, a character shared by every scene where a case needs one, and random low scores below
  the threshold. Shot-specific names from `SHOT_SPECIFIC_TAGS` are mixed in where a case checks they're ignored.
- **Times:** every 2 s by default, uneven or out of order where a case needs it.
- **Blank:** a frame flagged blank, with a zero embedding.

The helper is one function in the test file, e.g. `planted(scenes=[(40, "AB"), (30, "A")], noise=0.3, seed=1)`. It
returns the arrays and the planted boundaries. A case then asserts the found boundaries against the planted ones.
These tests check the logic, not the quality on real video. Quality comes from §7.2's fixture test and the review page
(§7.3).

Cases:
- Three planted scenes with A/B alternation inside: boundaries found within one frame.
- Two scenes sharing one character but with different embeddings: kept apart, through the embedding.
- Blank frames inside a scene are bridged; blank frames between scenes end the first scene; a long blank run becomes
  its own scene.
- Frame spans:
  - uneven sampling
  - duplicate timestamps
  - `duration` given or not
- Out-of-order input: grouped in time order, with indices in input order.
- The `minSceneDuration` merge goes to the more similar neighbour.
- Edge cases:
  - 1 frame
  - all frames blank
  - no tags left after §5.2 (`w = 1`)
  - `embeddingWeight` at 0 and at 1
- Pooling:
  - hysteresis per category
  - smoothing never crosses a boundary
  - highlights and rating `peaks` carry the right `frame` and `path`
  - `maxTags` caps every list
  - the summary's rating is the most severe scene's

### 7.2 Model tests

- Extend the stand-in tagger (`conftest.py`, built with `onnx.helper`) with a `Gemm` head after a feature node. Check:
  - the embedding is found by `head_node`
  - a wrong `head_node` fails clearly
  - its width is the head's input width
  - scores are unchanged with the extra output
  - an fp16 head input comes out as float32
- `TagScenes` through the worker on the stand-in:
  - a missing timestamp fails before any model loads
  - `includeFrames` on and off
  - `lists: "file"` writes and reuses, with `scenes` and `frames` in the file and `summary` and `count` in the
    result
  - Cancel between batches
- `slow` marker, skipped unless the model is on disk: the real `wd-eva02-large` gives a 1024-wide embedding, and its
  scores match TagFrames' on the same frames to 1e-5. The same goes for `pixai-fp16`.
- **A planted video from the repo's own images** (`slow`, real `wd-eva02-large`): a `frames` list built from
  `tests/fixtures/images/anime-*.jpg`, written to `tmp_path`.
  - Three "scenes" of 30–40 s each, at a frame every 2 s.
  - Each frame is a different crop, flip and brightness of its scene's image, so no two frames are identical.
  - One scene alternates two crops of the same image far apart (A/B cutting).
  - A few black frames lie between two scenes and inside one.

  The test asserts that the boundaries fall within one frame of the planted ones, that the blanks are bridged, and that
  each scene's tags are non-empty. These images are AI-generated test fixtures, not the user's videos, so printing
  their tags is fine.

### 7.3 Review page (`python/scripts/review_scenes.py`)

This is a dev script, not shipped in the Plugin and not published anywhere, because it shows frames from the user's
videos.

- **Run:** `uv run python scripts/review_scenes.py <folder or sheet> … [--set NAME=VALUE …]`.
  - It runs the `TagScenes` worker directly on each input, with no Host, as `eval.py` drives workers.
  - `--set` overrides a setting or a `scenes.py` constant.
  - It writes `<name>.result.json` and a self-contained `<name>.review.html` to the gitignored
    `.scratch/image-tags/notes/review/`.
- **Page:** one card per scene, starting from the reference's `write_report`:
  - the scene's times, duration, rating and peaks
  - its tags and highlights as chips
  - its frames as thumbnails, loaded from their file URLs
- **Review by exception:** everything counts as right until clicked.
  - A **"merge with previous"** toggle on each card's header.
  - **Click the gap between two thumbnails** to mark "a scene starts here".
  - **Click a chip** to strike out a wrong tag, highlight or rating.
  - A **note** box per scene, for anything else, such as a missing tag.
  - **Save review** gives a JSON file keyed by **time**, not scene number, so comments still match after the grouping
    changes. The format is `{video, settings, merges: [t], cuts: [t], wrong: [{tag, start, end}], notes: [{start,
    end, text}]}`. The page saves it with a download, since it is a local file and not an Artifact. The user keeps it
    in the same folder.
- **Since last review:** when a saved review is beside it, the page says how many boundaries changed and marks each
  earlier comment as addressed or still open.
- One run per page; there is no side-by-side view.

### 7.4 Through fyflow

1. `check_plugin` and `reload_plugins`.
2. A `try_node_type` on a small DetectScenes folder with `includeTags: false`, then the same with `lists: "file"`.
3. The `testTagging` Flow with TagScenes in place of TagFrames.

---

## 8. Performance budget

For a 25-min episode of about 280–750 frames:
- **Tagging:** as TagFrames now. The extra output adds no measurable time.
- **Embedding session:** built once per process. `onnx.load` plus serialising takes about 1 s and briefly needs about
  twice the model's size in RAM.
- **Grouping and pooling:** under 1 s. There are at most N × 15 candidate pairs, and smoothing runs within each scene.
- **Memory beyond the model:** under 10 MB at this size, about 50 MB at 10000 frames.
