# 24: A page to review TagScenes' scenes

**What to build:** `python/scripts/review_scenes.py`, which runs TagScenes on real frame folders and writes a local
HTML page. On the page, the user marks only what is wrong, then saves a review file keyed by time. See
`TagScenes-spec.md` §7.3.

**Blocked by:** 23

**Status:** ready-for-human

- [x] Runs the `TagScenes` worker directly (no Host) on each folder or sheet given.
  - `--set NAME=VALUE` overrides a setting or a `scenes.py` constant.
  - It writes `<name>.result.json` and `<name>.review.html` to the gitignored `.scratch/image-tags/notes/review/`.
- [x] The page has one card per scene: times, rating and peaks, tags and highlights as chips, and thumbnails from file
  URLs. It is self-contained and works in light and dark.
- [x] Review by exception:
  - "merge with previous" per card
  - click a gap between thumbnails to mark a cut
  - click a chip to strike it out
  - a note per scene
- [x] **Save review** downloads `{video, settings, merges, cuts, wrong, notes}`, keyed by times.
- [x] With a saved review beside the page: "N boundaries changed since your last review", and each earlier comment is
  marked addressed or still open.
- [x] Prints times and counts only, never tag names.

## Notes

- The page is never published as an Artifact or committed: it shows frames from the user's videos.

## Changed while building it (2026-11-19)

- The script reads the input's frames itself (`read_frames`) to get each frame's time and image. TagScenes runs
  without `includeFrames`, so no frame's own tags are computed or written.
- A sheet's frames have no file, so each cell is cropped into the page as a 240-px JPEG data URI. A folder's frames
  load from their `file:///` URLs.
- The page's own script renders it from a JSON block in the page. A test holds "no external resources" (no `http(s):`,
  `@import` or `<link>`). Another runs `node --check` on the script when node is on the PATH.
- **The saved review also holds `boundaries`**: this run's scene starts, after the first. With them, the next run
  counts the boundaries that changed: the new ones with none of the old within ±1 s, and the old ones with none of the
  new. The page says "N boundaries changed since your last review (A new, G gone)". A moved boundary counts once as
  new and once as gone. If a review has no `boundaries`, the page says it cannot compare.
- **`wrong` entries carry `kind`**: `tag`, `highlight` or `rating`. A rating peak's chip is `rating` too. An entry is
  addressed when the scene covering the middle of its `start` and `end` no longer lists that name among its kind.
- A merge's `t` is the scene's `start`. A cut's `t` is the midpoint of the two frames' times, rounded to the ms.
- The comparison is a pure function, `since_last(review, result)`. It runs in Python and its answer is embedded. The
  page shows it in a side panel and pre-fills nothing. Clicking a comment scrolls to the scene at its time.
- The saved review's `settings` are the effective ones: the Node's settings (defaults, then `--set`), the `scenes.py`
  constants and `BLANK_STD`. `SHOT_SPECIFIC_TAGS` is included only when overridden. `video` is the result's `video`,
  else its `path`.
- Marks are kept in `localStorage`, keyed by the result file's name and a hash of the result. A run that changes
  anything starts clean. A **Clear marks** button empties them.
- `--set` reads a value as JSON, else as text. A constant keeps its current type: a list becomes a frozenset or a
  tuple, and a whole number becomes a float. An unknown name or a wrong type fails before anything runs. The settings
  are `tagger`, `thresholds`, `maxTags`, `embeddingWeight`, `minSceneDuration`, `ignoreTags`, `minCoverage` and
  `includeTags`.
- Extra options:
  - `--out`
  - `--data-dir`: where a tagger downloads, by default `python/.cache/review/`
  - `--model-dir`: by default `python/.cache/models/` when the tagger is in it, so nothing downloads
- What it prints for each input: the frame count, the scene count, the duration, each scene's start, the number of
  warnings, the comparison's counts and the two file paths.
- Tests: `tests/test_review_scenes.py` (15, on the stand-in tagger).

## How to run

From `python/`, on a DetectScenes folder (frames about every 2 s). Add more folders to review several videos:

```
uv run --extra cpu python scripts/review_scenes.py "C:\Users\matij\AppData\Roaming\fyflow\plugin-data\fyflow-ext.video-thumbnail\scenes\<folder>"
```

Add `--set NAME=VALUE` to vary a setting or a constant, for example `--set minSceneDuration=20` or
`--set LINK_PERCENTILE=80`.

1. Open `.scratch/image-tags/notes/review/<name>.review.html`.
2. Mark what is wrong, then click **Save review**.
3. Move the downloaded `<name>.review.json` into that folder.

The next run on the same video shows what changed since that review.
