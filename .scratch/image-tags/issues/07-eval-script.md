# 07: Eval script

**What to build:** a by-hand script that measures the classifiers on the user's own videos and settles the provisional defaults. Not part of `pnpm check`; run before a release is tagged. See the spec's Testing Decisions (Eval set).

**Blocked by:** 03, 04

**Status:** resolved

- [x] Takes a CSV of `path,label` (labels anime, cg, live_action) and the `thumbs-collage` binary's path; no Host needed.
- [x] Makes a sheet per video with the README's recommended settings and runs it through the same engine as `ClassifyFrames`.
- [x] Prints a confusion matrix for `style`, and for the raw models against the labels they can express; the time per video; and the `minConfidence` that best separates the classes.
- [x] Can run `anime_classification` v1.3 and v1.5 side by side to pick one.
- [x] README section on how to run it and what it decides (`anime_classification` version, `style`'s default `minConfidence`, whether `cg` needs another model).

## Notes

- **Known `cg` weakness (found 2026-11-10 on the reference frames in `python/tests/fixtures/images/`):** `style` tags four of the five Blender CG films (`cg-01` to `cg-04`) `live_action` (0.72–0.93); only the flat-shaded `cg-05-coffee-run` comes out `cg`. On those films `anime_classification`'s top label is mostly `3d` (`illustration` for Cosmos Laundromat). They come out live action because `anime_real` calls them real, and the rule takes `live_action` straight from `anime_real`. So try a rule change before reaching for another model, e.g. let a strong `3d` override `anime_real`'s `real`. Include the reference frames in the eval alongside the user's own videos.

## Result (2026-11-10)

Run on 132 videos of the user's own library (read in place over the network; only the sheets were kept, in the gitignored cache) plus the reference frames. Decided, for the follow-up change: `anime_classification` v1.5; `style`'s rule `real_pow4` (live action `real⁴`); default `minConfidence` 0.55. Anime matters most: with that, every anime video is found and nothing else is tagged anime among the labelled items. CG vs live action needs another model (ticket 08), and "anime" covers Western 2D too. The README's Eval section has the numbers.

Found on the way: a `.ts` broadcast recording whose 16 frames came out identical (a wrong file, or the thumbnails service not seeking in `.ts`: worth a test there), and a video that fails to decode.
