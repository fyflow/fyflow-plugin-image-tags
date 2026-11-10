# 03: The anime_classification model, the style preset and uncertain

**What to build:** `ClassifyImage` answers anime / cg / live_action, or `uncertain`, by default. The catalogue gains the `anime_classification` model and the `style` preset; `classifier` becomes the full enum (models and presets, with titles) defaulting to `style`; `minConfidence` is optional and falls back to the chosen entry's own default. See the spec's Catalogue and preprocessing and Settings sections.

**Blocked by:** 01

**Status:** resolved

- [x] `anime_classification` model entry: `deepghs/anime_classification` @ `5ee62e06f5f4cd68a1c2f3bc5dc9805e827d37df`, `mobilenetv3_v1.3_dist` (provisional until the eval), with its golden test.
- [x] `style` preset: labels `anime`, `cg`, `live_action`; per frame, with `not_painting` dropped and the other four renormalised: `live_action = p_real`, `cg = (1 − p_real) · p_3d`, `anime = (1 − p_real) · (p_bangumi + p_illustration + p_comic)`; default `minConfidence` 0.6.
- [x] Raw models default to `minConfidence` 0 (never `uncertain` unless asked).
- [x] `tag` is the top label when it reaches `minConfidence`, else `uncertain`; low confidence is never a failure.
- [x] `models` in the result lists every model a preset used; `outputSchema`'s `tag` description lists every classifier's labels; `examples` show a `style` result.
- [x] Unit tests for the rule and the threshold; an end-to-end `ClassifyImage` Task with the default classifier.
