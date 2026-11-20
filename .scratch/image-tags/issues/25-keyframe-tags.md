# 25: Keyframes steered by tags (`keyframeTags`, `keyframeShare`)

**What to build:** two TagScenes settings that steer which frame is a scene's keyframe. Today it is always the most
typical frame (closest to the scene's mean embedding), often a plain wide shot or a frame mid-transition. With
`keyframeTags`, the keyframe is chosen among the scene's most typical frames by weighted tags, and falls back to the
most typical frame when none of them has a weighted tag. See `TagScenes-spec.md` §3.3 and §5.5 (Keyframe).

**Blocked by:** none

**Status:** ready-for-human

## Decided (2026-11-20)

- One map, tag → weight; negative weights avoid. Ratings are keyed `rating:<label>`.
- No `@category` keys (such as "any character") for now.
- Candidates are the most typical share of the scene, by default the top half, and the share is a setting
  (`keyframeShare`, 1 = every non-blank frame).
- Not in this ticket: skipping dark or blurry frames by default, more than one keyframe per scene, and a "wrong
  keyframe" mark on the review page.

## To do

- [x] `scenes.py`:
  - `Settings` gains `keyframe_tags: Mapping[str, float]` (default empty) and `keyframe_share: float` (0.5).
  - `_keyframe` as §5.5: typicality, the candidates, the preference from the scene's smoothed scores and ratings,
    ties to the more typical frame. Ignored tags weigh 0.
  - Unchanged result without `keyframeTags`, whatever `keyframeShare` is.
- [x] `worker.py` (`TagScenes._grouping`): read and check both settings. Weights must be finite numbers, keys
  non-empty strings, and the share in (0, 1]; anything else fails the Task before anything is read, naming the
  setting. Echo both in the result's `settings`. One warning lists the weighted tags the tagger lacks and those
  `ignoreTags` drops; the Task still runs.
- [x] `plugin.ts`: both settings on TagScenes, with descriptions. `keyframeTags` is an object with
  `additionalProperties: true`, since fyflow's schema subset takes no schema there, and the Worker checks values
  (as `thresholds` does). `keyframeShare` is a number with `exclusiveMinimum` 0 if the subset has it (else `minimum`
  0 and the Worker refuses 0) and `maximum` 1. Add the default to `SCENE_DEFAULTS`.
- [x] `python/scripts/review_scenes.py`: both in `SETTINGS`, so that `--set keyframeTags={"smile":1}` and
  `--set keyframeShare=1` work.
- [x] README: the TagScenes section names both settings, with an example.
- [x] Tests, all on synthetic frames (`test_scenes.py`) or the stand-in tagger (`test_tag_scenes.py`), none slow:
  - no `keyframeTags` gives today's keyframe, at any `keyframeShare`
  - a weighted tag picks a candidate other than the most typical
  - a frame with the tag outside the top half does not win at 0.5, and wins at 1
  - a negative weight steers away; a `rating:<label>` key works
  - no candidate with a weighted tag gives the most typical frame
  - a tag in `ignoreTags` weighs 0
  - an all-blank scene still gives the middle frame
  - the Worker: the warning for unknown and ignored tags; bad weights and shares fail; both echoed in `settings`
  - `plugin.test.ts`: the schema has both settings, and `SCENE_DEFAULTS` the share
- [x] `pnpm check` passes (454 pytest, 87 vitest). Not run on real models: that waits for the user's go-ahead.

## Changed while building it (2026-11-20)

- **`keyframeShare` in the schema:** fyflow's subset has no `exclusiveMinimum`, so the schema says `minimum: 0` and
  the Worker refuses 0.
- **The echo:** both settings go into the settings a reused file is keyed by (the Task's `settings`, as the other
  grouping settings), so a changed `keyframeTags` makes a new file. TagScenes' result itself carries no settings.
- **A rating's own name** (`explicit`) is no tag: only `rating:explicit` weighs it, and the bare name is left out with
  the warning.
- **Smoothing** spreads a weighted tag a little onto nearby frames, so with the tagged frame outside the typical
  share, a typical neighbour of it may win over the most typical frame.
- **To try it on real videos:** `review_scenes.py … --set keyframeTags={"smile":1} --set keyframeShare=1`. The
  keyframe has the blue border on the page.
