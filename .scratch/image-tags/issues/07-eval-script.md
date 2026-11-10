# 07: Eval script

**What to build:** a by-hand script that measures the classifiers on the user's own videos and settles the provisional defaults. Not part of `pnpm check`; run before a release is tagged. See the spec's Testing Decisions (Eval set).

**Blocked by:** 03, 04

**Status:** ready-for-agent

- [ ] Takes a CSV of `path,label` (labels anime, cg, live_action) and the `thumbs-collage` binary's path; no Host needed.
- [ ] Makes a sheet per video with the README's recommended settings and runs it through the same engine as `ClassifyFrames`.
- [ ] Prints a confusion matrix for `style`, and for the raw models against the labels they can express; the time per video; and the `minConfidence` that best separates the classes.
- [ ] Can run `anime_classification` v1.3 and v1.5 side by side to pick one.
- [ ] README section on how to run it and what it decides (`anime_classification` version, `style`'s default `minConfidence`, whether `cg` needs another model).
