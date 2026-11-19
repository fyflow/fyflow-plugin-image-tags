# 23: The TagScenes Node Type

**What to build:** `TagScenes`: reads frames as TagFrames does, tags and embeds them in batches, groups and pools them
with `scenes.py`, and hands on scenes and a summary, or writes them to a file. See `TagScenes-spec.md` §3, §4 and §6.

**Blocked by:** 21, 22

**Status:** ready-for-agent

- [ ] `frames.py`:
  - blank detection (`BLANK_STD`, `ImageStat` on a 64-px thumbnail) at batch load time
  - `duration` from thumbnail comments onto `Frames` when the frames agree on one
  - ClassifyFrames' and TagFrames' output unchanged
- [ ] `results.py`: `summary` / `reused` take the list fields to move. `count` comes from the caller. ClassifyFrames
  and TagFrames behave as before.
- [ ] `worker.py`: `class TagScenes(_Tag)`.
  - The input is `path` or `frames`.
  - Every frame needs a timestamp; the Task fails before any model if not, and the message says how to get them.
  - Tag and embed via `scores_and_embeddings`, keeping only §4.4. Progress 0 → 0.9, then grouping to 1.0. Cancel is
    checked between batches and stages.
  - The result follows §3.4. `includeFrames` adds `frames` with `blank` and `scene`.
- [ ] `plugin.ts`:
  - the worker and Node Type (pool as TagFrames)
  - settings §3.3, with `fileSettingsProperties`
  - output schema with `examples`, reusing `TAGGED_FRAME`; nothing required that `lists: "file"` removes
- [ ] README:
  - the Node Type and the Flows of §2
  - the settings
  - "sample about every 2 s, and give DetectScenes a `maxFrames` above the frame count"
- [ ] **Tests:**
  - pytest per §7.2: the worker on the stand-in, and the `slow` planted video from the fixture images
  - vitest for the Node Type, settings and schemas
  - `pnpm check` passes
- [ ] On a Host: §7.4.
