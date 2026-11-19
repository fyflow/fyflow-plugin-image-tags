# 20: Release with frame lists and answers in a file

**What to build:** the release that ships tickets 18 and 19. It follows the usual release flow: version bump, `uv lock`,
`pnpm check`, tag, then the index PR after the Release workflow.

**Blocked by:** 18, 19

**Status:** ready-for-human (bumped and checked; tag, push and index PR wait for the user)

- [x] Version 0.4.0 in `package.json`, `plugin.ts` and `python/pyproject.toml`, then `uv lock`. It is a minor version
  because the frames output schemas no longer require `frames` (or `path`).
- [x] `pnpm check` passes, with 377 pytest (slow included) and 78 vitest.
- [ ] Push `main` and the tag `fyflow-ext.image-tags@0.4.0`. The Release workflow then builds
  `fyflow-plugin-image-tags-0.4.0.tgz`.
- [ ] Open the index PR to fyflow-plugins once the asset exists, with the user's approval. The SDK range and Node Types
  are unchanged from 0.3.0.

## What is new in 0.4.0

- **`frames` input** on `ClassifyFrames` and `TagFrames`: a list of `{path, timestamp?}`, as an alternative to `path`.
- **Thumbnail timestamps:** the timestamp and video come from a VideoThumbnail JPEG's comment, whether it is read alone,
  in a folder or in a list.
- **Answers in a file:** `lists: "file"`, `output`, `overwrite` and the Profile's `outputDir`. A file made before is
  reused, and the `results/` folder is shown in Data.
- **Limit:** 10000 frames per Task, up from 1000. A result over 15 MiB fails, telling the user to set `lists` to "file".
- Tickets 18 and 19 still need a run on a Host.
