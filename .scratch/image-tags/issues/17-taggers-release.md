# 17: Release with the taggers

**What to build:** the release that ships `TagImage`, `TagFrames` and the three taggers. It follows the usual release
flow: version bump, `uv lock`, `pnpm check`, tag, then the index PR after the Release workflow.

**Blocked by:** 14, 16

**Status:** resolved

- [x] Version 0.3.0 in `package.json`, `plugin.ts` and `python/pyproject.toml`, then `uv lock`.
- [x] The README says what is new.
- [x] Tag, then open the index PR to fyflow-plugins once the asset exists, with the user's approval.

## Changed while building it (2026-11-17)

- **README:** no change was needed. Its intro, Node Types and Taggers sections already describe the taggers, and it
  has no changelog.
- **Checks:** `pnpm check` passed, with 336 pytest (slow included).
- **Release:** tag `fyflow-ext.image-tags@0.3.0`; the Release workflow built `fyflow-plugin-image-tags-0.3.0.tgz`.
- **Index PR:** fyflow/fyflow-plugins#11 adds 0.3.0 with SDK range `^0.11.0`, the same three Python variants and
  Node Types `ClassifyImage`, `ClassifyFrames`, `TagImage` and `TagFrames`. The Plugin's index description now says
  it tags as well as classifies. `check-pr.ts` passed locally.
