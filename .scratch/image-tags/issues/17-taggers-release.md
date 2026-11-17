# 17: Release with the taggers

**What to build:** the release that ships `TagImage`, `TagFrames` and the three taggers. It follows the usual release
flow: version bump, `uv lock`, `pnpm check`, tag, then the index PR after the Release workflow.

**Blocked by:** 14, 16

**Status:** ready-for-human

- [ ] Version 0.3.0 in `package.json`, `plugin.ts` and `python/pyproject.toml`, then `uv lock`.
- [ ] The README says what is new.
- [ ] Tag, then open the index PR to fyflow-plugins once the asset exists, with the user's approval.
