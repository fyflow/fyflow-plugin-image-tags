# 02: Hardened model fetching

**What to build:** model downloads that behave well under real use. Tasks in one process that need the same model at once share one download; it reports progress through the Task and stops on Cancel. A machine without internet uses the Profile start-up field `modelDir` instead of downloading. The Plugin's status says which models are present or still to download. See the spec's Models section.

**Blocked by:** 01

**Status:** resolved

- [x] One download per model per process, shared by concurrent Tasks; progress via `ctx.progress`; Cancel stops it and leaves no file.
- [x] A sha256 mismatch fails that Task only and leaves no file; the next Task tries again.
- [x] Temp file plus rename into `<dataDir>/models/<repo>/<commit>/<name>/`, so processes of different pools or Profiles cannot collide.
- [x] `modelDir` (start-up field): nothing is downloaded; a missing model fails the Task, naming the folder.
- [x] `status()`: `ok` with the models present, or `info` "Models download on first use" listing the missing ones.
- [x] Tests against a local HTTP server: one download for concurrent Tasks, progress reported, Cancel stops it, a mismatch leaves no file, `modelDir` downloads nothing.
