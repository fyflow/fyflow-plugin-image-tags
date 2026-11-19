# 18: A list of frames, each with its timestamp

**What to build:** `ClassifyFrames` and `TagFrames` also take `frames: [{ path, timestamp? }]` instead of `path`. This
gives each frame its timestamp when the frames are image files rather than a frame sheet. Until now a folder's frames
and a single image's frame had `timestamp: null`, and nothing in the result said which file each frame was.

**Blocked by:** 14

**Status:** ready-for-human (built and tested; not yet run on a Host)

- [x] **Input:** exactly one of `path` (as now) or `frames`. The schema subset has no `oneOf`, so the schema leaves
  both optional and the Worker refuses neither or both.
  - `frames`: 1 to 1000 objects, read in the order given.
  - Each object's `path` is an absolute path to an image, checked against the Task's reach. A refused one fails the
    Task: the caller named it, unlike a folder's entry, which is skipped.
  - Each object's `timestamp` is optional: seconds into the video, a number of 0 or more, or null.
  - A frame sheet in the list is a plain image, as in a folder.
  - A path that is not a file or not an image fails the Task before any model loads, naming the frame by its index.
- [x] **Result:**
  - Each frame's `timestamp` is the one given, or null.
  - Each frame of a list or a folder gets its own `path`: as given for a list, and the folder joined with the file
    name for a folder.
  - A list input's result has no top-level `path`, so the output schemas no longer require `path`.
- [x] **Tests:** a list keeps its order, timestamps and paths; each validation failure; reach refusal; a folder's
  frames name their files; neither or both inputs fail.
- [x] **Thumbnails:** a JPEG of one frame made by VideoThumbnail, whose comment holds a JSON object with `timestamp`
  (and `video`), gives its frame that timestamp. This applies alone, in a folder or in a list, and a listed timestamp
  wins over it. A thumbnail's bad timestamp counts as no timestamp. The result's `video` is the one video the frames'
  thumbnails name, and is left out when they name none or several.

## Changed while building it (2026-11-19)

- **Schema:** the subset has no `oneOf` or `maxItems`. `frames` has `minItems: 1`; the 1000 limit is in its
  description and enforced by the Worker.
- **Breaking for wiring:** the outputs of `ClassifyFrames` and `TagFrames` no longer require `path`. A Flow that
  reads `outcome.result.path` from them still gets it for a `path` input.
- **Tests:** 362 pytest and 77 vitest pass.
- **Left:** run a list input and a folder of thumbnails on a Host. The ClassifyImage and TagImage results still do
  not say a thumbnail's timestamp.
