# 04: ClassifyFrames on frame sheets and folders

**What to build:** a `ClassifyFrames` Node Type that classifies a whole video from a `VideoThumbnail` frame sheet, or a folder of frames, and hands on one aggregated answer plus each frame's own. A Flow `VideoThumbnail → ClassifyFrames → Edges on result.tag` works with no transform. See the spec's Input, Results and Running sections.

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] A JPEG whose COM segment holds the service's JSON with `frames` is a frame sheet: each non-null frame is cut at its `rect`; null frames are left out; `timestamp_actual` and `video` reach the result; no decodable frame fails with `The frame sheet has no frames:`.
- [ ] `config.render_timestamps` true adds a `warnings` entry.
- [ ] A folder: images directly inside it, sorted by name, `timestamp: null`, sheets read as plain images; empty fails with `The folder has no images:`, over 1000 with `The folder has more than 1000 images:`. Any other image is one frame.
- [ ] Result adds `agreement` and `frames: [{ timestamp, tag, scores }]`; `scores` is the mean over frames, `confidence` the tag's mean score.
- [ ] Frames run in batches of 32, `ctx.progress` advancing and Cancel checked between batches. Pool: `maxThreads: 1`, `maxConcurrentTasks: 2`.
- [ ] `ClassifyImage` given a frame sheet scores it as one image with a `warnings` entry pointing to `ClassifyFrames`.
- [ ] Fixture: a sheet made by the video-thumbnail Plugin from its `sample.mp4`; tests for frame count, rects, timestamps, the warning, a plain image, and folder order.
- [ ] README: the recommended `VideoThumbnail` settings (grid of 9 to 16, `fit: "stretch"`, spacing and margin 0, timestamps off, skipBlankFrames on, offsets past openings and credits).
