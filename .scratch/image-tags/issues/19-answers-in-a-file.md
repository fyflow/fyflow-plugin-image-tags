# 19: Frames' answers in a file, and up to 10000 frames

**What to build:** `ClassifyFrames` and `TagFrames` can write their per-frame answers to a JSON file, so a large one
does not sit in the Flow. This follows the pattern of VideoThumbnail's `ExtractFrames`: `lists`, `output`,
`overwrite`, the Profile's `outputDir`, and the Plugin's own folder, with a file made before reused. A Task may now
take up to 10000 frames.

**Blocked by:** 18

**Status:** ready-for-human (built and tested; not yet run on a Host)

- [x] **Limit:** 10000 images in a folder, and 10000 entries in a `frames` list (`MAX_FOLDER_IMAGES`).
- [x] **Settings** on both frames Node Types:
  - `lists`: `result` (the default, as before) or `file`.
  - `output`: a `.json` file, written every time, or a folder.
  - `overwrite`.
- [x] **Where the file goes:** `output`, else the Profile's `outputDir` (a per-Task field), else
  `<dataDir>/results/`. A path given needs `write` access in the Allowed roots, checked before any model runs; the
  Plugin's own folder needs none.
- [x] **The file** holds the whole result, `frames` included. It is written to `<file>.<pid>.part` and renamed into
  place.
- [x] **The Task's result** with `file`: everything but `frames`, plus `file`, `count` and `reused`. The output
  schemas no longer require `frames`.
- [x] **Reuse:**
  - A file is named `<video, sheet, folder or a list's first frame's folder>.<key>.json`.
  - The key is the first 16 hex digits of a sha256 over the Node Type, the input, each file read (real path, size,
    mtime_ns), the effective settings (merged thresholds included), the models' repo, commit and file, and a
    `FORMAT` number.
  - A matching file is handed on with `reused: true`, and no model loads. `overwrite` runs the models again.
- [x] **Too large to hand on:** with `lists: "result"`, a result over 15 MiB fails the Task saying to set `lists` to
  `file`, rather than the Host refusing it at 16 MiB.
- [x] **Data link:** the `results/` folder is an item once it exists.
- [x] **Docs:** the README's "Answers in a file" section, Allowed roots and Data.
- [x] **Tests:** `test_results.py` (15 tests) and vitest for the settings, the Profile field and the Data item.
  377 pytest and 78 vitest pass.

## Notes

- **Settings only:** `lists`, `output` and `overwrite` are Node settings only, not Task input fields as in
  video-thumbnails. A `map` over videos writes each into one folder, named per video.
- **Key cost:** for a folder or list, the key stats every file, which is cheap next to running a model even at 10000
  files. It does not hash the files' contents: a file rewritten with the same size within the same mtime tick would be
  missed.
- **Left:** a Host run with `lists: "file"` on a large folder, checking the reuse, the time and the Data link.
