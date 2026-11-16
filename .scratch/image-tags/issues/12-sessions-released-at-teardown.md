# 12: Sessions and GPU memory released when a Worker ends

**What to build:** after the stuck Run of ticket 11, a PC restart fixed it, which points at GPU state an earlier
Worker left. Make sure a Worker that ends (Reload, a pool's idle-out, a Cancel that ends the process, the Host
stopping) drops its ONNX sessions, and find out whether anything keeps the process or its CUDA context alive.

**Blocked by:** 11

**Status:** resolved

- [x] **`Models.close()`:** drops every session (clear the dict under the session lock, then `gc.collect()`); a session
  asked for afterwards is made again.
- [x] **`teardown` on `_Classify`:** both Workers close their models; `ClassifyImage` still shuts its executors down.
  `ClassifyFrames` had no `teardown`.
- [x] **Download threads:** daemon threads; check nothing else (executors, the batchers) keeps the process from
  exiting once the shim calls `os._exit`.
- [x] **Tests:** `teardown` empties the sessions on both Workers; a second `teardown` is harmless.
- [x] **Verified by hand on Windows with the `cuda` build:** run a Task, Reload, and check with `nvidia-smi` and the
  process list that the old `python.exe` is gone and its memory freed. The comment by the pool config in `plugin.ts`
  ("frees the models' memory and the units") is true, or corrected.
- [x] **Anything left over by the Host or the shim** (fyflow-app) is written up to be raised there, not worked around.

## Changed while building it (2026-11-16)

- **The premise didn't hold:** the stuck Run reproduced on a fresh Host and in a standalone script with no earlier Worker (see ticket 11), so it wasn't GPU state left by an old Worker. `close()` and `teardown` are still in, so a Worker drops its sessions as soon as it is told to stop.
- **Nothing kept a process alive:** download threads and the shim's own threads are daemons. The executors' threads aren't, but both of the shim's exit paths end in `os._exit`. The batchers hold `Models`, not sessions.
- **Verified by hand on this machine** (RTX 3050 Ti Laptop, `cuda` build, baseline 873 MiB used by other apps). Worker `python.exe` processes were checked with the process list, and memory with `nvidia-smi --query-gpu=memory.used` (per-process memory reads N/A under WDDM):
  - **Reload right after a ClassifyImage Task:** the Worker was gone within 0.5 s and memory went back to 873 MiB.
  - **The pool's idle-out (5 s) with no Reload:** the Worker was gone between 3 and 7 s after the Task, and memory was freed.
  - **Cancel of a Task stuck in the GPU** (released 0.2.0): the Run was cancelled 10.3 s after Cancel (the Host's Python cancel grace), the Worker was gone, and memory was back to 873 MiB. A second stuck Run, resumed after a Host restart, ended the same way.
  - **Host stopped with a Worker stuck in the GPU:** the Worker went with it, and memory was freed.
- **`plugin.ts`'s comment** ("frees the models' memory and the units") holds; unchanged.
- **Nothing for fyflow-app:** no process or CUDA context outlived its Worker on any path.
