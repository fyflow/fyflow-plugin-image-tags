# 11: Progress that doesn't lie: a finished download says so, and loading and the first batch are reported

**What to build:** a Run stuck (2026-11-16, Windows, RTX 3050 Ti Laptop 4 GB, `cuda` build) showed "Downloading the
model anime_real" at 0.896 for minutes after the download had ended and both CUDA sessions had loaded; the Worker was
busy in what came after, which says nothing. Make each step after the download say what it is doing.

**Blocked by:** none

**Status:** resolved

- [x] **`_Download.wait()`** reports 1.0 with its message when the download ends well (not on an error or a Cancel).
- [x] **`Models.session()`** reports "Loading the model <id> on <provider>" (the GPU provider chosen, or the CPU)
  before `ort.InferenceSession(...)`, through the Task's `ctx` when there is one. Only when a session is made, not on
  every call.
- [x] **`ClassifyFrames.run`** reports "Classifying 0 of N frames" before the first batch.
- [x] **`ClassifyImage`** reports what applies to one image (e.g. "Classifying" once its sessions are there), without
  breaking the shared batches.
- [x] **Tests** with the fake `ctx` the existing tests use: a Task whose model downloads reports 1.0 for the download,
  then the loading message, then the classifying message, in that order; a Task whose models are already loaded reports
  no loading message.

## Changed while building it (2026-11-16)

- **Order a fresh ClassifyFrames Task reports:** "Downloading the model <id>" up to 1.0; "Loading the model <id> on CUDA" (or DirectML, or the CPU); "Classifying 0 of N frames"; then "Classified n of N frames". The loading and the first classifying report are at 0.0, since the shim's `progress` takes a number, so the bar drops back from 1.0 when loading starts.
- **Sessions come first:** both Workers now get every model's session before the first batch, so a second model's download or loading is no longer hidden behind the first batch's message. ClassifyImage says "Classifying" from the Task's own coroutine, not from the batches it shares.
- **Reproduced on this machine** (RTX 3050 Ti Laptop 4 GB, driver 610.60, `cuda` build, onnxruntime-gpu 1.31.1) on a fresh Host with released 0.2.0: stuck at 0.90 "Downloading the model anime_real" with the model on disk and its CUDA session loaded. A standalone script in the Plugin's environment, with no Host and no earlier Worker, showed the cause is the GPU in that state: a forward pass took 0.2–0.8 s at batch 4–12 (the CPU does batch 16 in 0.07 s), and at batch 12–16 a pass sometimes never ended (GPU at 100%), with either cuDNN algorithm search. Batch 1 ran at 0.01 s. The GPU reported software power cap and thermal slowdown active, with another app holding it at 100%.
