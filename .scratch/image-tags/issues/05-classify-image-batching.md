# 05: Batching ClassifyImage across Tasks

**What to build:** a Flow that maps `ClassifyImage` over many images runs them as shared batches in one process instead of one forward pass per Task. See the spec's Running section.

**Blocked by:** 01

**Status:** resolved

- [x] `async def run`; each Task decodes and preprocesses with `asyncio.to_thread`, then hands its tensor to a per-model batcher.
- [x] A batch runs at 16 frames or 20 ms after the first arrived, whichever comes first, off the event loop. Constants, not settings.
- [x] Pool: `maxThreads: 1`, `maxConcurrentTasks: 32`.
- [x] Each Task awaits its result through `asyncio.shield`: a cancelled Task's queued frame is dropped, one in a running batch has its result discarded, and other Tasks' results are intact.
- [x] Tests: flush at size, flush at the window, cancel leaves the batch intact, results map back to the right Tasks.

## Changed while building it (2026-11-10)

- Decoding and model loading run on the Worker's own thread pool (64 threads), not `asyncio.to_thread`: the loop's default pool starved Tasks waiting on downloads.
- The batch size is the Profile start-up field `batchSize` (default 16), not a constant: on a GPU the best size depends on its memory. `ClassifyFrames` uses it for its chunks too (was 32). The 20 ms window stays a constant.
- Resources are held per process, not per Task: both Node Types take no per-Task `groups` and hold `residentGroups: { gpu: 1, cpu: 4 }` with `idleTimeout: 5_000` (was 15 s: the two Node Types take turns on the one `gpu` unit, and the wait is the idle time plus 5 s). onnxruntime's threads come from `pool.holds["cpu"]`, else 4; a Profile cannot yet change the resident counts, see `../fyflow-app-request-resident-units.md`.
