# 05: Batching ClassifyImage across Tasks

**What to build:** a Flow that maps `ClassifyImage` over many images runs them as shared batches in one process instead of one forward pass per Task. See the spec's Running section.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] `async def run`; each Task decodes and preprocesses with `asyncio.to_thread`, then hands its tensor to a per-model batcher.
- [ ] A batch runs at 16 frames or 20 ms after the first arrived, whichever comes first, off the event loop. Constants, not settings.
- [ ] Pool: `maxThreads: 1`, `maxConcurrentTasks: 32`.
- [ ] Each Task awaits its result through `asyncio.shield`: a cancelled Task's queued frame is dropped, one in a running batch has its result discarded, and other Tasks' results are intact.
- [ ] Tests: flush at size, flush at the window, cancel leaves the batch intact, results map back to the right Tasks.
