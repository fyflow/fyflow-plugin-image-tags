"""One model's forward passes shared by the Tasks of a process: each hands in its frame and awaits its own row.

A batch runs once it holds `size` frames (the Profile's `batchSize`), or `WINDOW_SECONDS` after its first frame arrived, whichever comes
first; on the batcher's executor, never on the event loop. One batch of a model runs at a time: frames arriving
meanwhile wait for the next, which runs as soon as the forward pass is free if they have waited their window by then.

A Task awaits its row through `asyncio.shield`, so its Cancel never reaches the batch: a frame still waiting is taken
out, and the row of one in a running batch is dropped. The other Tasks' rows are untouched.
"""

import asyncio
from concurrent.futures import Executor

import numpy as np

from .catalogue import Model
from .models import Models

#: How long a batch waits for more frames after its first one arrived.
WINDOW_SECONDS = 0.02


class _Frame:
    __slots__ = ("input", "arrived", "row")

    def __init__(self, input: np.ndarray, arrived: float, row: asyncio.Future):
        self.input = input
        self.arrived = arrived
        self.row = row


class Batcher:
    """Batches `model`'s frames for `models.scores`, at most `size` to a batch, run on `executor`. Used from one event
    loop at a time."""

    def __init__(self, model: Model, models: Models, executor: Executor, size: int):
        self.model = model
        self.size = size
        self._models = models
        self._executor = executor
        self._waiting: list[_Frame] = []
        self._timer: asyncio.TimerHandle | None = None
        self._running: asyncio.Task | None = None

    async def score(self, input: np.ndarray) -> np.ndarray:
        """`input`'s scores, `[len(labels)]`, from the next batch it goes in. What the batch raises, it raises."""
        loop = asyncio.get_running_loop()
        frame = _Frame(input, loop.time(), loop.create_future())
        self._waiting.append(frame)
        self._next()
        try:
            return await asyncio.shield(frame.row)
        except asyncio.CancelledError:
            if frame in self._waiting:
                self._waiting.remove(frame)
                self._next()
            frame.row.cancel()  # its row, should the batch be running, is dropped
            raise

    def _next(self) -> None:
        """Starts the next batch if one is due and none is running, else sets the timer for when one will be due."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._running is not None or not self._waiting:
            return
        loop = asyncio.get_running_loop()
        due_in = self._waiting[0].arrived + WINDOW_SECONDS - loop.time()
        if len(self._waiting) < self.size and due_in > 0:
            self._timer = loop.call_later(due_in, self._next)
            return
        batch, self._waiting = self._waiting[: self.size], self._waiting[self.size :]
        self._running = loop.create_task(self._run(batch))

    async def _run(self, batch: list[_Frame]) -> None:
        try:
            inputs = np.stack([frame.input for frame in batch])
            rows = await asyncio.get_running_loop().run_in_executor(
                self._executor, self._models.scores, self.model, inputs
            )
        except Exception as error:  # noqa: BLE001  (each Task waiting fails with it; the batcher goes on)
            for frame in batch:
                if not frame.row.done():
                    frame.row.set_exception(error)
        else:
            for frame, row in zip(batch, rows, strict=True):
                if not frame.row.done():
                    frame.row.set_result(row)
        finally:
            self._running = None
            self._next()
