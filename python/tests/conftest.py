"""What the tests share: a stand-in for the Host's Task and context, and a local Hugging Face serving the pinned models.

The models are fetched once from huggingface.co at their pinned commits into `python/.cache/` (gitignored, checked
against the catalogue's sha256), then served from a local HTTP server that the Worker reaches through `HF_ENDPOINT`.
"""

import asyncio
import ctypes
import http.server
import inspect
import shutil
import threading
import time
from pathlib import Path

import pytest

from image_tags.catalogue import MODELS
from image_tags.models import DEFAULT_ENDPOINT, download, sha256_of

PYTHON = Path(__file__).resolve().parent.parent
CACHE = PYTHON / ".cache" / "hf"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
IMAGES = FIXTURES / "images"


class Task:
    """A Task as the Host's shim hands it to `run`."""

    def __init__(self, input, settings=None):
        self.input = input
        self.settings = {"profile": {}, **(settings or {})}


class Pool:
    """A Worker's pool as the shim builds it: the units it holds, by group (`grants`)."""

    def __init__(self, grants=None):
        self.grants = grants or {}


class Context:
    """A Task's context as the shim makes it, recording what `run` says."""

    def __init__(self, grants=None):
        self.grants = grants or {}
        self.cancelled = False
        self.progress_reports = []
        self.logs = []

    def progress(self, fraction, message=None):
        self.progress_reports.append((fraction, message))

    def log(self, message):
        self.logs.append(str(message))


def cached(model):
    """The model's file in the test cache, laid out as the hub serves it, fetched from huggingface.co the first time
    (by the Plugin's own downloader, so it is the pinned file)."""
    path = CACHE / model.remote_path
    if not path.exists() or sha256_of(path) != model.sha256:
        download(model, DEFAULT_ENDPOINT, path)
    return path


def served(hub, model):
    """Where `hub` serves `model` from, for a test to spoil or remove."""
    return hub.root / model.remote_path


class Hub:
    """A local Hugging Face: serves what is under `root` and counts the requests for each path.

    `hold()` makes it stall each file after its first `HEAD_BYTES`, as a slow network would, until `release()`: a test
    can then look at a download midway.
    """

    HEAD_BYTES = 256 * 1024

    def __init__(self, root):
        self.root = root
        self.requests = []
        self._flowing = threading.Event()
        self._flowing.set()
        hub = self

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(hub.root), **kwargs)

            def do_GET(self):
                hub.requests.append(self.path)
                path = Path(self.translate_path(self.path))
                if not path.is_file():
                    self.send_error(404, "Entry not found")
                    return
                body = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body[: hub.HEAD_BYTES])
                    self.wfile.flush()
                    hub._flowing.wait(60)
                    self.wfile.write(body[hub.HEAD_BYTES :])
                except ConnectionError:
                    pass  # the downloader hung up: what a stopped download does

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def hold(self):
        self._flowing.clear()

    def release(self):
        self._flowing.set()

    def close(self):
        self.release()
        self.server.shutdown()
        self.server.server_close()


_loop = None


def event_loop():
    """The event loop `async def run`s go on, as the shim's: one for the process, on a thread of its own."""
    global _loop
    if _loop is None:
        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, name="test-asyncio", daemon=True).start()
    return _loop


def run(worker, task, ctx=None, timeout=60):
    """What `worker.run` gives for `task`, called as the shim calls it: an `async def run` on the shared event loop, a
    plain one on a thread of its own."""
    return Running(worker, task, ctx).result(timeout)


class Running:
    """A Task running as the shim runs it: a plain `def run` on a thread of its own, an `async def run` on the shared
    event loop. `cancel()` does what the shim's cancel does: it marks the context cancelled, then raises
    `KeyboardInterrupt` inside the Task's thread, or cancels the Task's coroutine."""

    def __init__(self, worker, task, ctx=None):
        self.ctx = ctx or Context()
        self._outcome = None
        self.thread = self.future = None
        if inspect.iscoroutinefunction(worker.run):
            self.future = asyncio.run_coroutine_threadsafe(worker.run(task, self.ctx), event_loop())
        else:
            self.thread = threading.Thread(target=self._run, args=(worker, task), daemon=True)
            self.thread.start()

    def _run(self, worker, task):
        try:
            self._outcome = (worker.run(task, self.ctx), None)
        except BaseException as error:  # noqa: BLE001  (KeyboardInterrupt is how a cancel arrives)
            self._outcome = (None, error)

    def cancel(self):
        self.ctx.cancelled = True
        if self.future is not None:
            self.future.cancel()
            return
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(self.thread.ident), ctypes.py_object(KeyboardInterrupt)
        )

    def result(self, timeout=30):
        """What `run` returned, or what it raised, raised again; fails if the Task is still running after `timeout`."""
        if self.future is not None:
            try:
                return self.future.result(timeout)
            except TimeoutError:
                raise AssertionError(f"the Task is still running after {timeout} s") from None
        self.thread.join(timeout)
        assert not self.thread.is_alive(), f"the Task is still running after {timeout} s"
        result, error = self._outcome
        if error is not None:
            raise error
        return result


def wait_until(condition, timeout=10, what="the condition"):
    """Waits for `condition()` to be true, failing after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"{what} did not happen within {timeout} s"
        time.sleep(0.01)


@pytest.fixture(scope="session")
def models():
    return {model.id: cached(model) for model in MODELS.values()}


@pytest.fixture
def hub(models, tmp_path, monkeypatch):
    """A local hub serving copies of the pinned models (a test may spoil them), reached through `HF_ENDPOINT`."""
    root = tmp_path / "hub"
    shutil.copytree(CACHE, root)
    served = Hub(root)
    monkeypatch.setenv("HF_ENDPOINT", served.url)
    yield served
    served.close()


@pytest.fixture
def data_dir(tmp_path):
    path = tmp_path / "data"
    path.mkdir()
    return path


@pytest.fixture
def model_dir(models, tmp_path):
    """A folder of the catalogue's models laid out as `modelDir` holds them, as a person would fill it by hand."""
    root = tmp_path / "offline-models"
    for model in MODELS.values():
        (root / model.local_path).parent.mkdir(parents=True)
        shutil.copyfile(models[model.id], root / model.local_path)
    return root


@pytest.fixture
def config(data_dir):
    """The Worker's config as the Host gives it: start-up fields, and `host.dataDir` since the Plugin asks for one."""
    return {"host": {"dataDir": str(data_dir)}}
