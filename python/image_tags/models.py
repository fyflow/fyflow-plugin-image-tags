"""Getting the catalogue's models: downloaded on first use, checked, and kept as inference sessions.

A model is fetched from its pinned commit over plain HTTPS into a temp file beside its place in the data directory,
checked against its sha256, then renamed into `<dataDir>/models/<repo>/<commit, 12 characters>/<name>/`. So a
half-written or unpinned file is never loaded, and two processes fetching at once cannot spoil each other's.

Within one process, the Tasks that need a model at once share one download, run on a thread of its own: a cancelled
Task stops waiting at once without tearing the download from under the others (a `ClassifyFrames` Task gets the
shim's `KeyboardInterrupt` in its thread; a `ClassifyImage` Task's coroutine is cancelled, and the thread it waits on
sees `ctx.cancelled` within `POLL_SECONDS`), and the download stops, its temp file gone, once no Task waits on it any
more.

With a model folder (the Profile's `modelDir`), for a machine without internet, nothing is ever downloaded: the models
must be there already, in the same layout.
"""

import hashlib
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import devices
from .catalogue import Model

#: Where models come from: Hugging Face, or a mirror of it named the way `huggingface_hub` reads it.
DEFAULT_ENDPOINT = "https://huggingface.co"
#: Small, so that a download no Task waits on any more stops soon after.
CHUNK_BYTES = 64 * 1024
TIMEOUT_SECONDS = 60
#: How often a Task waiting on a download reports its progress and looks for its own Cancel.
POLL_SECONDS = 0.1


class ModelUnavailable(Exception):
    """A model could not be had; it fails the Task that wanted it, and the next one tries again."""

    def __init__(self, model: Model, reason: str):
        super().__init__(f"Could not get the model {model.id}: {reason}")
        self.reason = reason


class Cancelled(Exception):
    """The Task was cancelled while it waited for a model (the shim reports it as cancelled, whatever it says)."""


class _Abandoned(Exception):
    """No Task waits on the download any more."""


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    model: Model, endpoint: str, path: Path, on_chunk: Callable[[int, int | None], None] | None = None
) -> None:
    """Fetches `model` from `endpoint` to `path`, which appears only once the whole file matches the pinned sha256.

    `on_chunk(received, total)` is called after each chunk with the bytes so far and the file's size (`None` when the
    server does not say); what it raises stops the download, and leaves no file.
    """
    url = f"{endpoint}/{model.remote_path}"
    partial = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, partial = tempfile.mkstemp(dir=path.parent, prefix=f".{model.file}.", suffix=".part")
        digest = hashlib.sha256()
        with os.fdopen(handle, "wb") as file, urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            length = response.headers.get("Content-Length", "")
            total = int(length) if length.isdigit() else None
            received = 0
            while chunk := response.read(CHUNK_BYTES):
                digest.update(chunk)
                file.write(chunk)
                received += len(chunk)
                if on_chunk is not None:
                    on_chunk(received, total)
        if digest.hexdigest() != model.sha256:
            raise ModelUnavailable(model, f"{url} is not the pinned file (its sha256 is {digest.hexdigest()})")
        os.replace(partial, path)
    except ModelUnavailable:
        raise
    except urllib.error.HTTPError as error:
        raise ModelUnavailable(model, f"{url} answered {error.code} {error.reason}") from None
    except urllib.error.URLError as error:
        raise ModelUnavailable(model, f"{url}: {error.reason}") from None
    except OSError as error:
        # Writing it: a folder that cannot be written, or a path too long for Windows (260 characters unless long
        # paths are enabled); or a connection that breaks midway.
        raise ModelUnavailable(model, f"cannot download {url} to {path}: {error}") from None
    finally:
        if partial is not None and os.path.exists(partial):
            os.unlink(partial)


class _Waiter:
    """A Task waiting on a download; it counts until its Task is cancelled (`ctx` is `None` for a caller that cannot
    be)."""

    def __init__(self, ctx):
        self.ctx = ctx

    @property
    def cancelled(self) -> bool:
        return bool(getattr(self.ctx, "cancelled", False))


class _Download:
    """One model's download in this process, on a thread of its own, and the Tasks waiting on it.

    The download goes on while any of them is not cancelled. They are kept as a set rather than a count each takes back
    as it leaves, because a Cancel's `KeyboardInterrupt` may land anywhere in a Task's thread, even in the `finally`
    that would take it back.
    """

    def __init__(self, model: Model, endpoint: str, path: Path, lock: threading.Lock, on_end: Callable[[], None]):
        self.model = model
        self._endpoint = endpoint
        self._path = path
        # The Models' lock: whether a Task joins a download, and whether the download gives up, are decided under it.
        self._lock = lock
        self._on_end = on_end
        self._waiters: set[_Waiter] = set()
        self._received = 0
        self._total: int | None = None
        self._error: ModelUnavailable | None = None
        self._done = threading.Event()
        #: No Task waits on it any more: it is stopping, and no Task may join it.
        self.abandoned = False

    def join(self, ctx) -> _Waiter:
        """Counts a Task in. Under the lock."""
        waiter = _Waiter(ctx)
        self._waiters.add(waiter)
        return waiter

    def start(self) -> None:
        threading.Thread(target=self._fetch, name=f"image-tags-download-{self.model.id}", daemon=True).start()

    def _fetch(self) -> None:
        try:
            download(self.model, self._endpoint, self._path, self._on_chunk)
        except ModelUnavailable as error:
            self._error = error
        except _Abandoned:
            self._error = ModelUnavailable(self.model, "its download was cancelled")
        except BaseException as error:  # noqa: BLE001  (no Task waiting on it may wait for ever)
            self._error = ModelUnavailable(self.model, f"{type(error).__name__}: {error}")
        finally:
            with self._lock:
                self._on_end()
            self._done.set()

    def _on_chunk(self, received: int, total: int | None) -> None:
        with self._lock:
            self._received, self._total = received, total
            if all(waiter.cancelled for waiter in self._waiters):
                self.abandoned = True
                raise _Abandoned

    def wait(self, waiter: _Waiter) -> None:
        """Waits for the download to end, reporting its progress through the Task. Raises `Cancelled` as soon as the
        Task is cancelled, and `ModelUnavailable` when the download failed: the same for every Task that waited."""
        ctx = waiter.ctx
        message = f"Downloading the model {self.model.id}"
        reported = None
        try:
            while True:
                received, total = progress = (self._received, self._total)
                if ctx is not None and total and progress != reported:
                    ctx.progress(min(received / total, 1.0), message)
                    reported = progress
                if self._done.wait(POLL_SECONDS):
                    break
                if waiter.cancelled:
                    raise Cancelled(f"Cancelled while downloading the model {self.model.id}")
        finally:
            with self._lock:
                self._waiters.discard(waiter)
        if self._error is not None:
            raise ModelUnavailable(self.model, self._error.reason)


class Models:
    """The models of one Worker instance: their files under `data_dir` (or in `model_dir`, which is never downloaded
    into), and one session each once loaded, running on GPU `device` when the installed ONNX Runtime build has a GPU
    provider (`devices.py`), else on the CPU, on `threads` threads."""

    def __init__(
        self,
        data_dir: str,
        endpoint: str | None = None,
        model_dir: str | None = None,
        threads: int = 0,
        device: int = 0,
    ):
        #: The threads a session spreads a forward pass over (onnxruntime's `intra_op_num_threads`; 0, its default:
        #: one per processor).
        self.threads = threads
        #: The GPU the sessions run on, when the build has a GPU provider: the `gpu` unit the process holds.
        self.device = device
        self.model_dir = Path(model_dir) if model_dir else None
        self.root = self.model_dir or Path(data_dir) / "models"
        self.endpoint = (endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
        self._sessions: dict[str, ort.InferenceSession] = {}
        self._session_lock = threading.Lock()
        # One forward pass at a time in the process: the DirectML provider takes no concurrent `run` on a session, and
        # on the CPU each pass already spreads over the threads of the units the process holds.
        self._run_lock = threading.Lock()
        self._lock = threading.Lock()
        self._downloads: dict[str, _Download] = {}

    def path_of(self, model: Model) -> Path:
        return self.root / model.local_path

    def session(self, model: Model, ctx=None) -> ort.InferenceSession:
        """The model's inference session, made on first use, once the model's file is there: downloaded if need be,
        with progress and Cancel through the Task's `ctx`."""
        session = self._sessions.get(model.id)
        if session is not None:
            return session
        path = self._file(model, ctx)
        with self._session_lock:
            session = self._sessions.get(model.id)
            if session is None:
                chosen = devices.providers(ort.get_available_providers(), self.device)
                devices.prepare(chosen)
                session = ort.InferenceSession(
                    str(path),
                    devices.session_options(chosen, self.threads),
                    providers=[name for name, _ in chosen],
                    provider_options=[options for _, options in chosen],
                )
                self._sessions[model.id] = session
                _say_where(model, session, chosen, self.device)
            return session

    def scores(self, model: Model, batch: np.ndarray, ctx=None) -> np.ndarray:
        """Each image's label probabilities, `[n, len(labels)]`: the models end in a softmax."""
        session = self.session(model, ctx)
        with self._run_lock:
            (output,) = session.run(["output"], {"input": batch})
        return output

    def _file(self, model: Model, ctx) -> Path:
        path = self.path_of(model)
        if self.model_dir is not None:
            if not path.is_file():
                raise ModelUnavailable(model, f"it is not in the model folder {self.model_dir}: expected {path}")
            # Copied there by hand, so checked as a download is: nothing unpinned ever loads.
            if (digest := sha256_of(path)) != model.sha256:
                raise ModelUnavailable(model, f"{path} is not the pinned file (its sha256 is {digest})")
            return path
        with self._lock:
            if path.is_file():
                return path
            fetching = self._downloads.get(model.id)
            if fetching is None or fetching.abandoned:
                fetching = _Download(model, self.endpoint, path, self._lock, lambda: self._ended(model, fetching))
                self._downloads[model.id] = fetching
                waiter = fetching.join(ctx)
                fetching.start()
            else:
                waiter = fetching.join(ctx)
        fetching.wait(waiter)
        return path

    def _ended(self, model: Model, fetching: _Download) -> None:
        # Under the lock. An abandoned download may end after a new one has taken its place.
        if self._downloads.get(model.id) is fetching:
            del self._downloads[model.id]


def _say_where(model: Model, session: ort.InferenceSession, chosen: devices.Providers, device: int) -> None:
    """A line on stderr (the Host's log, and the Plugins page's "Python output") naming the provider the session runs
    on, and saying so when it is not the GPU provider the build offered (one that could not start: onnxruntime falls
    back to the CPU)."""
    running = session.get_providers()[0]
    line = f"image-tags: the model {model.id} runs on {running}"
    if running != devices.CPU:
        line += f" (gpu unit {device}: {', '.join(f'{key}={value}' for key, value in chosen[0][1].items())})"
    wanted = chosen[0][0]
    if running != wanted:
        line += f", not {wanted}: it could not start"
    line += f"; onnxruntime {ort.__version__}, {session.get_session_options().intra_op_num_threads} threads"
    print(line, file=sys.stderr, flush=True)
