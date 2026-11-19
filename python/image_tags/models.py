"""Getting the catalogue's models: downloaded on first use, checked, and kept as inference sessions; and a tagger's tag
list, downloaded and checked the same way.

A model (or tag list) is fetched from its pinned commit over plain HTTPS into a temp file beside its place in the data
directory, checked against its sha256, then renamed into `<dataDir>/models/<repo>/<commit, 12 characters>/[<name>/]`.
So a half-written or unpinned file is never loaded, and two processes fetching at once cannot spoil each other's.

Within one process, the Tasks that need a model at once share one download, run on a thread of its own: a cancelled
Task stops waiting at once without tearing the download from under the others (a `ClassifyFrames` Task gets the
shim's `KeyboardInterrupt` in its thread; a `ClassifyImage` Task's coroutine is cancelled, and the thread it waits on
sees `ctx.cancelled` within `POLL_SECONDS`), and the download stops, its temp file gone, once no Task waits on it any
more.

With a model folder (the Profile's `modelDir`), for a machine without internet, nothing is ever downloaded: the models
must be there already, in the same layout.

A tagger has a second session too, for what needs each image's embedding beside its scores: the same graph with a
second output, `embedding`, made in memory from the model's file the first time it is asked for.
"""

import gc
import hashlib
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from . import devices
from .catalogue import Labels, Model, Pinned, Tagger, labels_from_csv

#: Where models come from: Hugging Face, or a mirror of it named the way `huggingface_hub` reads it.
DEFAULT_ENDPOINT = "https://huggingface.co"
#: Small, so that a download no Task waits on any more stops soon after.
CHUNK_BYTES = 64 * 1024
TIMEOUT_SECONDS = 60
#: How often a Task waiting on a download reports its progress and looks for its own Cancel.
POLL_SECONDS = 0.1
#: The output a tagger's embedding session adds: each image's embedding, `[batch, d]` float32.
EMBEDDING = "embedding"
#: What may lie between a tagger's head and its output: a bias, a Sigmoid, a cast to float32.
_AFTER_THE_HEAD = ("Sigmoid", "Add", "Cast")
_HEADS = ("Gemm", "MatMul")


class ModelUnavailable(Exception):
    """A model (or tag list) could not be had; it fails the Task that wanted it, and the next one tries again."""

    def __init__(self, model: Pinned, reason: str):
        super().__init__(f"Could not get {model.subject}: {reason}")
        self.reason = reason


class OutOfGpuMemory(Exception):
    """A model's session, or a forward pass, needed more GPU memory than there was: the Task fails, saying what to do.
    It never falls back to the CPU."""

    def __init__(self, model: Model | Tagger):
        super().__init__(f"{model.id} needs more GPU memory than this GPU has: {model.out_of_gpu_memory}")


class GraphChanged(Exception):
    """A tagger's graph is not what its catalogue entry says of it: its embedding cannot be found."""

    def __init__(self, tagger: Tagger, reason: str):
        super().__init__(
            f"The graph of the tagger {tagger.id} changed: {reason}. Its catalogue entry's `head_node` must name the "
            "Gemm (or MatMul) that gives its scores, its data input `[batch, d]`"
        )


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
    model: Pinned, endpoint: str, path: Path, on_chunk: Callable[[int, int | None], None] | None = None
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

    def __init__(self, model: Pinned, endpoint: str, path: Path, lock: threading.Lock, on_end: Callable[[], None]):
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
        name = f"image-tags-download-{self.model.id}-{self.model.file}"
        threading.Thread(target=self._fetch, name=name, daemon=True).start()

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
        message = f"Downloading {self.model.subject}"
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
                    raise Cancelled(f"Cancelled while downloading {self.model.subject}")
        finally:
            with self._lock:
                self._waiters.discard(waiter)
        if self._error is not None:
            raise ModelUnavailable(self.model, self._error.reason)
        if ctx is not None:
            # Whole, so the Task's progress does not stay at the last chunk while it goes on to what comes next.
            ctx.progress(1.0, message)


class Models:
    """The models of one Worker instance: their files under `data_dir` (or in `model_dir`, which is never downloaded
    into), and one session each once loaded, running on GPU `device` when the installed ONNX Runtime build has a GPU
    provider (`devices.py`), else on the CPU, on `threads` threads. `variant` is the build the Host synced, for the
    line each session says on stderr."""

    def __init__(
        self,
        data_dir: str,
        endpoint: str | None = None,
        model_dir: str | None = None,
        threads: int = 0,
        device: int = 0,
        variant: str | None = None,
    ):
        #: The threads a session spreads a forward pass over (onnxruntime's `intra_op_num_threads`; 0, its default:
        #: one per processor).
        self.threads = threads
        #: The GPU the sessions run on, when the build has a GPU provider: the `gpu` unit the process holds.
        self.device = device
        #: The build of ONNX Runtime the Host synced (`host.variant`); `None` from a Host before SDK 0.11.
        self.variant = variant
        self.model_dir = Path(model_dir) if model_dir else None
        self.root = self.model_dir or Path(data_dir) / "models"
        self.endpoint = (endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
        self._sessions: dict[str, ort.InferenceSession] = {}
        #: The taggers' sessions with their `embedding` output, kept apart from the plain ones.
        self._embedding_sessions: dict[str, ort.InferenceSession] = {}
        self._labels: dict[str, Labels] = {}
        self._session_lock = threading.Lock()
        # One forward pass at a time in the process: the DirectML provider takes no concurrent `run` on a session, and
        # on the CPU each pass already spreads over the threads of the units the process holds.
        self._run_lock = threading.Lock()
        self._lock = threading.Lock()
        # By the file's place, which tells a tagger's model and its tag list apart.
        self._downloads: dict[str, _Download] = {}

    def path_of(self, model: Pinned) -> Path:
        return self.root / model.local_path

    def labels(self, tagger: Tagger, ctx=None) -> Labels:
        """The tagger's tag list, read once its file is there: downloaded if need be, as `session` downloads a
        model."""
        labels = self._labels.get(tagger.id)
        if labels is None:
            path = self._file(tagger.tag_list, ctx)
            labels = self._labels[tagger.id] = labels_from_csv(path.read_text(encoding="utf-8"))
        return labels

    def runs_on_cpu(self, model: Model | Tagger, embedding: bool = False) -> bool:
        """Whether the model's session (its embedding session, with `embedding`), once made, runs on the CPU: no GPU
        provider, or one that could not start."""
        return self.session(model, embedding=embedding).get_providers()[0] == devices.CPU

    def session(self, model: Model | Tagger, ctx=None, embedding: bool = False) -> ort.InferenceSession:
        """The model's inference session, made on first use, once the model's file is there: downloaded if need be,
        with progress and Cancel through the Task's `ctx` (`None` for a caller without a Task), which is also told when
        the session is being made, and where.

        With `embedding`, a tagger's other session, kept beside the plain one: its graph has a second output,
        `embedding`, each image's `[batch, d]` float32 embedding (see `with_embedding`). It is made in memory, and
        nothing is written to disk."""
        if embedding and not isinstance(model, Tagger):
            raise ValueError(f"{model.id} is not a tagger: only a tagger's session gives an embedding")
        sessions = self._embedding_sessions if embedding else self._sessions
        session = sessions.get(model.id)
        if session is not None:
            return session
        path = self._file(model, ctx)
        with self._session_lock:
            session = sessions.get(model.id)
            if session is None:
                chosen = devices.providers(ort.get_available_providers(), self.device)
                if ctx is not None:
                    ctx.progress(0.0, f"Loading the model {model.id} on {devices.name_of(chosen)}")
                graph = with_embedding(model, path) if embedding else str(path)
                devices.prepare(chosen)
                with _out_of_gpu_memory_said(model, chosen[0][0]):
                    session = ort.InferenceSession(
                        graph,
                        devices.session_options(chosen, self.threads),
                        providers=[name for name, _ in chosen],
                        provider_options=[options for _, options in chosen],
                    )
                del graph
                sessions[model.id] = session
                _say_where(model, session, chosen, self.device, self.variant)
            return session

    def scores(self, model: Model | Tagger, batch: np.ndarray, ctx=None) -> np.ndarray:
        """Each image's output, `[n, outputs]`: a classifier's label probabilities (they end in a softmax), a tagger's
        score per tag."""
        session = self.session(model, ctx)
        with self._run_lock, _out_of_gpu_memory_said(model, session.get_providers()[0]):
            (output,) = session.run([model.output], {"input": batch})
        return output

    def scores_and_embeddings(self, tagger: Tagger, batch: np.ndarray, ctx=None) -> tuple[np.ndarray, np.ndarray]:
        """Each image's scores, `[n, tags]`, as `scores` gives them, and its embedding, `[n, d]` float32, from one
        forward pass of the tagger's embedding session."""
        session = self.session(tagger, ctx, embedding=True)
        with self._run_lock, _out_of_gpu_memory_said(tagger, session.get_providers()[0]):
            scores, embeddings = session.run([tagger.output, EMBEDDING], {"input": batch})
        if embeddings.ndim != 2 or embeddings.shape[0] != len(batch):
            raise GraphChanged(tagger, f"its embedding is {list(embeddings.shape)} for {len(batch)} images")
        return scores, embeddings

    def close(self) -> None:
        """Drops every session, and with it its memory, on the GPU too: what a Worker ending does. A session asked for
        afterwards is made again; closing twice does nothing more. A forward pass running meanwhile keeps its session
        until it ends."""
        with self._session_lock:
            self._sessions.clear()
            self._embedding_sessions.clear()
        gc.collect()

    def _file(self, model: Pinned, ctx) -> Path:
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
            fetching = self._downloads.get(model.local_path)
            if fetching is None or fetching.abandoned:
                fetching = _Download(model, self.endpoint, path, self._lock, lambda: self._ended(model, fetching))
                self._downloads[model.local_path] = fetching
                waiter = fetching.join(ctx)
                fetching.start()
            else:
                waiter = fetching.join(ctx)
        fetching.wait(waiter)
        return path

    def _ended(self, model: Pinned, fetching: _Download) -> None:
        # Under the lock. An abandoned download may end after a new one has taken its place.
        if self._downloads.get(model.local_path) is fetching:
            del self._downloads[model.local_path]


def with_embedding(tagger: Tagger, path: Path) -> bytes:
    """The tagger's graph, from its model's file at `path`, with a second output, `embedding`: the data input of its
    head (`Tagger.head_node`), each image's pooled features, `[batch, d]`, cast to float32 when the graph has it in
    float16. Its scores stay as they were.

    The head is checked by walking back from the scores' output through `Sigmoid`, `Add` and `Cast` to the first `Gemm`
    or `MatMul`: a head missing, another node than that, or an input that is not `[batch, d]`, is `GraphChanged`. The
    graph comes back serialized, for `ort.InferenceSession` to take as it is: every tagger is under protobuf's 2 GB.
    """
    model = onnx.load(str(path))
    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}
    weights = {initializer.name: initializer for initializer in graph.initializer}

    head = next((node for node in graph.node if node.name == tagger.head_node), None)
    if head is None:
        raise GraphChanged(tagger, f"it has no node `{tagger.head_node}`")
    found = _head_before(producers, weights, tagger.output)
    if found is not head:
        named = "nothing" if found is None else f"`{found.name}` ({found.op_type})"
        raise GraphChanged(tagger, f"its scores come from {named}, not `{tagger.head_node}`")

    features, weight = head.input[0], weights.get(head.input[1])
    if weight is None or len(weight.dims) != 2:
        raise GraphChanged(tagger, f"`{head.name}`'s weight is not a matrix of the graph's")
    attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in head.attribute}
    if head.op_type == "Gemm":
        # A Gemm's inputs are matrices: its data input is `[batch, d]` unless it is transposed.
        if attributes.get("transA", 0):
            raise GraphChanged(tagger, f"`{head.name}` takes its input transposed, `[d, batch]`")
        width = weight.dims[1] if attributes.get("transB", 0) else weight.dims[0]
    else:
        width = weight.dims[0]
    known = {info.name: info for info in [*graph.value_info, *graph.input, *graph.output]}.get(features)
    shape = known.type.tensor_type.shape if known is not None and known.type.tensor_type.HasField("shape") else None
    if shape is not None:
        if len(shape.dim) != 2:
            raise GraphChanged(tagger, f"`{head.name}`'s input `{features}` has {len(shape.dim)} dimensions, not 2")
        given = shape.dim[1].dim_value if shape.dim[1].HasField("dim_value") else width
        if given != width:
            raise GraphChanged(tagger, f"`{head.name}`'s input `{features}` is {given} wide, not {width}")
    elif head.op_type != "Gemm":
        raise GraphChanged(tagger, f"the shape of `{head.name}`'s input `{features}` is not in the graph")

    # The input's type is the weight's: an operator's inputs share theirs.
    embedding = features
    if weight.data_type == TensorProto.FLOAT16:
        embedding = f"{features}/{EMBEDDING}_float32"
        graph.node.append(helper.make_node("Cast", [features], [embedding], to=TensorProto.FLOAT, name=embedding))
    elif weight.data_type != TensorProto.FLOAT:
        raise GraphChanged(tagger, f"`{head.name}`'s input is {TensorProto.DataType.Name(weight.data_type)}")
    graph.node.append(helper.make_node("Identity", [embedding], [EMBEDDING], name=EMBEDDING))
    graph.output.append(helper.make_tensor_value_info(EMBEDDING, TensorProto.FLOAT, ["batch", width]))
    return model.SerializeToString()


def _head_before(producers: dict, weights: dict, output: str):
    """The `Gemm` or `MatMul` the scores' `output` comes from through `Sigmoid`, `Add` and `Cast` only; `None` when
    another node comes first."""
    node = producers.get(output)
    while node is not None and node.op_type in _AFTER_THE_HEAD:
        # An Add's other input is its bias, a weight of the graph.
        node = next((producers[name] for name in node.input if name in producers and name not in weights), None)
    return node if node is not None and node.op_type in _HEADS else None


@contextmanager
def _out_of_gpu_memory_said(model: Model | Tagger, provider: str) -> Iterator[None]:
    """Makes what runs out of GPU memory in it `OutOfGpuMemory`, saying what to do, when `provider` (the session's) is
    not the CPU's; any other failure is left as it is."""
    try:
        yield
    except Exception as error:
        if provider != devices.CPU and devices.out_of_memory(error):
            raise OutOfGpuMemory(model) from None
        raise


def _say_where(
    model: Model | Tagger, session: ort.InferenceSession, chosen: devices.Providers, device: int, variant: str | None
) -> None:
    """A line on stderr (the Host's log, and the Plugins page's "Python output") naming the provider the session runs
    on, and saying so when it is not the GPU provider the build offered (one that could not start: onnxruntime falls
    back to the CPU); then the build, when the Host named it."""
    running = session.get_providers()[0]
    line = f"image-tags: the model {model.id} runs on {running}"
    if running != devices.CPU:
        line += f" (gpu unit {device}: {', '.join(f'{key}={value}' for key, value in chosen[0][1].items())})"
    wanted = chosen[0][0]
    if running != wanted:
        line += f", not {wanted}: it could not start"
    line += "; " if variant is None else f"; the {variant} build, "
    line += f"onnxruntime {ort.__version__}, {session.get_session_options().intra_op_num_threads} threads"
    print(line, file=sys.stderr, flush=True)
