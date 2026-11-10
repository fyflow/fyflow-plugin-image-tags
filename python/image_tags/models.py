"""Getting the catalogue's models: downloaded on first use, checked, and kept as inference sessions.

A model is fetched from its pinned commit over plain HTTPS into a temp file beside its place in the data directory,
checked against its sha256, then renamed into `<dataDir>/models/<repo>/<commit, 12 characters>/<name>/`. So a
half-written or unpinned file is never loaded, and two processes fetching at once cannot spoil each other's.
"""

import hashlib
import os
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .catalogue import Model

#: Where models come from: Hugging Face, or a mirror of it named the way `huggingface_hub` reads it.
DEFAULT_ENDPOINT = "https://huggingface.co"
CHUNK_BYTES = 1 << 20
TIMEOUT_SECONDS = 60


class ModelUnavailable(Exception):
    """A model could not be had; it fails the Task that wanted it, and the next one tries again."""

    def __init__(self, model: Model, reason: str):
        super().__init__(f"Could not get the model {model.id}: {reason}")


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download(model: Model, endpoint: str, path: Path) -> None:
    """Fetches `model` from `endpoint` to `path`, which appears only once the whole file matches the pinned sha256."""
    url = f"{endpoint}/{model.remote_path}"
    partial = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, partial = tempfile.mkstemp(dir=path.parent, prefix=f".{model.file}.", suffix=".part")
        digest = hashlib.sha256()
        with os.fdopen(handle, "wb") as file, urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            while chunk := response.read(CHUNK_BYTES):
                digest.update(chunk)
                file.write(chunk)
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
        # paths are enabled).
        raise ModelUnavailable(model, f"cannot write {path}: {error}") from None
    finally:
        if partial is not None and os.path.exists(partial):
            os.unlink(partial)


class Models:
    """The models of one Worker instance: their files under `data_dir`, and one session each once loaded."""

    def __init__(self, data_dir: str, endpoint: str | None = None):
        self.root = Path(data_dir) / "models"
        self.endpoint = (endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
        self._sessions: dict[str, ort.InferenceSession] = {}
        self._lock = threading.Lock()

    def path_of(self, model: Model) -> Path:
        return self.root / model.local_path

    def session(self, model: Model) -> ort.InferenceSession:
        """The model's inference session, made on first use (downloading the model first if need be)."""
        with self._lock:
            session = self._sessions.get(model.id)
            if session is None:
                path = self.path_of(model)
                if not path.is_file():
                    download(model, self.endpoint, path)
                session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                self._sessions[model.id] = session
            return session

    def scores(self, model: Model, batch: np.ndarray) -> np.ndarray:
        """Each image's label probabilities, `[n, len(labels)]`: the models end in a softmax."""
        (output,) = self.session(model).run(["output"], {"input": batch})
        return output
