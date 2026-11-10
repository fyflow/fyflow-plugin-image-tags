"""What the tests share: a stand-in for the Host's Task and context, and a local Hugging Face serving the pinned models.

The models are fetched once from huggingface.co at their pinned commits into `python/.cache/` (gitignored, checked
against the catalogue's sha256), then served from a local HTTP server that the Worker reaches through `HF_ENDPOINT`.
"""

import http.server
import shutil
import threading
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
    """A local Hugging Face: serves what is under `root` and counts the requests for each path."""

    def __init__(self, root):
        self.root = root
        self.requests = []
        hub = self

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(hub.root), **kwargs)

            def do_GET(self):
                hub.requests.append(self.path)
                super().do_GET()

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


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
def config(data_dir):
    """The Worker's config as the Host gives it: start-up fields, and `host.dataDir` since the Plugin asks for one."""
    return {"host": {"dataDir": str(data_dir)}}
