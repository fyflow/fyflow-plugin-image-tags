"""Where the models' sessions run: on the GPU the process holds, when the variant's ONNX Runtime build offers one.

The Host syncs the build the Plugin picks for the machine (`pick-build.ts`), or the one a Profile sets
(`pyproject.toml`'s extras): `cpu` offers the CPU only, `cuda` (onnxruntime-gpu) CUDA, `directml` (onnxruntime-directml)
DirectML. The providers come from what the installed build offers, not from the build's name (`host.variant`).

The device is the first `gpu` unit the process holds (`pool.grants["gpu"]`, numbered as CUDA numbers devices, fastest
first), passed to the provider; the Host sets no `CUDA_VISIBLE_DEVICES`. DirectML numbers adapters as Windows lists
them, where a laptop's integrated GPU comes first, so the first unit asks it for the fastest GPU instead; another unit
is DirectML's adapter of that number. The CPU stays in the list, for what a GPU provider cannot run, and for a GPU
provider that cannot start (onnxruntime then falls back to the CPU, which the session's load line on stderr shows).
"""

import threading
from collections.abc import Iterable

import onnxruntime as ort

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
DIRECTML = "DmlExecutionProvider"
#: The GPU providers, in the order they are preferred should a build offer more than one.
GPU_PROVIDERS = (CUDA, DIRECTML)

Providers = list[tuple[str, dict]]

_preloaded = threading.Lock()
_preload_done = False


def device_of(pool) -> int:
    """The GPU the process holds: its first `gpu` unit, else 0 (none granted, or a class built without its pool)."""
    units = (getattr(pool, "grants", None) or {}).get("gpu") or [0]
    return units[0]


def providers(available: Iterable[str], device: int) -> Providers:
    """The providers a session runs on, with their options, given those the installed build offers: its GPU provider
    on `device`, then the CPU; or the CPU alone."""
    offered = set(available)
    for provider in GPU_PROVIDERS:
        if provider in offered:
            options = {"device_id": device}
            if provider == DIRECTML and device == 0:
                # A `device_id` would win over the preference (DirectML's adapter 0 on a laptop: the integrated GPU).
                options = {"performance_preference": "high_performance", "device_filter": "gpu"}
            return [(provider, options), (CPU, {})]
    return [(CPU, {})]


def session_options(chosen: Providers, threads: int) -> ort.SessionOptions:
    """A session's options: a forward pass spread over `threads` threads (onnxruntime's `intra_op_num_threads`), and
    what the DirectML provider requires, no memory patterns and sequential execution."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    if any(name == DIRECTML for name, _ in chosen):
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def prepare(chosen: Providers) -> None:
    """What a provider needs before its first session, once per process: for CUDA, loading the CUDA and cuDNN DLLs of
    the `nvidia-*` packages the `cuda` variant installs, so the machine needs only a driver."""
    global _preload_done
    if not any(name == CUDA for name, _ in chosen):
        return
    with _preloaded:
        if not _preload_done:
            preload = getattr(ort, "preload_dlls", None)
            if preload is not None:
                preload()
            _preload_done = True
