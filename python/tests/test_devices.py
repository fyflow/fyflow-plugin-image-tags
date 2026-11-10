"""Where a session runs: on the GPU the process was granted when the variant's ONNX Runtime offers one, else on the
CPU. `pnpm check` has the `cpu` build only, so the GPU cases are the choice itself, given what a build offers."""

import onnxruntime as ort
import pytest

from image_tags.catalogue import MODELS
from image_tags.devices import CPU, CUDA, DIRECTML, device_of, providers, session_options
from image_tags.worker import ClassifyFrames, ClassifyImage
from tests.conftest import IMAGES, Pool, Task, run

ANIME = IMAGES / "anime-02-train-window.jpg"


def test_the_cuda_build_runs_on_the_granted_device_with_the_cpu_as_fallback():
    # onnxruntime-gpu, as it lists them
    assert providers(["TensorrtExecutionProvider", CUDA, CPU], 1) == [(CUDA, {"device_id": 1}), (CPU, {})]


def test_the_directml_build_runs_the_first_unit_on_the_fastest_gpu():
    # DirectML numbers adapters as Windows lists them, where a laptop's integrated GPU comes before its NVIDIA one:
    # the `gpu` units are numbered as CUDA numbers devices, fastest first.
    assert providers([DIRECTML, CPU], 0) == [
        (DIRECTML, {"performance_preference": "high_performance", "device_filter": "gpu"}),
        (CPU, {}),
    ]


def test_the_directml_build_runs_another_unit_on_that_adapter():
    assert providers([DIRECTML, CPU], 2) == [(DIRECTML, {"device_id": 2}), (CPU, {})]


def test_the_cpu_build_runs_on_the_cpu():
    assert providers([CPU], 0) == [(CPU, {})]
    # The build `pnpm check` runs, whatever its version offers.
    assert providers(ort.get_available_providers(), 0) == [(CPU, {})]


@pytest.mark.parametrize(
    ("pool", "device"),
    [
        (Pool({"gpu": [1]}), 1),
        (Pool({"gpu": [0]}), 0),
        # A Host that grants no `gpu` unit (none held, or before 0.9.0), and a class built without a pool: the first.
        (Pool(), 0),
        (Pool({"gpu": []}), 0),
        (None, 0),
    ],
)
def test_the_device_is_the_first_gpu_unit_the_process_holds(pool, device):
    assert device_of(pool) == device


def test_a_directml_session_runs_sequentially_without_memory_patterns():
    # As the DirectML provider requires (onnxruntime's docs).
    options = session_options([(DIRECTML, {"device_id": 0}), (CPU, {})], threads=4)

    assert options.enable_mem_pattern is False
    assert options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert options.intra_op_num_threads == 4


@pytest.mark.parametrize("chosen", [[(CUDA, {"device_id": 0}), (CPU, {})], [(CPU, {})]])
def test_other_sessions_keep_onnxruntimes_defaults_but_the_threads(chosen):
    options = session_options(chosen, threads=6)

    assert options.enable_mem_pattern is True
    assert options.intra_op_num_threads == 6


@pytest.mark.parametrize("worker_class", [ClassifyImage, ClassifyFrames])
def test_a_worker_runs_its_sessions_on_the_gpu_its_process_holds(hub, config, worker_class):
    assert worker_class(config, Pool({"gpu": [1]})).models.device == 1
    assert worker_class(config).models.device == 0


def test_a_session_says_where_it_runs_when_it_loads(hub, config, capfd):
    worker = ClassifyImage(config, Pool({"gpu": [0]}))
    run(worker, Task({"path": str(ANIME)}, {"classifier": "anime_real"}))

    assert worker.models.session(MODELS["anime_real"]).get_providers() == [CPU]
    assert "image-tags: the model anime_real runs on CPUExecutionProvider" in capfd.readouterr().err
