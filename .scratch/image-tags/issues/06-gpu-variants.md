# 06: GPU variants

**What to build:** on a Host with `python.accelerator` set to `cuda` or `directml`, both Node Types run on the GPU they were granted. See the spec's Plugin and Running sections.

**Blocked by:** 01

**Status:** resolved

- [x] `variants` as conflicting extras: `cpu` → `onnxruntime`, `cuda` → `onnxruntime-gpu[cuda,cudnn]`, `directml` → `onnxruntime-directml`.
- [x] `onnxruntime` moves out of the base dependencies into the `cpu` extra (ticket 01 put it in the base): otherwise it is installed beside `onnxruntime-gpu` or `onnxruntime-directml` and the two clash.
- [x] `cuda`: `onnxruntime.preload_dlls()` before the first session; no system CUDA install needed.
- [x] Device from `pool.grants["gpu"][0]`, passed as the provider's `device_id`.
- [x] Both Node Types' pools: `residentGroups: ["gpu"]`, `idleTimeout: 5_000`.
- [x] README: setting `python.accelerator` (`fyflow.config.json` or `FYFLOW_PYTHON_ACCELERATOR`, then a Host restart), the `gpu` unit count on the Resource Groups page, the Plugins page showing the variant, the two Node Types taking turns on one GPU, and the `gpu` unit being held on `cpu` too.
- [x] Verified by hand on a GPU (directml and/or cuda); `pnpm check` stays on the CPU.

## Changed while building it (2026-11-10)

- Pools hold `residentGroups: { gpu: 1, cpu: 4 }`, not `["gpu"]` (ticket 05's resource model).
- DirectML numbers adapters in Windows' order, so `device_id = 0` ran on the integrated Intel GPU of a laptop with an NVIDIA GPU. fyflow's `gpu` unit 0 now passes `performance_preference=high_performance, device_filter=gpu` and no `device_id`; units 1 and up are still DirectML's raw adapter number. CUDA uses `device_id` as planned. See `../todo.md`.
- A session's `run` is serialised per process (DirectML takes no concurrent `run`), and each session logs its provider to stderr.
- Verified by hand on an RTX 3050 Ti Laptop GPU with `cuda` and `directml`: same tags as `cpu` on all 18 reference images and the sample sheet.
