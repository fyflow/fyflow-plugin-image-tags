# 06: GPU variants

**What to build:** on a Host with `python.accelerator` set to `cuda` or `directml`, both Node Types run on the GPU they were granted. See the spec's Plugin and Running sections.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] `variants` as conflicting extras: `cpu` → `onnxruntime`, `cuda` → `onnxruntime-gpu[cuda,cudnn]`, `directml` → `onnxruntime-directml`.
- [ ] `onnxruntime` moves out of the base dependencies into the `cpu` extra (ticket 01 put it in the base): otherwise it is installed beside `onnxruntime-gpu` or `onnxruntime-directml` and the two clash.
- [ ] `cuda`: `onnxruntime.preload_dlls()` before the first session; no system CUDA install needed.
- [ ] Device from `pool.grants["gpu"][0]`, passed as the provider's `device_id`.
- [ ] Both Node Types' pools: `residentGroups: ["gpu"]`, `idleTimeout: 5_000`.
- [ ] README: setting `python.accelerator` (`fyflow.config.json` or `FYFLOW_PYTHON_ACCELERATOR`, then a Host restart), the `gpu` unit count on the Resource Groups page, the Plugins page showing the variant, the two Node Types taking turns on one GPU, and the `gpu` unit being held on `cpu` too.
- [ ] Verified by hand on a GPU (directml and/or cuda); `pnpm check` stays on the CPU.
