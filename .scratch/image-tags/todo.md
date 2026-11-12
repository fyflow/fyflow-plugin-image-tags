# Later: ideas and loose ends

These aren't tickets yet. Each one gets explored, and then turned into a ticket or dropped.

## GPU or CPU per node

- **Done** in ticket 08, on `fyflow-plugin-sdk` 0.11.0: the Plugin picks its build (`pick-build.ts`), a Profile
  overrides it, and the `cpu` build holds no `gpu` unit.
- **What we'd want:** some nodes of a Flow run on the GPU and others on the CPU.
- **Why it doesn't work now:** the build (cpu, cuda or directml) is Host-wide. fyflow-app syncs one variant per Plugin, chosen by `python.accelerator`. The pool's resident holds are fixed in `plugin.ts`, so a CPU pool would still hold the `gpu` unit.
- **Way 1, inside the Plugin:**
  - A Profile start-up field `device: gpu | cpu`. A GPU build of onnxruntime also has the CPU provider, and each node picks its Profile in the Flow settings.
  - This needs fyflow-app to let a Profile set a held count to 0 (`gpu: 0`). That would extend the Profile-set resident units fyflow-app shipped in fyflow-plugin-sdk 0.11.0.
- **Way 2, in fyflow-app:**
  - Choose the variant per Profile instead of per Host. fyflow-app already keeps one environment per (Plugin, variant, lock) (`packages/runtime/src/python-envs.ts`). It would have to sync each variant some Profile uses, and start each pool's process from its Profile's environment.
  - A Plugin would also need resident holds per variant: on `cpu`, `{ cpu: n }` only; on `cuda`, `{ gpu: 1, cpu: n }`. That would also remove the "the `gpu` unit is held on the cpu variant" caveat.
- **Cost of Way 2:** disk and first-sync time. The CUDA environment is about 2 GB.
- **Chosen direction:** Way 2, with the Plugin picking its own build. This was requested from fyflow-app:
  - a `pickVariant` hook, with SDK helpers such as `hasNvidiaDriver()`;
  - a per-Profile override of the build;
  - resident holds per build;
  - `python.accelerator` dropped, or kept only as an override.

## ClassifyImage and ClassifyFrames take turns on one GPU

- **What happens:** they are separate pools, and each holds the one `gpu` unit. In a Flow that runs both at once, they take turns, waiting about 10 s each time they switch.
- **What to look at:** whether a Host with one GPU should count it as more than one unit for small models. Or whether fyflow-app could let pools of one Plugin share a hold.

## DirectML on machines with several GPUs, and integrated GPUs

- **How it works now:** fyflow's GPU unit 0 asks DirectML for its high-performance GPU. Units 1 and up are passed through as DirectML's raw adapter number, which follows Windows' order: unit 1 may be an integrated GPU, or the same GPU as unit 0. On a machine whose only GPU is integrated, the Plugin runs on it.
- **Decided for now (2026-11-12):** leave the code as it is and document it in the README:
  - unit 0 is the fastest GPU;
  - several `gpu` units aren't supported on `directml`;
  - to avoid an integrated GPU, set the Profile's build to `cpu`.
- **If it comes back:**
  - **Benchmark first:** on this laptop, force DirectML onto the Intel iGPU (raw `device_id=0`) and compare a batch with the `cpu` build. If the iGPU is clearly slower, the default is wrong for machines whose only GPU is integrated.
  - **Exclude iGPUs:** enumerate adapters through DXCore in `ctypes`, which reports `IsHardware` and `IsIntegrated`.
    - Units count usable GPUs, fastest first: discrete only, unless a Profile start-up field (for example `integratedGpu`) allows integrated ones after them.
    - Each unit maps to DirectML's `device_id` by LUID. Check first which adapter order `device_id` follows.
    - On a machine whose only GPU is integrated, `pickVariant` still picks `directml` (it sees no Profile), so the Worker would fall back to the CPU, holding the `gpu` unit for nothing.

## Settings that depend on the chosen model

- **The need:** a richer model, such as an object detector, has parameters to tune per Node (score threshold, NMS IoU, maximum detections, classes to keep), and different models take different ones.
- **What fyflow allows now (2026-11-12):** one settings schema per Node Type, in fyflow's keyword subset (`fyflow-app/packages/web/src/editor/schema-form.ts`, `SUPPORTED_KEYWORDS`): no `oneOf`, `anyOf` or `if`/`then`, so the form can't show fields by the chosen `classifier`.
- **Ways within that:**
  - **A Node Type per kind of model:** detection returns boxes, not a tag, so it gets its own Node Type, such as `DetectObjects`, whose shared parameters are plain settings. The preferred way for detection.
  - **One optional section per model:** `options: { <model>: {...} }`. The Worker reads only the chosen model's section and warns about the others.
  - **Flat optional fields,** each described as "used by X": fine for two or three.
- **To look into in fyflow-app:** conditional schemas: `oneOf` branches told apart by a `const` on the choosing field, the form showing only the chosen branch's fields, and the Host checking a Node's settings against that branch. This would be a request to fyflow-app, like the 0.11 ones.
- **The batching catch:** `ClassifyImage` shares forward passes between Tasks. Parameters applied after the pass (thresholds, NMS, class filters) can differ per Task. Parameters that change the pass (input size, model variant) mean grouping batches by them, or putting them on the Profile, like `batchSize`.

## Small things

- **A flaky test:** `test_after_a_cancelled_download_the_next_task_downloads_again` failed once during ticket 06's runs and passed on every rerun since.
- **The frame-sheet contract:** the thumbnails repo (`fyflow-plugin-video-thumbnails`) should type and test `rect`, `video` and `config.render_timestamps`. It hasn't been filed, because that repo has no tracker set up.
- **The fyflow-app requests:** done. Both were sent, answered and shipped in `fyflow-plugin-sdk` 0.11.0 (ticket 08).
