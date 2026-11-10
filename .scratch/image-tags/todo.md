# Later: ideas and loose ends

These aren't tickets yet. Each one gets explored, and then turned into a ticket or dropped.

## GPU or CPU per node

- **What we'd want:** some nodes of a Flow run on the GPU and others on the CPU.
- **Why it doesn't work now:** the build (cpu, cuda or directml) is Host-wide. fyflow-app syncs one variant per Plugin, chosen by `python.accelerator`. The pool's resident holds are fixed in `plugin.ts`, so a CPU pool would still hold the `gpu` unit.
- **Way 1, inside the Plugin:**
  - A Profile start-up field `device: gpu | cpu`. A GPU build of onnxruntime also has the CPU provider, and each node picks its Profile in the Flow settings.
  - This needs fyflow-app to let a Profile set a held count to 0 (`gpu: 0`). That would be an extension of `fyflow-app-request-resident-units.md`.
- **Way 2, in fyflow-app:**
  - Choose the variant per Profile instead of per Host. fyflow-app already keeps one environment per (Plugin, variant, lock) (`packages/runtime/src/python-envs.ts`). It would have to sync each variant some Profile uses, and start each pool's process from its Profile's environment.
  - A Plugin would also need resident holds per variant: on `cpu`, `{ cpu: n }` only; on `cuda`, `{ gpu: 1, cpu: n }`. That would also remove the "the `gpu` unit is held on the cpu variant" caveat.
- **Cost of Way 2:** disk and first-sync time. The CUDA environment is about 2 GB.

## ClassifyImage and ClassifyFrames take turns on one GPU

- **What happens:** they are separate pools, and each holds the one `gpu` unit. In a Flow that runs both at once, they take turns, waiting about 10 s each time they switch.
- **What to look at:** whether a Host with one GPU should count it as more than one unit for small models. Or whether fyflow-app could let pools of one Plugin share a hold.

## DirectML on machines with several GPUs

- **How it works now:** fyflow's GPU unit 0 asks DirectML for its high-performance GPU. Units 1 and up are passed through as DirectML's raw adapter number, which follows Windows' order. That number is wrong on a laptop with an integrated GPU listed first.
- **What fixing it means:** listing the adapters through DXGI by GPU preference, so unit n is the n-th fastest.

## Small things

- **A flaky test:** `test_after_a_cancelled_download_the_next_task_downloads_again` failed once during ticket 06's runs and passed on every rerun since.
- **The frame-sheet contract:** the thumbnails repo (`fyflow-plugin-video-thumbnails`) should type and test `rect`, `video` and `config.render_timestamps`. It hasn't been filed, because that repo has no tracker set up.
- **The fyflow-app request:** `fyflow-app-request-resident-units.md` hasn't been sent yet.
