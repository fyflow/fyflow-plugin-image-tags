# 09: Allowed roots: the Plugin reads only what a Profile allows

**What to build:** opt in to fyflow-app's Allowed roots (`roots: true`, SDK 0.11.0, ADR-0030 in fyflow-app). Every
path a Task gives either Node Type is checked against the Task's reach before anything is read, so the Plugin reads
only folders a person has allowed on the Profiles page.

Read first:
- the SDK README's "Allowed roots: the folders a Task may touch", and the `roots: true` item under "Python Workers"
  (`../fyflow-app/packages/plugin-sdk/README.md`);
- `../fyflow-app/docs/adr/0030-allowed-roots-are-the-hosts.md`;
- the Host's Python checks: `allowed_path` in `../fyflow-app/packages/runtime/python/fyflow_worker/_shim.py`.

**Blocked by:** 08

**Status:** resolved

- [x] **`roots: true` in `plugin.ts`.**
  - Neither Worker's `settingsSchema` may name `roots`, and a test should guard that.
  - We have no roots field of our own, so nothing has to migrate.
- [x] **ClassifyImage checks its input `path`:** `ctx.allowed_path(path, "read")` before opening the image.
  - The check touches the disk, so it runs off the event loop, in the same blocking thread pool that reads the image.
  - The image is read from the path `allowed_path` returns, with links resolved.
  - The result's `path` stays the input path as given.
- [x] **ClassifyFrames checks its input `path`,** whether it's a sheet, a folder or a single image.
  - **For a folder:** the folder is checked, and so is each image read from it. A link inside an allowed folder can point outside it, and `allowed_path` refuses that.
  - **Refused images inside a folder:** they're skipped with a warning, like an unreadable image today. They don't fail the Task.
  - **The sheet's `video` field:** it's metadata from the COM JSON and isn't read, so it isn't checked.
- [x] **A refused path fails the Task** with the `PermissionError`'s message. That message already tells the person what to do: add a root on the Profiles page, or the path is outside every root.
- [x] **Not checked:**
  - **`modelDir`:** a Profile-only start-up field, so it's already a Profile's grant.
  - **The Host's data directory.**
- [x] **Tests:**
  - **Fake reach:** the fake Context in `conftest.py` gets an `allowed_path` driven by `task.settings["roots"]`. It follows the shim's rules:
    - absolute paths only;
    - links resolved;
    - inside a root with enough access;
    - an empty reach refuses with the "add one on the Profiles page" message.
  - **Every existing test still passes** with a fixtures root in its reach.
  - **New tests:**
    - an empty reach refuses;
    - a path outside the reach refuses;
    - for a folder: a link inside an allowed folder pointing outside it is skipped with a warning. Skip that test where links can't be made, as on Windows without the privilege.
- [x] **`pnpm eval`** calls `ClassifyFrames` itself. Its context grants the folders the CSV names, or doesn't check at all; it's a local tool.
- [x] **Docs:**
  - **README:**
    - add an Allowed roots section;
    - say the Plugin refuses every path until a Profile has a read root covering it;
    - say how to add one: the Profile's roots, or the Plugin's own in its section.
  - **Spec:** drop "no Allowed roots in v1" and the out-of-scope item.
  - **`todo.md`:** remove the Allowed roots item.
- [x] **Verified by hand on a Host:**
  - with no roots, a run fails with the Profiles-page message;
  - with a read root on the fixtures folder, the same run tags the image;
  - with a root elsewhere, the run is refused.

## Changed while building it (2026-11-12)

- **A folder where every image is refused** fails the Task with the first `PermissionError`, not "The folder has no images", so the person sees why.
- **Messages after the check** ("Cannot read the image …", "The folder has no images: …") name the resolved path. They differ from the given path only when links are involved.
- **The tests' fake `allowed_path`** leaves out `unfollowed` and the "root that is no folder" rule, because the Workers use neither. By default the fake Task's reach is the fixtures folder plus the system temp folder, so the existing `tmp_path` tests are unchanged.
- **`pnpm eval`'s context** resolves the path and checks nothing; it's a local tool.
- **Verified by hand** on fyflow-app at its 0.11.0 commit, with ClassifyImage on a reference image and ClassifyFrames on the fixtures folder:
  - **With no roots:** both were refused with "this Profile has no Allowed roots for it yet. Add one with read access on the Profiles page".
  - **With a read root on the fixtures folder:** both completed.
  - **With a read root on another folder:** both were refused with "is outside every Allowed root with read access on this Profile".
  - A refused Task leaves its run `awaiting_action`, as any failed Task does on the Host.
