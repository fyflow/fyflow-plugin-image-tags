# 10: fyflow-plugin-sdk 0.12.0: the Plugin describes its data, so a person can see and delete it

**What to build:** adopt `fyflow-plugin-sdk` 0.12.0's `data()`. On a 0.12 Host, the Plugin's card gets a Data link:
the Host lists what the Plugin's data directory holds, with sizes, deletes an item or all of it when a person asks,
and offers to delete it on uninstall. The Plugin only describes; it writes no deleting code. The SDK's notes are
`../fyflow-app/packages/plugin-sdk/UPGRADING.md` ("0.11 to 0.12") and its README, "Describing a Plugin's data".

When it's done, a person can see each downloaded model with its size, delete one, and see which downloaded models this
version no longer uses, so they can be cleared after an update.

**Blocked by:** 08

**Status:** resolved

- [x] **Peer range stays `^0.11.0`;** the devDependency goes to `0.12.0` for the types.
  - 0.12 only adds, and a 0.12 Host loads a `^0.11.0` Plugin (`SDK_COMPATIBLE`). A 0.11 Host never asks for `data()`, so the Plugin works there as before.
  - Per the SDK's rule, raise the range only for what a Host must do at run time; `data()` is optional.
  - Check the index's `sdk` stays equal to the peer range, so 0.1.0's entry and the next one match.
- [x] **`data({ dataDir })` in `plugin.ts`:**
  - **Summary:** what the folder holds, e.g. "Downloaded models".
  - **An item per downloaded model version:** each `models/<repo>/<commit, 12 characters>` folder that exists (the repo has an owner, so it's `models/deepghs/anime_real_cls/097a6c1e9e62`).
    - **A catalogue model:** labelled with its id and source (`anime_real: deepghs/anime_real_cls at 097a6c1e9e62`); lost: "Downloads again when next used".
    - **A commit the catalogue no longer pins** (left by an earlier version): labelled as not used by this version; lost: nothing this version needs. This replaces the pruning-at-start idea: the person clears it.
  - **Only what exists:** list the folders found on disk, not the catalogue. A missing `models/` folder gives no items.
  - **Leftover temp files** (`.<file>.*.part` from an interrupted download) need no item: they show under "Not described by the Plugin" and go with "Delete all". Decide whether an item for them reads better.
  - **Fast and safe:** directory listings only, no hashing, no reading files; well under the Host's one second. Never throw: on an error, answer what was found so far, or no items.
  - Paths are relative and `/`-separated, as the Host requires; never absolute, never with `..`.
- [x] **Workers cope with a deleted model,** as on a first run: the Host stops the Workers before deleting, and the next Task downloads again. Check that nothing caches a "downloaded" state across processes, and that `status()` reports the model as not downloaded after.
- [x] **`modelDir`** (a Profile's model folder, for machines without internet) is outside the data directory, so it is not described or deleted. Say so in the README.
- [x] **Tests** (vitest, with a temp data directory):
  - no `models/` folder → no items;
  - each catalogue model downloaded → one item each, labelled with its id, lost "Downloads again when next used";
  - a folder of an unpinned commit → an item marked not used by this version;
  - a temp `.part` file → no item (or the item decided above);
  - every item's path is relative, `/`-separated and inside the data directory;
  - a folder that can't be read → no throw.
- [x] **Docs:**
  - **README:** a short "Data" section: what the Data link shows, that deleting a model only costs a download, and that a model left by an earlier version can be cleared there. Drop advice to delete files by hand if any.
  - **Spec and `todo.md`:** the Data link replaces "prune at start" and a `ClearModelCache` Node Type.
- [x] **Verified by hand on a 0.12 Host:**
  - after a Task has downloaded the models, the Data link lists both with sizes;
  - a fake old-commit folder shows as not used by this version;
  - deleting one model, then running a Task, downloads it again and the Task completes;
  - uninstalling offers to delete the data, with the summary.

## Changed while building it (2026-11-15)

- **`.part` files get no item:** a download writes its temp file next to the final one, inside the version's folder, so an interrupted download's leftovers go with that model's item.
- **Order:** the catalogue's models first, in catalogue order, then the folders left by earlier versions, by name.
- **An unpinned commit** reads "deepghs/anime_real_cls at 0123456789ab: not used by this version", lost "Nothing this version needs".
- **Workers already coped with a deleted model:** `_file()` checks the disk on every request. One pytest covers it: a fresh Worker downloads a deleted model again.
- **Spec and `todo.md`** never mentioned pruning at start or `ClearModelCache`. The spec's Models section now says the Data link takes their place, and the TensorRT plan says `data()` will list engine folders too. `todo.md` is unchanged.
- **The index's `sdk`** is in fyflow-plugins, not here; it stays `^0.11.0` with the peer range.
- **Verified by hand** on fyflow-app at SDK 0.12.0, with the packed `.tgz` installed by upload on a fresh Host:
  - after a ClassifyImage Task, the Data link listed both models (about 16.8 MB each), "Not described" 0;
  - a fake `models/deepghs/anime_real_cls/0123456789ab` showed as not used by this version;
  - deleting `anime_real` removed its folder, the status turned to "Models download on first use" naming it, and the next Task downloaded it again and completed;
  - uninstalling with `deleteData` removed the data folder. The web UI's uninstall offer lists each item's label and what is lost, with the total.
