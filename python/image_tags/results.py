"""Where a frames Task's per-frame answers go: in its result (`lists: "result"`, the default), or in a JSON file
(`lists: "file"`), so a large one does not sit in the Flow, as VideoThumbnail's `ExtractFrames` does with its lists.

The file holds the whole result, `frames` included; the Task's result is then that without `frames`, plus `file`,
`count` and `reused`. Where the file goes: the Node's `output` (a `.json` file, or a folder), else the Profile's
`outputDir` (a per-Task field, which a Flow or Node may set), else the Plugin's own `results/` folder in its data
directory. A path given (`output`, `outputDir`) must be inside an Allowed root with write access; the Plugin's own
folder needs none.

In a folder the file is named `<name>.<key>.json`: the key covers the Node Type, the input, each frame file's path,
size and last change, the settings and the models' commits. So a Task on unchanged frames with the same settings finds
the file it wrote before and hands it on (`reused: true`) without running a model, unless `overwrite` is set. A `.json`
path given is always written. Each file is written beside its place and renamed into it, so a file there is whole.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .frames import AllowedPath

#: What a result file holds changed: a file written before has another key, so it is not reused.
FORMAT = 1
#: The Plugin's own folder for result files, in its data directory.
RESULTS_FOLDER = "results"
#: What the Host takes as a Task's result is 16 MiB: a larger one fails the Task, so this says what to do first.
MAX_RESULT_BYTES = 15 * 1024 * 1024


@dataclass
class FileOptions:
    """A Task's `lists: "file"`: its `output` (None when it gives none) and `overwrite`."""

    output: str | None
    overwrite: bool


def file_options(settings: dict) -> FileOptions | None:
    """The Task's file options, or None for `lists: "result"`. Checked before anything is read or run."""
    lists = settings.get("lists") or "result"
    if lists not in ("result", "file"):
        raise ValueError(f'`lists` is not "result" or "file": {lists!r}')
    output = settings.get("output")
    if output is not None and (not isinstance(output, str) or output == ""):
        raise ValueError(f"`output` is not a path: {output!r}")
    overwrite = settings.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise ValueError(f"`overwrite` is not true or false: {overwrite!r}")
    return FileOptions(output, overwrite) if lists == "file" else None


def key_of(node_type: str, input, files: list[str], settings: dict, models: list) -> str:
    """What makes a result, hashed: the same frames, unchanged, settings and models make the same key."""
    stats = []
    for file in files:
        stat = os.stat(file)
        stats.append([file, stat.st_size, stat.st_mtime_ns])
    made_of = {
        "format": FORMAT,
        "nodeType": node_type,
        "input": input,
        "files": stats,
        "settings": settings,
        "models": [[model.id, model.repo, model.commit, getattr(model, "file", None)] for model in models],
    }
    return hashlib.sha256(json.dumps(made_of, sort_keys=True, default=str).encode()).hexdigest()[:16]


def target(
    options: FileOptions,
    output_dir: str | None,
    data_dir: str | None,
    name: str,
    key: str,
    allowed_path: AllowedPath,
) -> tuple[str, bool]:
    """The file to write, and whether its name was made here (so a file there already may be reused). A path given is
    checked for write access; the Plugin's own folder is not."""
    if options.output is not None and options.output.lower().endswith(".json"):
        return allowed_path(options.output, "write"), False
    given = options.output if options.output is not None else output_dir
    if given:
        folder = allowed_path(given, "write")
    elif data_dir:
        folder = os.path.join(data_dir, RESULTS_FOLDER)
    else:
        raise ValueError(
            "No folder to write the frames' answers in: this Host gives the Plugin no data directory, so set `output`, "
            "or `outputDir` in this Node's Profile"
        )
    return os.path.join(folder, f"{_safe(name)}.{key}.json"), True


def reused(path: str) -> dict | None:
    """The result a file there holds, or None when there is none, or none that reads as one."""
    try:
        with open(path, encoding="utf-8") as file:
            result = json.load(file)
    except (OSError, ValueError):
        return None
    return result if isinstance(result, dict) and isinstance(result.get("frames"), list) else None


def write(path: str, result: dict) -> None:
    """`result` as JSON at `path`, written beside it first, so a file there is always whole."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    part = f"{path}.{os.getpid()}.part"
    try:
        with open(part, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, separators=(",", ":"))
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)


def summary(result: dict, path: str, was_reused: bool) -> dict:
    """What a Task hands on when its answers went to a file: the result without `frames`, and where they are."""
    small = {key: value for key, value in result.items() if key != "frames"}
    small.update({"file": path, "count": len(result["frames"]), "reused": was_reused})
    return small


def fits(result: dict) -> dict:
    """`result`, when the Host takes it; else a failure saying to write the answers to a file."""
    size = len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode())
    if size > MAX_RESULT_BYTES:
        raise ValueError(
            f"The result is {size / 1024 / 1024:.1f} MiB, more than a Task may hand on: set `lists` to \"file\" to "
            "write the frames' answers to a file"
        )
    return result


def name_of(path: str | None, video: str | None, files: list[str]) -> str:
    """What a result file is named after: the video, else the sheet, folder or image, else a list's first frame's
    folder."""
    if video:
        return Path(video.replace("\\", "/")).stem
    if path:
        return Path(path).stem or Path(path).name
    return Path(files[0]).parent.name if files else "frames"


def _safe(name: str) -> str:
    """`name` as a file name: no separators or characters Windows refuses, never empty."""
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .") or "frames"
