"""Runs TagScenes on a video's frames and writes a page to review its scenes on:
`uv run --extra cpu python scripts/review_scenes.py <folder or sheet> … [--set NAME=VALUE …] [--tagger NAME]`.

Run by hand, from `python/`; never by `pnpm check`, never published, since the page shows frames of the user's videos.
Each input (a folder of timestamped frames, such as DetectScenes' or VideoThumbnail's, or a frame sheet) goes through
the `TagScenes` Worker itself, with no Host: its Task's Allowed roots are the inputs given (a folder, or a sheet's
folder). `--set` overrides a setting (camelCase, as a Node sets it: `minSceneDuration=20`) or a constant (UPPER_CASE:
one of `scenes.py`'s, or `BLANK_STD`), its value read as JSON where it parses, else as text. Constants are put back
once the run is over.

For each input, named after its video's stem (else the folder's name, or the sheet's stem), it writes to
`.scratch/image-tags/notes/review/` (gitignored):

- `<name>.result.json`: the Task's result, as TagScenes hands it on;
- `<name>.review.html`: one card per scene (times, rating and peaks, tags and highlights as chips, the frames as
  thumbnails: a folder's from their file URLs, a sheet's cropped into the page), self-contained.

On the page everything counts as right until it is clicked: "merge with previous" on a card, the gap between two
thumbnails ("a scene starts here"), a chip (a wrong tag, highlight or rating), a note per scene. **Save review**
downloads `<name>.review.json`, keyed by time: `{video, settings, boundaries, merges, cuts, wrong, notes}`. Kept beside
the page, it is read on the next run: the page then says how many boundaries changed and whether each earlier comment
is addressed or still open.

What it prints is times, counts and file paths only, never a tag: the videos are private.
"""

import argparse
import base64
import contextlib
import hashlib
import html
import io
import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PYTHON = SCRIPTS.parent
sys.path[:0] = [str(PYTHON)]
from image_tags import frames as framing  # noqa: E402
from image_tags import scenes  # noqa: E402
from image_tags.catalogue import DEFAULT_TAGGER, TAGGERS  # noqa: E402
from image_tags.frames import Frames, read_frames  # noqa: E402
from image_tags.worker import SCENE_CONSTANTS, SCENE_MAX_TAGS, TagScenes  # noqa: E402

REVIEWS = PYTHON.parent / ".scratch" / "image-tags" / "notes" / "review"
#: Where the Worker keeps what it downloads, when the tagger is not in `MODELS` already.
DATA = PYTHON / ".cache" / "review"
#: The taggers laid out as a Profile's `modelDir`, put there by hand (the README's Development): used when the
#: tagger is in it.
MODELS = PYTHON / ".cache" / "models"

#: The settings `--set` may give, by the names a Node uses.
SETTINGS = (
    "tagger",
    "thresholds",
    "maxTags",
    "embeddingWeight",
    "minSceneDuration",
    "ignoreTags",
    "minCoverage",
    "includeTags",
)
#: The constants `--set` may give, and the module each is in.
CONSTANTS = {
    **{name: scenes for name in (*SCENE_CONSTANTS, "SHOT_SPECIFIC_TAGS", "CATEGORIES")},
    "BLANK_STD": framing,
}
#: Two boundaries this many seconds apart or less are the same one, between two runs.
NEAR = 1.0
#: A sheet's frames are cropped into the page this wide, in pixels.
THUMB_WIDTH = 240


# --- `--set`.


def parse_sets(items: Sequence[str]) -> tuple[dict, dict]:
    """`NAME=VALUE` items as `(settings, constants)`: a camelCase name is a setting, an UPPER_CASE one a constant. A
    value is JSON where it parses, else text; a constant's takes its current value's type (a list becomes a frozenset
    or tuple, a whole number a float)."""
    settings, constants = {}, {}
    for item in items:
        name, equals, text = item.partition("=")
        name = name.strip()
        if not equals or not name:
            raise ValueError(f"`--set {item}` is not NAME=VALUE")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        if name.isupper():
            if name not in CONSTANTS:
                raise ValueError(f"No constant `{name}`: one of {', '.join(CONSTANTS)}")
            constants[name] = _like(name, getattr(CONSTANTS[name], name), value)
        elif name in SETTINGS:
            settings[name] = value
        else:
            raise ValueError(f"No setting `{name}`: one of {', '.join(SETTINGS)}")
    return settings, constants


def _like(name: str, current, value):
    """`value` in `current`'s type, or a ValueError saying it cannot be."""
    if isinstance(current, frozenset | tuple):
        if not isinstance(value, list):
            raise ValueError(f"`{name}` is a list: {value!r}")
        return type(current)(value)
    if isinstance(current, dict):
        if not isinstance(value, dict):
            raise ValueError(f"`{name}` is an object: {value!r}")
        return value
    if isinstance(current, int | float):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"`{name}` is a number: {value!r}")
        return float(value) if isinstance(current, float) else value
    return value


@contextlib.contextmanager
def overridden(constants: Mapping[str, object]) -> Iterator[None]:
    """The constants set while the block runs, and put back after it."""
    before = {name: getattr(CONSTANTS[name], name) for name in constants}
    try:
        for name, value in constants.items():
            setattr(CONSTANTS[name], name, value)
        yield
    finally:
        for name, value in before.items():
            setattr(CONSTANTS[name], name, value)


def effective(settings: Mapping[str, object], constants: Mapping[str, object]) -> dict:
    """The settings and constants a run used, as a saved review records them. The shot-specific tags only when
    overridden. Read while `overridden` holds."""
    defaults = scenes.Settings()
    used = {
        "tagger": DEFAULT_TAGGER,
        "maxTags": SCENE_MAX_TAGS,
        "embeddingWeight": defaults.embedding_weight,
        "minSceneDuration": defaults.min_scene_duration,
        "ignoreTags": [],
        "minCoverage": defaults.min_coverage,
        "includeTags": defaults.include_tags,
        **settings,
    }
    shown = [*SCENE_CONSTANTS, "BLANK_STD", *(["SHOT_SPECIFIC_TAGS"] if "SHOT_SPECIFIC_TAGS" in constants else [])]
    used["constants"] = {name: _plain(getattr(CONSTANTS[name], name)) for name in shown}
    return used


def _plain(value):
    return sorted(value) if isinstance(value, frozenset) else list(value) if isinstance(value, tuple) else value


# --- Running TagScenes.


class _Task:
    def __init__(self, path: Path, settings: Mapping, roots: list[dict]):
        self.input = {"path": str(path)}
        self.settings = {"profile": {}, **settings, "roots": roots}


class _Context:
    """A Task's context, its reach the inputs given: what `allowed_path` lets through is inside one of `roots`."""

    cancelled = False

    def __init__(self, roots: list[dict]):
        self.roots = [os.path.normcase(os.path.realpath(root["path"])) for root in roots]
        self._said = None

    def progress(self, fraction, message=None):
        said = f"{round(fraction * 100)}%"
        if said != self._said:
            self._said = said
            print(f"\r  {said} ", end="", file=sys.stderr, flush=True)

    def log(self, message):
        pass

    def allowed_path(self, path, access="read"):
        real = os.path.realpath(path)
        mine = os.path.normcase(real)
        if any(mine == root or mine.startswith(root.rstrip(os.sep) + os.sep) for root in self.roots):
            return real
        raise PermissionError(f'"{path}" is outside every Allowed root of this review')


def roots_of(paths: Sequence[Path]) -> list[dict]:
    """The Allowed roots granted for `paths`: a folder itself, a file's folder."""
    return [{"path": str(path if path.is_dir() else path.parent), "access": "read"} for path in paths]


def name_of(path: Path, source: Frames) -> str:
    """What an input's files are named after: its video's stem, else the folder's name, or the sheet's stem."""
    if source.video:
        return Path(source.video.replace("\\", "/")).stem
    return path.name if path.is_dir() else path.stem


def thumbnails(source: Frames) -> list[str]:
    """Each frame's image for the page: its file URL, or for a sheet's cell, the cell cropped into a JPEG data URI."""
    out = []
    for frame in source.frames:
        if frame.path is not None:
            out.append(Path(frame.path).resolve().as_uri())
            continue
        image = frame.load()
        image.thumbnail((THUMB_WIDTH, THUMB_WIDTH))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=80)
        out.append("data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode())
    return out


def boundaries(result: Mapping) -> list[float]:
    """Where the result's scenes after the first start, in seconds."""
    return [scene["start"] for scene in result["scenes"][1:]]


# --- Since the last review.


def since_last(review: Mapping, result: Mapping, near: float = NEAR) -> dict:
    """What changed since a saved review, against this run's `result`: the boundaries `added` and `gone` (a review
    saved without `boundaries` gives None for both), and each earlier comment with its `status`:

    - a merge (`t`, a boundary then) is addressed when no boundary is within `near` of `t` now;
    - a cut (`t`) is addressed when a boundary is within `near` of it now;
    - a wrong tag, highlight or rating (`tag`, `kind`, `start`, `end`) is addressed when the scene covering the middle
      of `start` and `end` now no longer lists it among its kind;
    - a note is `"note"`: an earlier note, to be read again.
    """
    now = boundaries(result)
    before = review.get("boundaries")
    if isinstance(before, list):
        added = [t for t in now if not _near(t, before, near)]
        gone = [t for t in before if not _near(t, now, near)]
    else:
        added = gone = None
    comments = []
    for t in review.get("merges") or []:
        comments.append({"kind": "merge", "t": t, "status": "open" if _near(t, now, near) else "addressed"})
    for t in review.get("cuts") or []:
        comments.append({"kind": "cut", "t": t, "status": "addressed" if _near(t, now, near) else "open"})
    for wrong in review.get("wrong") or []:
        scene = covering(result, (wrong["start"] + wrong["end"]) / 2)
        still = scene is not None and wrong["tag"] in listed(scene, wrong.get("kind", "tag"))
        status = "open" if still else "addressed"
        comments.append({**wrong, "kind": "wrong", "what": wrong.get("kind", "tag"), "status": status})
    for note in review.get("notes") or []:
        comments.append({"kind": "note", **note, "status": "note"})
    return {"added": added, "gone": gone, "comments": comments}


def _near(t: float, times: Sequence[float], near: float) -> bool:
    return any(abs(t - other) <= near for other in times)


def covering(result: Mapping, t: float) -> Mapping | None:
    """The scene whose time holds `t`: from its start, up to its end (the last scene's included)."""
    found = result["scenes"]
    for k, scene in enumerate(found):
        if scene["start"] <= t < scene["end"] or (k == len(found) - 1 and t == scene["end"]):
            return scene
    return None


def listed(scene: Mapping, kind: str) -> set[str]:
    """The names a scene shows as chips of `kind`: its tags, its highlights, or its rating's label and peaks."""
    if kind == "rating":
        rating = scene.get("rating") or {}
        return {rating.get("label"), *(peak["label"] for peak in rating.get("peaks", []))} - {None}
    return {entry["name"] for entry in scene.get("highlights" if kind == "highlight" else "tags", [])}


# --- The page.


def page(
    name: str,
    result: Mapping,
    times: Sequence[float],
    images: Sequence[str],
    settings: Mapping,
    earlier: Mapping | None,
) -> str:
    """The review page: self-contained, its data in a JSON script the page's own script renders from. `times` and
    `images` are each input frame's time and thumbnail (`thumbnails`); `earlier` is `since_last`'s, when a saved
    review is beside the page."""
    text = json.dumps(result, sort_keys=True)
    data = {
        "name": name,
        "resultFile": f"{name}.result.json",
        "runId": hashlib.sha1(text.encode()).hexdigest()[:12],
        "video": result.get("video") or result.get("path"),
        "tagger": result["tagger"],
        "frameCount": result["frameCount"],
        "duration": result["duration"],
        "threshold": result["threshold"],
        "warnings": result.get("warnings", []),
        "settings": settings,
        "boundaries": boundaries(result),
        "scenes": result["scenes"],
        "frames": [{"t": t, "src": src} for t, src in zip(times, images, strict=True)],
        "earlier": earlier,
    }
    # `<` escaped, so nothing in the data can close the script element.
    embedded = json.dumps(data).replace("<", "\\u003c")
    return (
        PAGE.replace("__TITLE__", html.escape(f"{name} scenes"))
        .replace("__STYLE__", STYLE)
        .replace("__SCRIPT__", SCRIPT)
        .replace("__DATA__", embedded)
    )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>__STYLE__</style>
</head>
<body>
<header id="top"></header>
<div id="layout" class="layout"><main id="scenes"></main></div>
<script id="data" type="application/json">__DATA__</script>
<script>__SCRIPT__</script>
</body>
</html>
"""

STYLE = """
:root {
  --bg: #f6f6f4; --fg: #1d1d1f; --muted: #6b6b70; --card: #ffffff; --line: #deded9;
  --chip: #ebebe6; --char: #dbe7fb; --hl: #fbe9c6; --rate: #e6e1f7; --warn: #b3261e;
  --accent: #0b6bcb; --accent-soft: #d6e6f8; --ok: #1b7a3d; --ok-soft: #d8f0df; --open: #9a5b00; --open-soft: #fbead0;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #131315; --fg: #ececec; --muted: #9a9aa0; --card: #1c1c1f; --line: #313136;
    --chip: #2b2b30; --char: #1f3352; --hl: #4a3815; --rate: #332a4d; --warn: #ff8a80;
    --accent: #6cb2ff; --accent-soft: #1d3550; --ok: #7fd49a; --ok-soft: #173a24; --open: #ffc06b; --open-soft: #46320f;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.45 system-ui, sans-serif; }
header { position: sticky; top: 0; z-index: 2; background: var(--bg); border-bottom: 1px solid var(--line);
  padding: 10px 16px; display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; }
header h1 { font-size: 17px; margin: 0; overflow-wrap: anywhere; }
header .sub { color: var(--muted); font-size: 12.5px; }
header .tally { margin-left: auto; color: var(--muted); font-size: 12.5px; }
header .warnings { flex-basis: 100%; color: var(--warn); font-size: 12.5px; }
button.action { font: inherit; padding: 5px 12px; border-radius: 6px; border: 1px solid var(--line);
  background: var(--card); color: var(--fg); cursor: pointer; }
button.action.primary { background: var(--accent); border-color: var(--accent); color: var(--bg); font-weight: 600; }
.layout { display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; padding: 16px; }
@media (min-width: 1100px) {
  .layout.with-aside { grid-template-columns: minmax(0, 1fr) 330px; }
  aside { position: sticky; top: 70px; align-self: start; max-height: calc(100vh - 86px); overflow: auto; }
}
aside { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
aside h2 { font-size: 15px; margin: 0 0 4px; }
aside p { margin: 0 0 10px; color: var(--muted); }
aside ol { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
aside li { padding: 6px 8px; border-radius: 6px; background: var(--chip); cursor: pointer; overflow-wrap: anywhere; }
.badge { display: inline-block; font-size: 11px; font-weight: 600; border-radius: 4px; padding: 0 5px;
  margin-right: 6px; }
.badge.addressed { background: var(--ok-soft); color: var(--ok); }
.badge.open { background: var(--open-soft); color: var(--open); }
.badge.note { background: var(--accent-soft); color: var(--accent); }
section.scene { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px;
  margin: 0 0 14px; }
section.scene.merged { border-top: 3px dashed var(--accent); }
section.scene.flash { outline: 2px solid var(--accent); }
.head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 10px; margin-bottom: 8px; }
.head h2 { font-size: 15px; margin: 0; }
.head .meta { color: var(--muted); font-size: 12.5px; }
.head label.merge { margin-left: auto; font-size: 12.5px; cursor: pointer; user-select: none; }
.chips { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 10px; }
.chip { font: inherit; font-size: 12px; border: 0; border-radius: 999px; padding: 2px 9px; background: var(--chip);
  color: var(--fg); cursor: pointer; }
.chip.character { background: var(--char); }
.chip.highlight { background: var(--hl); }
.chip.rating { background: var(--rate); }
.chip.peak { background: var(--rate); color: var(--warn); }
.chip .num { color: var(--muted); margin-left: 4px; }
.chip.struck { text-decoration: line-through; opacity: 0.55; outline: 1.5px solid var(--warn); }
.thumbs { display: flex; flex-wrap: wrap; row-gap: 6px; }
figure { margin: 0; width: 160px; }
figure img { width: 160px; aspect-ratio: 16 / 9; object-fit: cover; border-radius: 4px; display: block;
  background: var(--chip); }
figure.key img { outline: 2px solid var(--accent); outline-offset: -2px; }
figcaption { font-size: 11px; color: var(--muted); }
button.gap { width: 12px; height: 90px; padding: 0; border: 0; background: transparent; cursor: col-resize;
  position: relative; }
button.gap::after { content: ""; position: absolute; left: 5px; top: 6px; bottom: 6px; width: 2px; border-radius: 1px; }
button.gap:hover::after, button.gap:focus-visible::after { background: var(--line); }
button.gap.cut::after { left: 3px; width: 6px; background: var(--warn); }
textarea { width: 100%; margin-top: 8px; min-height: 2.4em; font: inherit; color: var(--fg); background: var(--bg);
  border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; resize: vertical; }
"""

SCRIPT = r"""
"use strict";
const data = JSON.parse(document.getElementById("data").textContent);
const KEY = "review_scenes:" + data.resultFile + ":" + data.runId;
let state = { merges: {}, cuts: {}, wrong: {}, notes: {} };
try {
  const saved = localStorage.getItem(KEY);
  if (saved) state = Object.assign(state, JSON.parse(saved));
} catch (error) {
  // No storage: marks last until the page closes.
}

function persist() {
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch (error) {
    // As above.
  }
  tally();
}

function two(n) {
  return String(n).padStart(2, "0");
}

function clock(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return (h ? h + ":" + two(m) : two(m)) + ":" + two(s % 60);
}

function rounded(t) {
  return Math.round(t * 1000) / 1000;
}

function percent(x) {
  return Math.round(x * 100) + "%";
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function chip(scene, kind, tag, label, num, title, cls) {
  const key = [kind, tag, scene.start, scene.end].join("|");
  const node = el("button", "chip " + cls, label);
  node.type = "button";
  if (num) node.append(el("span", "num", num));
  node.title = title + " (click if wrong)";
  const sync = () => {
    const on = key in state.wrong;
    node.classList.toggle("struck", on);
    node.setAttribute("aria-pressed", String(on));
  };
  node.addEventListener("click", () => {
    if (key in state.wrong) delete state.wrong[key];
    else state.wrong[key] = { tag: tag, kind: kind, start: scene.start, end: scene.end };
    sync();
    persist();
  });
  sync();
  return node;
}

function card(scene) {
  const section = el("section", "scene");
  section.id = "scene-" + scene.index;
  const head = el("div", "head");
  head.append(el("h2", null, "Scene " + (scene.index + 1)));
  const link = scene.boundary && scene.boundary.link != null ? " · link " + scene.boundary.link.toFixed(2) : "";
  head.append(
    el(
      "span",
      "meta",
      clock(scene.start) + " – " + clock(scene.end) + " · " + Math.round(scene.duration) + " s · " +
        scene.frames.length + " frames" + link,
    ),
  );
  if (scene.index > 0) {
    const label = el("label", "merge");
    const box = el("input");
    box.type = "checkbox";
    const t = scene.start;
    box.checked = String(t) in state.merges;
    section.classList.toggle("merged", box.checked);
    box.addEventListener("change", () => {
      if (box.checked) state.merges[t] = t;
      else delete state.merges[t];
      section.classList.toggle("merged", box.checked);
      persist();
    });
    label.append(box, " merge with previous");
    head.append(label);
  }
  section.append(head);

  const chips = el("div", "chips");
  const rating = scene.rating;
  if (rating) {
    const score = rating.scores[rating.label];
    chips.append(chip(scene, "rating", rating.label, rating.label, score != null ? percent(score) : "",
      "The scene's rating, its mean score", "rating"));
    for (const peak of rating.peaks || []) {
      chips.append(chip(scene, "rating", peak.label, "⚠ " + peak.label, peak.score.toFixed(2) + " @" +
        clock(peak.timestamp), "A more severe rating's peak", "peak"));
    }
  }
  for (const tag of scene.tags || []) {
    chips.append(chip(scene, "tag", tag.name, tag.name, percent(tag.coverage) + " · " + tag.score.toFixed(2),
      "On for " + percent(tag.coverage) + " of the scene, mean score " + tag.score.toFixed(2), tag.category));
  }
  for (const hl of scene.highlights || []) {
    chips.append(chip(scene, "highlight", hl.name, "★ " + hl.name, hl.score.toFixed(2) + " @" + clock(hl.timestamp),
      "Highlight: peak " + hl.score.toFixed(2) + ", " + hl.frames + " frame(s) at the repeat bar", "highlight"));
  }
  section.append(chips);

  const thumbs = el("div", "thumbs");
  scene.frames.forEach((f, i) => {
    if (i) {
      const before = data.frames[scene.frames[i - 1]];
      const t = rounded((before.t + data.frames[f].t) / 2);
      const gap = el("button", "gap");
      gap.type = "button";
      gap.title = "A scene starts at " + clock(t) + "?";
      gap.setAttribute("aria-label", gap.title);
      const sync = () => {
        const on = String(t) in state.cuts;
        gap.classList.toggle("cut", on);
        gap.setAttribute("aria-pressed", String(on));
      };
      gap.addEventListener("click", () => {
        if (String(t) in state.cuts) delete state.cuts[t];
        else state.cuts[t] = t;
        sync();
        persist();
      });
      sync();
      thumbs.append(gap);
    }
    const figure = el("figure", scene.keyframe && scene.keyframe.frame === f ? "key" : null);
    const img = el("img");
    img.loading = "lazy";
    img.alt = "Frame at " + clock(data.frames[f].t);
    img.src = data.frames[f].src;
    figure.append(img, el("figcaption", null, clock(data.frames[f].t)));
    thumbs.append(figure);
  });
  section.append(thumbs);

  const noteKey = scene.start + "|" + scene.end;
  const note = el("textarea");
  note.placeholder = "Note (a missing tag, anything else)";
  note.value = state.notes[noteKey] ? state.notes[noteKey].text : "";
  note.addEventListener("input", () => {
    if (note.value.trim()) state.notes[noteKey] = { start: scene.start, end: scene.end, text: note.value };
    else delete state.notes[noteKey];
    persist();
  });
  section.append(note);
  return section;
}

function review() {
  const times = (marks) => Object.values(marks).sort((a, b) => a - b);
  return {
    video: data.video,
    settings: data.settings,
    boundaries: data.boundaries,
    merges: times(state.merges),
    cuts: times(state.cuts),
    wrong: Object.values(state.wrong).sort((a, b) => a.start - b.start),
    notes: Object.values(state.notes).sort((a, b) => a.start - b.start),
  };
}

function save() {
  const blob = new Blob([JSON.stringify(review(), null, 2) + "\n"], { type: "application/json" });
  const a = el("a");
  a.href = URL.createObjectURL(blob);
  a.download = data.name + ".review.json";
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

let tallyNode;
function tally() {
  const counts = [
    [Object.keys(state.merges).length, "merge", "merges"],
    [Object.keys(state.cuts).length, "cut", "cuts"],
    [Object.keys(state.wrong).length, "struck", "struck"],
    [Object.keys(state.notes).length, "note", "notes"],
  ];
  tallyNode.textContent = counts.map(([n, one, many]) => n + " " + (n === 1 ? one : many)).join(" · ");
}

function goTo(t) {
  const scene = data.scenes.find((s, k) => (s.start <= t && t < s.end) || k === data.scenes.length - 1);
  const node = scene && document.getElementById("scene-" + scene.index);
  if (!node) return;
  node.scrollIntoView({ behavior: "smooth", block: "start" });
  node.classList.add("flash");
  setTimeout(() => node.classList.remove("flash"), 1200);
}

function earlierPanel(earlier) {
  const aside = el("aside");
  aside.append(el("h2", null, "Since your last review"));
  let line;
  if (earlier.added == null) line = "The saved review has no boundaries to compare with.";
  else {
    const n = earlier.added.length + earlier.gone.length;
    line = n + " boundar" + (n === 1 ? "y" : "ies") + " changed since your last review (" + earlier.added.length +
      " new, " + earlier.gone.length + " gone).";
  }
  const open = earlier.comments.filter((c) => c.status === "open").length;
  const done = earlier.comments.filter((c) => c.status === "addressed").length;
  aside.append(el("p", null, line + " " + done + " addressed, " + open + " still open."));
  const list = el("ol");
  for (const c of earlier.comments) {
    const item = el("li");
    const status = c.status === "note" ? "earlier note" : c.status === "open" ? "still open" : "addressed";
    item.append(el("span", "badge " + c.status, status));
    let text;
    let at;
    if (c.kind === "merge") {
      text = "Merge at " + clock(c.t);
      at = c.t;
    } else if (c.kind === "cut") {
      text = "Scene starts at " + clock(c.t);
      at = c.t;
    } else if (c.kind === "wrong") {
      text = "Wrong " + c.what + " “" + c.tag + "” (" + clock(c.start) + " – " + clock(c.end) + ")";
      at = (c.start + c.end) / 2;
    } else {
      text = clock(c.start) + " – " + clock(c.end) + ": " + c.text;
      at = (c.start + c.end) / 2;
    }
    item.append(document.createTextNode(text));
    item.addEventListener("click", () => goTo(at));
    list.append(item);
  }
  aside.append(list);
  return aside;
}

function render() {
  const top = document.getElementById("top");
  const title = el("div");
  title.append(el("h1", null, data.name));
  title.append(
    el(
      "div",
      "sub",
      data.scenes.length + " scenes · " + data.frameCount + " frames · " + clock(data.duration) + " · " +
        data.tagger + (data.threshold != null ? " · link threshold " + data.threshold.toFixed(2) : "") +
        " · click what is wrong; everything else counts as right",
    ),
  );
  top.append(title);
  tallyNode = el("span", "tally");
  top.append(tallyNode);
  const clear = el("button", "action", "Clear marks");
  clear.type = "button";
  clear.addEventListener("click", () => {
    if (!confirm("Clear every mark and note on this page?")) return;
    state = { merges: {}, cuts: {}, wrong: {}, notes: {} };
    persist();
    location.reload();
  });
  const saveButton = el("button", "action primary", "Save review");
  saveButton.type = "button";
  saveButton.addEventListener("click", save);
  top.append(clear, saveButton);
  if (data.warnings.length) top.append(el("div", "warnings", data.warnings.join(" · ")));

  const main = document.getElementById("scenes");
  for (const scene of data.scenes) main.append(card(scene));
  if (data.earlier) {
    const layout = document.getElementById("layout");
    layout.classList.add("with-aside");
    layout.append(earlierPanel(data.earlier));
  }
  tally();
}

render();
"""


# --- Main.


def review_one(worker: TagScenes, path: Path, settings: Mapping, constants: Mapping, out: Path) -> list[Path]:
    """TagScenes on one input, then its result and page written to `out`. Prints times, counts and paths only."""
    roots = roots_of([path])
    ctx = _Context(roots)
    source = read_frames(str(path), ctx.allowed_path)
    name = name_of(path, source)
    print(f"{name}: {len(source.frames)} frames", flush=True)
    with overridden(constants):
        result = worker.run(_Task(path, settings, roots), ctx)
        used = effective(settings, constants)
    print("\r", end="", file=sys.stderr, flush=True)

    out.mkdir(parents=True, exist_ok=True)
    result_file = out / f"{name}.result.json"
    result_file.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    review_file = out / f"{name}.review.json"
    earlier = None
    if review_file.is_file():
        earlier = since_last(json.loads(review_file.read_text(encoding="utf-8")), result)
    times = [frame.timestamp for frame in source.frames]
    page_file = out / f"{name}.review.html"
    page_file.write_text(page(name, result, times, thumbnails(source), used, earlier), encoding="utf-8")

    starts = ", ".join(_clock(scene["start"]) for scene in result["scenes"])
    print(f"  {result['count']} scenes over {_clock(result['duration'])}, starting at {starts}")
    if earlier is not None:
        statuses = [comment["status"] for comment in earlier["comments"]]
        changed = "?" if earlier["added"] is None else len(earlier["added"]) + len(earlier["gone"])
        print(
            f"  since the last review: {changed} boundaries changed; {statuses.count('addressed')} comments addressed, "
            f"{statuses.count('open')} still open, {statuses.count('note')} notes"
        )
    print(f"  {len(result.get('warnings', []))} warnings")
    print(f"  wrote {result_file}")
    print(f"  wrote {page_file}")
    return [result_file, page_file]


def _clock(seconds: float) -> str:
    s = max(0, round(seconds))
    h, m = divmod(s, 3600)
    return (f"{h}:{m // 60:02}" if h else f"{m // 60:02}") + f":{s % 60:02}"


def model_dir(tagger: str) -> str | None:
    """`MODELS` when the tagger is in it, so nothing downloads."""
    entry = TAGGERS.get(tagger)
    return str(MODELS) if entry is not None and (MODELS / entry.local_path).is_file() else None


def main(argv: Sequence[str] | None = None) -> list[Path]:
    # File names are printed as they are; a Windows console's code page cannot spell every one.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0].rstrip(":"))
    parser.add_argument("inputs", type=Path, nargs="+", help="folders of timestamped frames, or frame sheets")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="NAME=VALUE",
                        help="a setting (camelCase) or a constant of scenes.py or BLANK_STD (UPPER_CASE)")  # fmt: skip
    parser.add_argument("--tagger", help=f"the tagger (default {DEFAULT_TAGGER}); as --set tagger=NAME")
    parser.add_argument("--out", type=Path, default=REVIEWS, help="where the result and page go")
    parser.add_argument("--data-dir", type=Path, default=DATA, help="where the Worker downloads a tagger to")
    parser.add_argument("--model-dir", type=Path, help=f"a modelDir holding the tagger (default {MODELS} if it does)")
    args = parser.parse_args(argv)

    try:
        settings, constants = parse_sets(args.sets)
    except ValueError as error:
        parser.error(str(error))
    if args.tagger:
        settings["tagger"] = args.tagger
    tagger = settings.get("tagger", DEFAULT_TAGGER)
    inputs = [path.resolve() for path in args.inputs]
    for path in inputs:
        if not path.exists():
            parser.error(f"no such folder or sheet: {path}")

    config = {"host": {"dataDir": str(args.data_dir)}}
    found = str(args.model_dir) if args.model_dir else model_dir(tagger)
    if found:
        config["modelDir"] = found
    worker = TagScenes(config)
    written = []
    try:
        for path in inputs:
            written += review_one(worker, path, settings, constants, args.out)
    finally:
        worker.teardown()
    return written


if __name__ == "__main__":
    main()
