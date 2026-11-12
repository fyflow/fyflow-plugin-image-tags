"""Measures the classifiers on labelled videos and images: `pnpm eval -- <csv> [--collage <thumbs-collage>]`.

Run by hand before a release is tagged; never by `pnpm check`. It measures `style` as shipped (its rule over
`anime_classification` v1.5, at its default `minConfidence`) beside the configuration it replaced (`previous`: live
action `real`, `anime_classification` v1.3, 0.6) and a candidate rule, so a run shows what a change would change.

The CSV holds `path,label` rows (a `path,label` header is optional; relative paths are from the CSV's folder), each
label one of anime, cg, live_action. A path may be:

- a video: the `thumbs-collage` service (`--collage`) makes its frame sheet with the README's recommended
  `VideoThumbnail` settings (the request `VideoThumbnail` sends), written to `python/.cache/eval/sheets/`;
- an image, or a folder of images: classified as it is, as `ClassifyFrames` does.

`--references` adds the reference frames, `python/tests/fixtures/images/<label>-NN-*.jpg`, each labelled by its name;
with it the CSV may be left out.

Each item goes through `ClassifyFrames` itself (the cpu build, `style`, timed), and each frame through every model
by the Worker's own code (`_score`), `anime_classification` in both versions. The rules below are worked out
from those frames' scores, as `ClassifyFrames` would: per frame, then the mean over the frames. Models download into
`python/.cache/eval/models/` (gitignored), from Hugging Face or `HF_ENDPOINT`.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

SCRIPTS = Path(__file__).resolve().parent
PYTHON = SCRIPTS.parent
sys.path[:0] = [str(PYTHON), str(SCRIPTS)]
import sheets  # noqa: E402
from image_tags.catalogue import ANIME_CLASSIFICATION, ANIME_REAL, PAINTED_FLOOR, STYLE, UNCERTAIN  # noqa: E402
from image_tags.catalogue import style as shipped_style  # noqa: E402
from image_tags.frames import read_frames  # noqa: E402
from image_tags.worker import ClassifyFrames  # noqa: E402

LABELS = STYLE.labels
CACHE = PYTHON / ".cache" / "eval"
REFERENCES = PYTHON / "tests" / "fixtures" / "images"

#: The version the catalogue had before the eval (2026-11-10) settled on v1.5, from the same commit: the same labels,
#: input and output. Held here, not in the catalogue, to compare against.
ANIME_CLASSIFICATION_V13 = replace(
    ANIME_CLASSIFICATION,
    id="anime_classification_v1.3",
    name="mobilenetv3_v1.3_dist",
    sha256="fa233c04467cba4c68ecabd0065ad169c6fe867dc19799627cf5bd566a97d115",
)
#: `anime_classification`'s versions, compared side by side: the catalogue's first.
VERSIONS = {"v1.5": ANIME_CLASSIFICATION, "v1.3": ANIME_CLASSIFICATION_V13}
RAW_MODELS = (ANIME_REAL, ANIME_CLASSIFICATION, ANIME_CLASSIFICATION_V13)
#: The configuration `style` shipped with before the eval: the `previous` rule over v1.3, at this `minConfidence`.
PREVIOUS_VERSION = "v1.3"
PREVIOUS_MIN_CONFIDENCE = 0.6

#: A raw model's labels in `style`'s terms. `anime_real` has no `cg`: its `cg` row is shown (where CG goes is the
#: question) but left out of its accuracy.
MAPPINGS = {
    "anime_real": {"anime": "anime", "real": "live_action"},
    "anime_classification": {
        "3d": "cg",
        "bangumi": "anime",
        "comic": "anime",
        "illustration": "anime",
        "not_painting": "live_action",
    },
}
EXPRESSIBLE = {"anime_real": ("anime", "live_action"), "anime_classification": LABELS}


# --- `style` rules besides the shipped one, eval only: each from the frames' scores keyed `anime_real` and
# --- `anime_classification` (whichever version) to `[n, 3]` anime / cg / live_action, summing to 1.


def _real(scores):
    return scores[ANIME_REAL.id][:, ANIME_REAL.labels.index("real")]


def _kind(scores, label):
    return scores[ANIME_CLASSIFICATION.id][:, ANIME_CLASSIFICATION.labels.index(label)]


def previous(scores):
    """The rule `style` shipped with before the eval: `live_action` is `real` itself; what is not real is shared out by
    the four painted kinds, as the shipped rule does."""
    real = _real(scores)
    painted = sum(_kind(scores, label) for label in ("3d", "bangumi", "comic", "illustration"))
    told = painted >= PAINTED_FLOOR
    share = np.where(told, painted, 1.0)
    cg = np.where(told, _kind(scores, "3d") / share, 0.5)
    cartoon = _kind(scores, "bangumi") + _kind(scores, "illustration") + _kind(scores, "comic")
    anime = np.where(told, cartoon / share, 0.5)
    return np.stack([(1 - real) * anime, (1 - real) * cg, real], axis=1)


def three_d_first(scores):
    """A strong `3d` overrides `real`: `cg` is `3d`'s raw score (of all five); what is left is split by `anime_real`."""
    cg = _kind(scores, "3d")
    real = _real(scores)
    return np.stack([(1 - cg) * (1 - real), cg, (1 - cg) * real], axis=1)


RULES: dict[str, Callable[[Mapping[str, np.ndarray]], np.ndarray]] = {
    "shipped": shipped_style,
    "previous": previous,
    "3d_first": three_d_first,
}

#: A sheet's cell, width and height: as wide as the models' input.
CELL = (384, 216)

#: The thresholds the table shows; the top label of three is never below 1/3.
THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.35, 0.96, 0.05))


# --- What is evaluated.


@dataclass
class Item:
    path: Path
    label: str
    #: video, image or folder
    kind: str


@dataclass
class Scored:
    item: Item
    frames: int = 0
    sheet_seconds: float | None = None
    classify_seconds: float = 0.0
    #: Each raw model's scores per frame, `[n, len(labels)]`, by model id.
    raw: dict[str, np.ndarray] = field(default_factory=dict)
    error: str | None = None


def kind_of(path: Path) -> str:
    if path.is_dir():
        return "folder"
    try:
        with Image.open(path):
            return "image"
    except (UnidentifiedImageError, OSError):
        return "video"


def read_csv(path: Path) -> list[Item]:
    items = []
    with open(path, newline="", encoding="utf-8-sig") as file:
        for number, row in enumerate(csv.reader(file), 1):
            row = [cell.strip() for cell in row]
            if not any(row) or row[0].startswith("#") or (number == 1 and row[:2] == ["path", "label"]):
                continue
            if len(row) < 2 or row[1] not in LABELS:
                sys.exit(f"{path}:{number}: not `path,label` with a label of {', '.join(LABELS)}: {row}")
            item_path = (path.parent / row[0]).resolve()
            if not item_path.exists():
                sys.exit(f"{path}:{number}: no such file or folder: {item_path}")
            items.append(Item(item_path, row[1], kind_of(item_path)))
    return items


def references(folder: Path = REFERENCES) -> list[Item]:
    """The reference frames, `<label>-NN-<what>.jpg`, each labelled by its name."""
    items = []
    for path in sorted(folder.glob("*-[0-9][0-9]-*.jpg")):
        label = path.name.split("-", 1)[0]
        if label in LABELS:
            items.append(Item(path, label, "image"))
    return items


# --- Sums.


def confusion(truths: Sequence[str], predictions: Sequence[str], rows: Sequence[str], columns: Sequence[str]):
    """Counts, `[len(rows), len(columns)]`: how many items of each truth got each prediction."""
    matrix = np.zeros((len(rows), len(columns)), dtype=int)
    for truth, prediction in zip(truths, predictions, strict=True):
        matrix[rows.index(truth), columns.index(prediction)] += 1
    return matrix


def tags(tops: Sequence[str], confidences: Sequence[float], threshold: float) -> list[str]:
    """Each item's tag: its top label when its confidence reaches `threshold`, else `uncertain`."""
    return [top if confidence >= threshold else UNCERTAIN for top, confidence in zip(tops, confidences, strict=True)]


@dataclass(frozen=True)
class Row:
    threshold: float
    total: int
    answered: int
    correct: int

    @property
    def coverage(self) -> float:
        return self.answered / self.total if self.total else 0.0

    @property
    def accuracy(self) -> float:
        """Accuracy on what is answered (not `uncertain`)."""
        return self.correct / self.answered if self.answered else 0.0


def threshold_table(truths, tops, confidences, thresholds=THRESHOLDS) -> list[Row]:
    rows = []
    for threshold in thresholds:
        answered = [(truth, tag) for truth, tag in zip(truths, tags(tops, confidences, threshold)) if tag != UNCERTAIN]
        correct = sum(truth == tag for truth, tag in answered)
        rows.append(Row(threshold, len(truths), len(answered), correct))
    return rows


def best_threshold(table: Sequence[Row], min_coverage: float) -> Row | None:
    """The threshold with the best accuracy on what is answered among those answering at least `min_coverage` of the
    items; of equals, the lowest (it answers the most)."""
    allowed = [row for row in table if row.answered and row.coverage >= min_coverage]
    return max(allowed, key=lambda row: (row.accuracy, -row.threshold), default=None)


def mapped(model_id: str, label: str) -> str:
    return MAPPINGS[model_id][label]


# --- Running it.


class _Task:
    def __init__(self, path, settings):
        self.input = {"path": str(path)}
        self.settings = {"profile": {}, **settings}


class _Context:
    """A Task's context, without the Host's check of Allowed roots: the eval is a local tool, reading what its CSV
    names."""

    cancelled = False

    def progress(self, fraction, message=None):
        pass

    def log(self, message):
        pass

    @staticmethod
    def allowed_path(path, access="read"):
        return os.path.realpath(path)


def make_sheet(url: str, video: Path, args) -> tuple[Path, float]:
    """The video's frame sheet, as `VideoThumbnail` asks for it with the README's recommended settings, and how long
    the service took."""
    grid = args.grid
    offsets = {"start_offset": args.start_offset, "end_offset": args.end_offset}
    selection = {"mode": "uniform", "frames": grid * grid, **offsets}
    width, height = CELL[0] * grid, CELL[1] * grid
    collage = {"mode": "grid", "rows": grid, "cols": grid, "spacing": 0, "margin": 0, "width": width, "height": height}
    body = sheets.request(path=str(video), selection=selection, collage=collage)
    key = hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:10]
    out = CACHE / "sheets" / f"{video.stem}-{key}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    post = urllib.request.Request(
        f"{url}/make_collage", data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(post, timeout=600) as response:
            out.write_bytes(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"the service answered {error.code}: {error.read().decode(errors='replace')}") from None
    return out, time.perf_counter() - start


def score(worker: ClassifyFrames, frames_path: Path, scored: Scored) -> None:
    """`ClassifyFrames` on the item (timed), then every raw model on its frames by the Worker's own code."""
    start = time.perf_counter()
    result = worker.run(_Task(frames_path, {"classifier": STYLE.id, "minConfidence": 0}), _Context())
    scored.classify_seconds = time.perf_counter() - start

    images = [frame.load() for frame in read_frames(str(frames_path), _Context.allowed_path).frames]
    scored.frames = len(images)
    for model in RAW_MODELS:
        scored.raw[model.id] = np.concatenate(
            [
                worker._score(model, images[start : start + worker.batch_size], None)
                for start in range(0, len(images), worker.batch_size)
            ]
        )
    # The same engine: the shipped rule on these frames gives what ClassifyFrames gave.
    mine = shipped_style(scores_of(scored, "v1.5")).mean(axis=0)
    theirs = np.array([result["scores"][label] for label in LABELS])
    if not np.allclose(mine, theirs, atol=1e-4):
        raise RuntimeError(f"ClassifyFrames gave {theirs}, the eval's own run {mine}")


def scores_of(scored: Scored, version: str) -> dict[str, np.ndarray]:
    """The frames' scores as a rule reads them, `anime_classification` in `version`."""
    return {ANIME_REAL.id: scored.raw[ANIME_REAL.id], ANIME_CLASSIFICATION.id: scored.raw[VERSIONS[version].id]}


# --- Printing.


def percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


def print_matrix(title: str, matrix: np.ndarray, rows: Sequence[str], columns: Sequence[str]) -> None:
    width = max(len(column) for column in [*columns, "truth \\ said"]) + 2
    print(f"\n{title}")
    print("truth \\ said".ljust(14) + "".join(column.rjust(width) for column in columns))
    for row, counts in zip(rows, matrix):
        print(row.ljust(14) + "".join(str(count).rjust(width) for count in counts))


def accuracy_line(truths, predictions, among) -> str:
    pairs = [(truth, said) for truth, said in zip(truths, predictions) if truth in among]
    correct = sum(truth == said for truth, said in pairs)
    return f"accuracy over {', '.join(among)}: {correct}/{len(pairs)} ({percent(correct, len(pairs))})"


def anime_vs_rest(truths: list[str], combos: dict[str, tuple[list[str], list[float]]]) -> None:
    """Anime against everything else, the question that matters most: per rule and threshold, how many anime items are
    tagged anime (found), and how many items that are not anime are (false). Tagged anime: anime is the item's top
    label (over the mean of its frames) and reaches the threshold."""
    anime = truths.count("anime")
    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    print(f"\n--- Anime vs the rest: anime found (of {anime}) / not anime but tagged anime (of {len(truths) - anime})")
    print("rule".ljust(18) + "".join(f"{thr:>10}" for thr in thresholds))
    for name, (tops, confidences) in combos.items():
        cells = []
        for thr in thresholds:
            tagged = [top == "anime" and confidence >= thr for top, confidence in zip(tops, confidences)]
            found = sum(tag and truth == "anime" for tag, truth in zip(tagged, truths))
            false = sum(tag and truth != "anime" for tag, truth in zip(tagged, truths))
            cells.append(f"{found}/{false}")
        print(name.ljust(18) + "".join(f"{cell:>10}" for cell in cells))


def report(results: list[Scored], min_coverage: float) -> None:
    done = [scored for scored in results if scored.error is None]
    truths = [scored.item.label for scored in done]
    counts = ", ".join(f"{label} {truths.count(label)}" for label in LABELS)
    print(f"\n=== {len(done)} items ({counts}); {len(results) - len(done)} failed")
    if not done:
        return

    # Each rule and version: the mean over the frames of the rule's scores per frame, as ClassifyFrames answers.
    combos = {}
    for rule_name, rule in RULES.items():
        for version in VERSIONS:
            means = np.stack([rule(scores_of(scored, version)).mean(axis=0) for scored in done])
            tops = [LABELS[i] for i in means.argmax(axis=1)]
            combos[f"{rule_name} {version}"] = (tops, means.max(axis=1).tolist())

    default = STYLE.min_confidence
    for title, name, threshold in [
        ("style as shipped (anime_classification v1.5)", "shipped v1.5", default),
        (f"style as it was ({PREVIOUS_VERSION})", f"previous {PREVIOUS_VERSION}", PREVIOUS_MIN_CONFIDENCE),
    ]:
        tops, confidences = combos[name]
        print_matrix(
            f"--- {title}, minConfidence {threshold}",
            confusion(truths, tags(tops, confidences, threshold), LABELS, [*LABELS, UNCERTAIN]),
            LABELS,
            [*LABELS, UNCERTAIN],
        )

    anime_vs_rest(truths, combos)

    print("\n--- Raw models: each item's top label from the mean over its frames")
    for model in RAW_MODELS:
        family = ANIME_REAL.id if model is ANIME_REAL else ANIME_CLASSIFICATION.id
        said = [model.labels[int(scored.raw[model.id].mean(axis=0).argmax())] for scored in done]
        print_matrix(model.id, confusion(truths, said, LABELS, model.labels), LABELS, model.labels)
        mapping = ", ".join(f"{label}->{to}" for label, to in MAPPINGS[family].items())
        print(f"as style ({mapping}): {accuracy_line(truths, [mapped(family, s) for s in said], EXPRESSIBLE[family])}")

    print(f"\n--- Rules x anime_classification version, at minConfidence {default}: tags per truth")
    print("rule".ljust(18) + "".join(label.rjust(28) for label in LABELS) + "  acc answered / coverage")
    for name, (tops, confidences) in combos.items():
        said = tags(tops, confidences, default)
        cells = []
        for label in LABELS:
            mine = [s for t, s in zip(truths, said) if t == label]
            cells.append(" ".join(f"{s[:2]}{mine.count(s)}" for s in [*LABELS, UNCERTAIN] if mine.count(s)) or "-")
        row = threshold_table(truths, tops, confidences, [default])[0]
        print(
            name.ljust(18)
            + "".join(cell.rjust(28) for cell in cells)
            + f"  {percent(row.correct, row.answered)} / {percent(row.answered, row.total)}"
        )
    print("(an:anime cg:cg li:live_action un:uncertain; count of items of that truth tagged so)")

    print(f"\n--- minConfidence: accuracy on answered / coverage, per threshold (pick: best accuracy, coverage >= "
          f"{min_coverage:.0%}, lowest of equals)")
    tables = {name: threshold_table(truths, *combo) for name, combo in combos.items()}
    print("thr".ljust(6) + "".join(name.rjust(16) for name in tables))
    for i, threshold in enumerate(THRESHOLDS):
        cells = [f"{percent(t[i].correct, t[i].answered)}/{percent(t[i].answered, t[i].total)}" for t in tables.values()]
        print(f"{threshold:<6}" + "".join(cell.rjust(16) for cell in cells))
    picks = []
    for name, table in tables.items():
        best = best_threshold(table, min_coverage)
        picks.append(f"{best.threshold} ({best.accuracy:.0%}/{best.coverage:.0%})" if best else "none")
    print("pick".ljust(6) + "".join(pick.rjust(16) for pick in picks))

    print("\n--- Time per item (s): making the sheet, then ClassifyFrames (style, cpu build, models loaded)")
    for scored in done:
        sheet = f"{scored.sheet_seconds:.2f}" if scored.sheet_seconds is not None else "-"
        print(f"{sheet:>7} {scored.classify_seconds:>7.2f}  {scored.frames:>3} frames  {scored.item.path.name}")
    videos = [s for s in done if s.sheet_seconds is not None]
    if videos:
        print(
            f"videos: mean {np.mean([s.sheet_seconds for s in videos]):.2f} s sheet, "
            f"{np.mean([s.classify_seconds for s in videos]):.2f} s classifying"
        )

    print(f"\n--- Misclassified under any rule (tag@confidence at minConfidence {default}; * wrong)")
    names = list(combos)
    print("item".ljust(44) + "truth".ljust(12) + "".join(name.rjust(22) for name in names))
    for i, scored in enumerate(done):
        cells = []
        for name in names:
            tops, confidences = combos[name]
            tag = tags([tops[i]], [confidences[i]], default)[0]
            cells.append(f"{'*' if tag != truths[i] else ''}{tag}@{confidences[i]:.2f}")
        if any(cell.startswith("*") for cell in cells):
            print(scored.item.path.name[:43].ljust(44) + truths[i].ljust(12) + "".join(c.rjust(22) for c in cells))
    for scored in results:
        if scored.error is not None:
            print(f"FAILED {scored.item.path}: {scored.error}")


def main():
    # File names are printed as they are; a Windows console's code page cannot spell every one.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, nargs="?", help="path,label rows: videos, images or folders")
    parser.add_argument("--references", action="store_true", help="add the reference frames, labelled by name")
    parser.add_argument("--collage", type=Path, help="the thumbs-collage executable (needed for videos)")
    parser.add_argument("--grid", type=int, default=3, help="rows = cols of the sheet (README: 3 to 4)")
    parser.add_argument("--start-offset", type=float, default=0.5, help="seconds skipped at the start (openings)")
    parser.add_argument("--end-offset", type=float, default=0.5, help="seconds skipped at the end (credits)")
    parser.add_argument("--min-coverage", type=float, default=0.9, help="for the minConfidence pick")
    argv = sys.argv[1:]
    args = parser.parse_args(argv[1:] if argv[:1] == ["--"] else argv)  # `pnpm eval -- …` hands on the `--`

    items = (read_csv(args.csv) if args.csv else []) + (references() if args.references else [])
    if not items:
        parser.error("nothing to evaluate: give a CSV, --references, or both")
    videos = [item for item in items if item.kind == "video"]
    if videos and (args.collage is None or not args.collage.is_file()):
        parser.error(f"{len(videos)} videos need --collage <thumbs-collage executable>")

    worker = ClassifyFrames({"host": {"dataDir": str(CACHE)}})
    for model in RAW_MODELS:  # downloaded and loaded before anything is timed
        worker.models.session(model)

    service = url = None
    if videos:
        service, url = sheets.started(args.collage.resolve())
    results = []
    try:
        for number, item in enumerate(items, 1):
            print(f"[{number}/{len(items)}] {item.kind} {item.path}", file=sys.stderr, flush=True)
            scored = Scored(item)
            try:
                frames_path = item.path
                if item.kind == "video":
                    frames_path, scored.sheet_seconds = make_sheet(url, item.path, args)
                score(worker, frames_path, scored)
            except Exception as error:  # noqa: BLE001  (one bad item is reported, not fatal)
                scored.error = f"{type(error).__name__}: {error}"
            results.append(scored)
    finally:
        if service is not None:
            service.stdin.close()
            try:
                service.wait(5)
            except Exception:  # noqa: BLE001
                service.kill()
    report(results, args.min_coverage)


if __name__ == "__main__":
    main()
