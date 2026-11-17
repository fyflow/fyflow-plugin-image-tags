"""The Plugin's Python Workers, as the Host builds them: one instance per Python process, `run` once per Task."""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .batcher import Batcher
from .catalogue import (
    CLASSIFIERS,
    DEFAULT_CLASSIFIER,
    DEFAULT_TAGGER,
    MODELS,
    TAGGERS,
    Classifier,
    Labels,
    Model,
    Tagger,
    answer,
    tag,
    thresholds_of,
)
from .devices import device_of
from .frames import SHEET_AS_ONE_IMAGE, Frames, read_frames, sheet_of
from .models import Cancelled, Models
from .preprocess import Preprocessing, open_image, to_input, to_rgb

#: The most frames in one forward pass when the Profile sets no `batchSize`: a `ClassifyImage` batch, a
#: `ClassifyFrames` Task's chunk.
DEFAULT_BATCH_SIZE = 16
#: The most images in one forward pass of a tagger when the Profile sets no `tagBatchSize`: a tagger is some 70 times
#: a classifier's size, so its batches are smaller.
DEFAULT_TAG_BATCH_SIZE = 4
#: The `cpu` units each process holds while it lives, when the Host does not say (before SDK 0.11): its pools'
#: `residentGroups` in `plugin.ts`. A forward pass runs on as many threads.
CPU_UNITS = 4
#: The most `ClassifyImage` Tasks a process runs at once, when the Host does not say: its pool's `maxConcurrentTasks` in
#: `plugin.ts`.
TASKS_AT_ONCE = 32
#: The most `TagImage` Tasks a process runs at once, when the Host does not say: its pool's `maxConcurrentTasks` in
#: `plugin.ts`. Enough to keep a few batches lined up.
TAG_TASKS_AT_ONCE = 16


def ran_on_the_cpu(tagger: Tagger) -> str:
    """The warning of a tagger's result when its session runs on the CPU."""
    return (
        f"{tagger.id} ran on the CPU, where a tagger is slow: a GPU build (`cuda` or `directml`) tags many times "
        "faster"
    )


def _pool_options(config) -> dict:
    """The pool options the process runs under, as the Host gives them (since SDK 0.11): the Node Type's, with its
    Profile's over them. Empty from an older Host."""
    return config["host"].get("pool") or {}


def _cpu_units(config) -> int:
    """The `cpu` units fyflow booked for the process, else `CPU_UNITS`."""
    cpu = (_pool_options(config).get("residentGroups") or {}).get("cpu") or {}
    return cpu.get("cost") or CPU_UNITS


def _tasks_at_once(config, default: int) -> int:
    """The most Tasks the process runs at once, else `default`."""
    return _pool_options(config).get("maxConcurrentTasks") or default


def _frame_by_frame(
    source: Frames, batch_size: int, score: Callable[[list[Image.Image]], np.ndarray], verbs: tuple[str, str], ctx
) -> np.ndarray:
    """`score`'s rows for every frame of `source`, in order, `batch_size` frames at a time, each batch's frames read
    only when it comes. The Task says how far it has got before the first batch and after each (`verbs`: what it is
    doing, then what it has done, as "Tagging" and "Tagged"), and a Cancel stops it before its next batch."""
    doing, did = verbs
    total = len(source.frames)
    ctx.progress(0.0, f"{doing} 0 of {total} frames")
    rows = []
    for start in range(0, total, batch_size):
        if getattr(ctx, "cancelled", False):
            raise Cancelled(f"Cancelled after {start} of {total} frames")
        batch = source.frames[start : start + batch_size]
        rows.append(score([frame.load() for frame in batch]))
        done = start + len(batch)
        ctx.progress(done / total, f"{did} {done} of {total} frames")
    return np.concatenate(rows)


def _frames_result(path: str, source: Frames, fields: dict, warnings: list[str] | None = None) -> dict:
    """A frames Task's result: the path it was given, the sheet's video when it names one, `fields`, then the
    warnings of reading the frames and `warnings`, when there are any."""
    result = {"path": path}
    if source.video is not None:
        result["video"] = source.video
    result.update(fields)
    if every := [*source.warnings, *(warnings or [])]:
        result["warnings"] = every
    return result


class _Worker:
    """What every Worker shares: the models, and how a Task names what it reads (`<what> to <verb>`)."""

    verb = "classify"
    what = "the image"

    def __init__(self, config, pool=None):
        # A Profile start-up field: `modelDir`, the folder of an offline machine's models (then nothing downloads).
        # On the GPU the process holds while it lives, when the variant's build has a GPU provider. The variant itself
        # (`host.variant`, since SDK 0.11) is only said on stderr: the providers are what the installed build offers.
        self.models = Models(
            config["host"]["dataDir"],
            model_dir=config.get("modelDir"),
            threads=_cpu_units(config),
            device=device_of(pool),
            variant=config["host"].get("variant"),
        )

    def _path(self, task) -> str:
        path = task.input.get("path") if isinstance(task.input, dict) else None
        if not isinstance(path, str) or path == "":
            raise ValueError(f"The input has no `path`: {self.what} to {self.verb}")
        return path

    def teardown(self):
        """The Worker ends (its process is about to): its models' sessions go. The shim may call it twice."""
        self.models.close()


class _Passes:
    """What the Tasks of an `async def run` Worker share: forward passes, one at a time in the process and off the
    event loop, in batches of up to `batch_size` per model (`batcher.py`); and threads for what blocks a Task.

    ONNX Runtime spreads a forward pass over the threads of the `cpu` units the process holds. What blocks a Task is
    reading its image, and getting each model, which may mean waiting on a download: `blocking_threads` threads, enough
    for every Task at once (the event loop's own executor, `asyncio.to_thread`'s, may have fewer), started only as they
    are needed."""

    def __init__(self, models: Models, batch_size: int, blocking_threads: int):
        self.batch_size = batch_size
        self.blocking_threads = blocking_threads
        self._models = models
        self._forward = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-tags-forward")
        self._blocking = ThreadPoolExecutor(blocking_threads, thread_name_prefix="image-tags-task")
        self._batchers: dict[str, Batcher] = {}

    def blocking(self, call, *args) -> asyncio.Future:
        """`call(*args)` on a thread of its own, awaitable."""
        return asyncio.get_running_loop().run_in_executor(self._blocking, call, *args)

    async def row(self, model: Model | Tagger, input: np.ndarray) -> np.ndarray:
        """`model`'s output for one input, from a batch shared with the other Tasks that need the model."""
        batcher = self._batchers.get(model.id)
        if batcher is None:
            batcher = self._batchers[model.id] = Batcher(model, self._models, self._forward, self.batch_size)
        return await batcher.score(input)

    def close(self) -> None:
        self._forward.shutdown(wait=False)
        self._blocking.shutdown(wait=False)


class _Classify(_Worker):
    """What both classifying Workers share: how a Task names its classifier, and `batchSize`."""

    def __init__(self, config, pool=None):
        super().__init__(config, pool)
        # A Profile start-up field: the most frames in one forward pass.
        self.batch_size = config.get("batchSize") or DEFAULT_BATCH_SIZE

    @staticmethod
    def _classifier(task) -> Classifier:
        name = task.settings.get("classifier") or DEFAULT_CLASSIFIER
        classifier = CLASSIFIERS.get(name)
        if classifier is None:
            raise ValueError(f"No classifier `{name}`: one of {', '.join(CLASSIFIERS)}")
        return classifier

    def _score(self, classifier: Classifier, images: list[Image.Image], ctx) -> np.ndarray:
        """The classifier's scores for each image, `[len(images), len(labels)]`, from one batch per model."""
        inputs = {}  # by preprocessing: models that take an image the same way take the same tensor
        scores = {}
        for model in classifier.models:
            if model.preprocessing not in inputs:
                inputs[model.preprocessing] = np.stack([to_input(image, model.preprocessing) for image in images])
            scores[model.id] = self.models.scores(model, inputs[model.preprocessing], ctx)
        return classifier.score(scores)


class ClassifyImage(_Classify):
    """One image per Task, through one classifier of the catalogue (the `classifier` setting): a model, or a preset
    over several. Its top label is the `tag` when it reaches `minConfidence` (the classifier's own when the setting is
    left empty), else `uncertain`. A frame sheet is scored as one image, with a warning. Its path must be in the Task's
    reach (Allowed roots), else the Task fails saying why.

    The Tasks of a process run at once on the shim's event loop: each reads its image on a thread, then hands it to
    each model's batcher (`batcher.py`), so Tasks running together share forward passes, whatever their classifier."""

    def __init__(self, config, pool=None):
        super().__init__(config, pool)
        # Each Task may wait on every model at once.
        self._passes = _Passes(self.models, self.batch_size, _tasks_at_once(config, TASKS_AT_ONCE) * len(MODELS))

    async def run(self, task, ctx):
        path = self._path(task)
        classifier = self._classifier(task)
        sheet, inputs = await self._passes.blocking(self._read, path, classifier, ctx)
        # Each model's session first, downloading the model if need be: blocking, so on a thread. A Cancel ends this
        # await at once; the download, which looks at `ctx.cancelled`, stops once no Task waits on it.
        await asyncio.gather(*(self._passes.blocking(self.models.session, model, ctx) for model in classifier.models))
        # From the Task's own coroutine, not the batches it shares: the first forward pass may be slow (cuDNN searches
        # for its algorithms then).
        ctx.progress(0.0, "Classifying")
        rows = await asyncio.gather(
            *(self._passes.row(model, inputs[model.preprocessing]) for model in classifier.models)
        )
        scores = classifier.score({model.id: row[np.newaxis] for model, row in zip(classifier.models, rows)})[0]

        result = {"path": path}
        if sheet is not None and isinstance(sheet.get("video"), str):
            result["video"] = sheet["video"]
        result.update(answer(classifier, scores, task.settings.get("minConfidence")))
        if sheet is not None:
            result["warnings"] = [SHEET_AS_ONE_IMAGE]
        return result

    @staticmethod
    def _read(path: str, classifier: Classifier, ctx) -> tuple[dict | None, dict[Preprocessing, np.ndarray]]:
        """The image's frame-sheet metadata (`None` unless it is a sheet), and its input for each way the classifier's
        models take an image. The image is read where its path really leads, once the Task's reach allows it: the
        check looks at the disk, so it is here, off the event loop."""
        image = open_image(ctx.allowed_path(path, "read"))
        sheet = sheet_of(image)
        rgb = to_rgb(image)
        inputs = {model.preprocessing: to_input(rgb, model.preprocessing) for model in classifier.models}
        return sheet, inputs

    def teardown(self):
        self._passes.close()
        super().teardown()


class ClassifyFrames(_Classify):
    """A whole video from its frame sheet, or a folder of frames, or one image as one frame: each frame through the
    classifier, in batches, then one answer from the frames' mean scores (`tag`, `confidence`, `scores`), how many
    frames agree with it (`agreement`), and each frame's own (`frames`). Its path must be in the Task's reach (Allowed
    roots), else the Task fails saying why; a folder's image outside it is skipped with a warning."""

    def run(self, task, ctx):
        path = self._path(task)
        classifier = self._classifier(task)
        min_confidence = task.settings.get("minConfidence")
        source = read_frames(path, ctx.allowed_path)
        # Each model's session first, so that loading it is said as such, then what the first batch does.
        for model in classifier.models:
            self.models.session(model, ctx)
        scores = _frame_by_frame(
            source,
            self.batch_size,
            lambda images: self._score(classifier, images, ctx),
            ("Classifying", "Classified"),
            ctx,
        )

        overall = answer(classifier, scores.mean(axis=0), min_confidence)
        # The label `confidence` is for: the tag, or the top label when the tag is uncertain.
        top = classifier.labels.index(max(overall["scores"], key=overall["scores"].get))
        frames = []
        for frame, row in zip(source.frames, scores, strict=True):
            own = answer(classifier, row, min_confidence)
            frames.append({"timestamp": frame.timestamp, "tag": own["tag"], "scores": own["scores"]})

        agreement = float(np.mean(scores.argmax(axis=1) == top))
        return _frames_result(path, source, {**overall, "agreement": agreement, "frames": frames})


class _Tag(_Worker):
    """What both tagging Workers share: how a Task names its tagger and sets its thresholds and `maxTags`,
    `tagBatchSize`, getting the tagger ready, and the warning of a tagger run on the CPU."""

    verb = "tag"

    def __init__(self, config, pool=None):
        super().__init__(config, pool)
        # A Profile start-up field: the most images in one forward pass of a tagger.
        self.batch_size = config.get("tagBatchSize") or DEFAULT_TAG_BATCH_SIZE

    @staticmethod
    def _tagger(task) -> Tagger:
        name = task.settings.get("tagger") or DEFAULT_TAGGER
        tagger = TAGGERS.get(name)
        if tagger is None:
            raise ValueError(f"No tagger `{name}`: one of {', '.join(TAGGERS)}")
        return tagger

    @staticmethod
    def _settings(task, tagger: Tagger) -> tuple[dict[str, float], int | None]:
        """The Task's thresholds, merged over the tagger's defaults, and its `maxTags`. Called before anything is read
        or downloaded, so a threshold the tagger has no category for fails the Task at once."""
        return thresholds_of(tagger, task.settings.get("thresholds")), task.settings.get("maxTags")

    def _ready(self, tagger: Tagger, ctx) -> Labels:
        """The tagger's tag list, then its session, each downloaded first if need be: so the model's download is the
        last one the Task says before it says the model loads."""
        labels = self.models.labels(tagger, ctx)
        self.models.session(tagger, ctx)
        return labels

    def _warnings(self, tagger: Tagger) -> list[str]:
        """The result's warnings of running the tagger: that it ran on the CPU, when it did."""
        return [ran_on_the_cpu(tagger)] if self.models.runs_on_cpu(tagger) else []


class TagImage(_Tag):
    """One image per Task, through one tagger of the catalogue (the `tagger` setting): the tags over their category's
    threshold (the `thresholds` setting, merged over the tagger's defaults), highest first, at most `maxTags`; and the
    tagger's rating. Its path must be in the Task's reach (Allowed roots), else the Task fails saying why.

    The Tasks of a process run at once on the shim's event loop, as `ClassifyImage`'s do, and share the tagger's
    forward passes, `tagBatchSize` images at most to one: thresholds and `maxTags` act on the output, so Tasks with
    different settings share them too."""

    def __init__(self, config, pool=None):
        super().__init__(config, pool)
        self._passes = _Passes(self.models, self.batch_size, _tasks_at_once(config, TAG_TASKS_AT_ONCE))

    async def run(self, task, ctx):
        path = self._path(task)
        tagger = self._tagger(task)
        thresholds, max_tags = self._settings(task, tagger)
        input = await self._passes.blocking(self._read, path, tagger, ctx)
        # The tag list and the session, downloading either if need be: blocking, so on a thread. A Cancel ends this
        # await at once, as `ClassifyImage`'s.
        labels = await self._passes.blocking(self._ready, tagger, ctx)
        # From the Task's own coroutine: the first forward pass may be slow (cuDNN searches for its algorithms then).
        ctx.progress(0.0, "Tagging")
        row = await self._passes.row(tagger, input)

        result = {"path": path, **tag(tagger, labels, row, thresholds, max_tags)}
        if warnings := self._warnings(tagger):
            result["warnings"] = warnings
        return result

    @staticmethod
    def _read(path: str, tagger: Tagger, ctx) -> np.ndarray:
        """The image as the tagger takes it, read where its path really leads once the Task's reach allows it."""
        return to_input(to_rgb(open_image(ctx.allowed_path(path, "read"))), tagger.preprocessing)

    def teardown(self):
        self._passes.close()
        super().teardown()


class TagFrames(_Tag):
    """A whole video from its frame sheet, or a folder of frames, or one image as one frame, read as `ClassifyFrames`
    reads them: each frame through the tagger, `tagBatchSize` at a time, and each frame's own tags and rating
    (`frames`), as `TagImage` gives them for an image, in the sheet's or folder's order. Its path must be in the Task's
    reach (Allowed roots), else the Task fails saying why; a folder's image outside it is skipped with a warning.

    A Worker and pool of its own, as `ClassifyFrames` has: fyflow gives each Node Type its own pool, so it could not
    share `TagImage`'s process. Both pools hold the `gpu` unit while they live, so they take turns on a GPU and never
    hold two copies of a tagger on it at once."""

    what = "the frames"

    def run(self, task, ctx):
        path = self._path(task)
        tagger = self._tagger(task)
        thresholds, max_tags = self._settings(task, tagger)
        source = read_frames(path, ctx.allowed_path)
        # The tag list and the session first, so that loading the model is said as such, then what the first batch
        # does.
        labels = self._ready(tagger, ctx)

        def score(images: list[Image.Image]) -> np.ndarray:
            inputs = np.stack([to_input(image, tagger.preprocessing) for image in images])
            return self.models.scores(tagger, inputs, ctx)

        rows = _frame_by_frame(source, self.batch_size, score, ("Tagging", "Tagged"), ctx)
        frames = []
        for frame, row in zip(source.frames, rows, strict=True):
            own = tag(tagger, labels, row, thresholds, max_tags)
            frames.append({"timestamp": frame.timestamp, "tags": own["tags"], "rating": own["rating"]})
        return _frames_result(path, source, {"tagger": tagger.id, "frames": frames}, self._warnings(tagger))
