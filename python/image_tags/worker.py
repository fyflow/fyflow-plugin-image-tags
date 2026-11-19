"""The Plugin's Python Workers, as the Host builds them: one instance per Python process, `run` once per Task."""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from PIL import Image

from . import frames as framing
from . import scenes
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
from .frames import (
    SHEET_AS_ONE_IMAGE,
    SHEET_TAGGED_AS_ONE_IMAGE,
    Frame,
    Frames,
    is_blank,
    read_frame_list,
    read_frames,
    sheet_of,
)
from .models import Cancelled, Models
from .preprocess import Preprocessing, open_image, to_input, to_rgb
from .results import FileOptions, file_options, fits, key_of, name_of, reused, summary, target, write

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
    warning = (
        f"{tagger.id} ran on the CPU, where a tagger is slow: a GPU build (`cuda` or `directml`) tags many times "
        "faster"
    )
    if tagger.precision == "fp16":
        warning += "; and fp16 on the CPU is slow too, no quicker than fp32, which is closer to the authors' scores"
    return warning


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
    """`score`'s rows for every frame of `source`, in order, `batch_size` frames at a time (`_batch_by_batch`)."""
    return np.concatenate(_batch_by_batch(source, batch_size, score, verbs, ctx))


def _batch_by_batch(
    source: Frames, batch_size: int, run: Callable[[list[Image.Image]], Any], verbs: tuple[str, str], ctx, share=1.0
) -> list:
    """What `run` gives for each batch of `source`'s frames, in order, `batch_size` frames at a time, each batch's
    frames read only when it comes. The Task says how far it has got before the first batch and after each (`verbs`:
    what it is doing, then what it has done, as "Tagging" and "Tagged"), the batches taking its progress from 0 to
    `share`, and a Cancel stops it before its next batch."""
    doing, did = verbs
    total = len(source.frames)
    ctx.progress(0.0, f"{doing} 0 of {total} frames")
    out = []
    for start in range(0, total, batch_size):
        if getattr(ctx, "cancelled", False):
            raise Cancelled(f"Cancelled after {start} of {total} frames")
        batch = source.frames[start : start + batch_size]
        out.append(run([frame.load() for frame in batch]))
        done = start + len(batch)
        ctx.progress(share * done / total, f"{did} {done} of {total} frames")
    return out


def _image_result(path: str, sheet: dict | None) -> dict:
    """The start of an image Task's result: the path it was given, then the video of a frame sheet that names one."""
    result = {"path": path}
    if sheet is not None and isinstance(sheet.get("video"), str):
        result["video"] = sheet["video"]
    return result


def _read_image(path: str, ctx) -> tuple[Image.Image, dict | None]:
    """The image at `path` in RGB, and its frame-sheet metadata (`None` unless it is a sheet). It is read where its
    path really leads, once the Task's reach allows it: the check looks at the disk, so a Task calls this off the event
    loop."""
    image = open_image(ctx.allowed_path(path, "read"))
    return to_rgb(image), sheet_of(image)


def _frames_result(path: str | None, source: Frames, fields: dict, warnings: list[str] | None = None) -> dict:
    """A frames Task's result: the path it was given (none for a `frames` list), the video the frames name when they
    name one, `fields`, then the warnings of reading the frames and `warnings`, when there are any."""
    result = {} if path is None else {"path": path}
    if source.video is not None:
        result["video"] = source.video
    result.update(fields)
    if every := [*source.warnings, *(warnings or [])]:
        result["warnings"] = every
    return result


def _named(frame: Frame) -> dict:
    """The start of a frame's own answer: its own image's path, for a folder's or a list's frame, then its
    timestamp."""
    return {"timestamp": frame.timestamp} if frame.path is None else {"path": frame.path, "timestamp": frame.timestamp}


class _Worker:
    """What every Worker shares: the models, and how a Task names what it reads (`<what> to <verb>`)."""

    verb = "classify"
    what = "the image"

    def __init__(self, config, pool=None):
        # A Profile start-up field: `modelDir`, the folder of an offline machine's models (then nothing downloads).
        # On the GPU the process holds while it lives, when the variant's build has a GPU provider. The variant itself
        # (`host.variant`, since SDK 0.11) is only said on stderr: the providers are what the installed build offers.
        self.data_dir = config["host"].get("dataDir")
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

    def _frames(self, task, ctx) -> tuple[str | None, Frames]:
        """A frames Task's input, read: exactly one of `path` (a sheet, folder or image) and `frames` (a list of
        `{path, timestamp?}`), with the path it was given, None for a list."""
        input = task.input if isinstance(task.input, dict) else {}
        has_path, has_list = input.get("path") is not None, input.get("frames") is not None
        if has_path and has_list:
            raise ValueError(f"The input has both `path` and `frames`: give one, {self.what} to {self.verb}")
        if has_list:
            return None, read_frame_list(input["frames"], ctx.allowed_path)
        if not has_path:
            raise ValueError(f"The input has no `path` or `frames`: {self.what} to {self.verb}")
        path = self._path(task)
        return path, read_frames(path, ctx.allowed_path)

    def _hand_on(
        self,
        task,
        ctx,
        options: FileOptions | None,
        path: str | None,
        source: Frames,
        settings: dict,
        models: list,
        make: Callable[[], dict],
        lists: tuple[str, ...] = ("frames",),
    ) -> dict:
        """A frames Task's result: `make()`'s, or with `lists: "file"`, its summary once the whole is in a file, without
        its list fields (`lists`), and `count` the length of the first. A file written before for the same frames,
        settings and models is handed on instead, no model run, unless `overwrite` is set (`results.py`)."""
        if options is None:
            return fits(make())
        input = {key: task.input.get(key) for key in ("path", "frames")}
        key = key_of(type(self).__name__, input, source.files, settings, models)
        output_dir = (task.settings.get("profile") or {}).get("outputDir")
        name = name_of(path, source.video, source.files)
        file, named = target(options, output_dir, self.data_dir, name, key, ctx.allowed_path)
        if named and not options.overwrite and (found := reused(file, lists)) is not None:
            count = len(found[lists[0]])
            ctx.progress(1.0, f"Reused the answers of {count} {lists[0]}")
            return summary(found, file, True, count, lists)
        result = make()
        write(file, result)
        return summary(result, file, False, len(result[lists[0]]), lists)

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

        result = _image_result(path, sheet)
        result.update(answer(classifier, scores, task.settings.get("minConfidence")))
        if sheet is not None:
            result["warnings"] = [SHEET_AS_ONE_IMAGE]
        return result

    @staticmethod
    def _read(path: str, classifier: Classifier, ctx) -> tuple[dict | None, dict[Preprocessing, np.ndarray]]:
        """The image's frame-sheet metadata (`None` unless it is a sheet), and its input for each way the classifier's
        models take an image."""
        rgb, sheet = _read_image(path, ctx)
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

    what = "the frames"

    def run(self, task, ctx):
        classifier = self._classifier(task)
        min_confidence = task.settings.get("minConfidence")
        options = file_options(task.settings)
        path, source = self._frames(task, ctx)
        settings = {"classifier": classifier.id, "minConfidence": min_confidence}
        return self._hand_on(
            task,
            ctx,
            options,
            path,
            source,
            settings,
            list(classifier.models),
            lambda: self._classify(classifier, min_confidence, path, source, ctx),
        )

    def _classify(self, classifier: Classifier, min_confidence, path: str | None, source: Frames, ctx) -> dict:
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
            frames.append({**_named(frame), "tag": own["tag"], "scores": own["scores"]})

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
        or downloaded, so a threshold the tagger has no category for, or a `maxTags` below 1, fails the Task at
        once."""
        max_tags = task.settings.get("maxTags")
        if max_tags is not None and (isinstance(max_tags, bool) or not isinstance(max_tags, int) or max_tags < 1):
            raise ValueError(f"`maxTags` is not a whole number of 1 or more: {max_tags!r}")
        return thresholds_of(tagger, task.settings.get("thresholds")), max_tags

    def _ready(self, tagger: Tagger, ctx) -> Labels:
        """The tagger's tag list, then its session, each downloaded first if need be: so the model's download is the
        last one the Task says before it says the model loads."""
        labels = self.models.labels(tagger, ctx)
        self.models.session(tagger, ctx)
        return labels

    def _warnings(self, tagger: Tagger, embedding: bool = False) -> list[str]:
        """The result's warnings of running the tagger (its embedding session, with `embedding`): that it ran on the
        CPU, when it did."""
        return [ran_on_the_cpu(tagger)] if self.models.runs_on_cpu(tagger, embedding=embedding) else []


class TagImage(_Tag):
    """One image per Task, through one tagger of the catalogue (the `tagger` setting): the tags over their category's
    threshold (the `thresholds` setting, merged over the tagger's defaults), highest first, at most `maxTags`; and the
    tagger's rating. A frame sheet is tagged as one image, with a warning. Its path must be in the Task's reach (Allowed
    roots), else the Task fails saying why.

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
        sheet, input = await self._passes.blocking(self._read, path, tagger, ctx)
        # The tag list and the session, downloading either if need be: blocking, so on a thread. A Cancel ends this
        # await at once, as `ClassifyImage`'s.
        labels = await self._passes.blocking(self._ready, tagger, ctx)
        # From the Task's own coroutine: the first forward pass may be slow (cuDNN searches for its algorithms then).
        ctx.progress(0.0, "Tagging")
        row = await self._passes.row(tagger, input)

        result = _image_result(path, sheet)
        result.update(tag(tagger, labels, row, thresholds, max_tags))
        if warnings := [*([SHEET_TAGGED_AS_ONE_IMAGE] if sheet is not None else []), *self._warnings(tagger)]:
            result["warnings"] = warnings
        return result

    @staticmethod
    def _read(path: str, tagger: Tagger, ctx) -> tuple[dict | None, np.ndarray]:
        """The image's frame-sheet metadata (`None` unless it is a sheet), and the image as the tagger takes it."""
        rgb, sheet = _read_image(path, ctx)
        return sheet, to_input(rgb, tagger.preprocessing)

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
        tagger = self._tagger(task)
        thresholds, max_tags = self._settings(task, tagger)
        options = file_options(task.settings)
        path, source = self._frames(task, ctx)
        settings = {"tagger": tagger.id, "thresholds": thresholds, "maxTags": max_tags}
        return self._hand_on(
            task,
            ctx,
            options,
            path,
            source,
            settings,
            [tagger],
            lambda: self._tag(tagger, thresholds, max_tags, path, source, ctx),
        )

    def _tag(self, tagger: Tagger, thresholds: dict, max_tags, path: str | None, source: Frames, ctx) -> dict:
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
            frames.append({**_named(frame), "tags": own["tags"], "rating": own["rating"]})
        return _frames_result(path, source, {"tagger": tagger.id, "frames": frames}, self._warnings(tagger))


#: The most entries in any one list of a `TagScenes` result (a scene's tags or highlights, the summary's tags, a
#: frame's tags) when a Node sets no `maxTags`.
SCENE_MAX_TAGS = 40
#: The share of a `TagScenes` Task's progress its tagging takes; grouping and pooling take the rest.
TAGGING_SHARE = 0.9
#: The constants of `scenes.py` a `TagScenes` result depends on, by name: a result file's key covers them, so a file
#: made under other ones (a review may vary them) is not reused.
SCENE_CONSTANTS = (
    "SCORE_FLOOR",
    "LINK_WINDOW",
    "MAX_LINK_LAG",
    "LINK_PERCENTILE",
    "DF_SCORE",
    "IDF_FLOOR",
    "CHARACTER_WEIGHT",
    "SMOOTHING",
    "SMOOTHING_REACH",
    "OFF_GAP",
    "PEAK_STRONG",
    "PEAK_REPEAT",
    "RATING_STRONG",
)


def _fraction(settings: dict, name: str, default: float) -> float:
    """A setting from 0 to 1, `default` when it is left empty."""
    value = settings.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
        raise ValueError(f"`{name}` is not a number from 0 to 1: {value!r}")
    return float(value)


def _switch(settings: dict, name: str, default: bool) -> bool:
    """A true-or-false setting, `default` when it is left empty."""
    value = settings.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"`{name}` is not true or false: {value!r}")
    return value


def _timed(source: Frames) -> None:
    """Fails unless every frame has a timestamp: grouping runs on time, and a guessed spacing would make scene times
    up."""
    missing = sum(frame.timestamp is None for frame in source.frames)
    if missing:
        has = "has" if missing == 1 else "have"
        raise ValueError(
            f"{missing} of {len(source.frames)} frames {has} no timestamp, and TagScenes groups frames by their time: "
            "give it frames that carry one (a frame sheet or thumbnails from VideoThumbnail, the frames of "
            "DetectScenes or ExtractFrames), or a `frames` list with each frame's `timestamp`"
        )


def _ratings(tagger: Tagger, labels: Labels) -> tuple[list[int], list[str]]:
    """The places of the tagger's ratings in its output, and their names as a result gives them, from the least to the
    most severe: the tag list's order (WD's `general`, `sensitive`, `questionable`, `explicit`), or `rating_names`'.
    None for a tagger that rates nothing."""
    if tagger.rating_code is None:
        return [], []
    at = [i for i, code in enumerate(labels.codes) if code == tagger.rating_code]
    if not tagger.rating_names:
        return at, [labels.names[i] for i in at]
    by_name = {labels.names[i]: i for i in at}
    if lacking := [theirs for theirs, _ in tagger.rating_names if theirs not in by_name]:
        raise ValueError(
            f"The tag list of {tagger.id} has no rating `{lacking[0]}`: its ratings are {', '.join(by_name)}"
        )
    return [by_name[theirs] for theirs, _ in tagger.rating_names], [ours for _, ours in tagger.rating_names]


class TagScenes(_Tag):
    """A whole video from its frames, read as `TagFrames` reads them, every one with a timestamp (else the Task fails
    before any model loads): each frame through the tagger's embedding session (`Models.scores_and_embeddings`),
    `tagBatchSize` at a time, then the frames grouped into scenes by their embeddings and tags, and each scene's tags,
    highlights, rating and keyframe pooled, and the video's (`scenes.py`). Its result has no frame's own tags unless
    `includeFrames` asks for them; with `lists: "file"`, its `scenes` and `frames` go to the file.

    Of each frame only what grouping needs is kept while the others are tagged: its unit-length embedding, its scores
    of `SCORE_FLOOR` or more, its ratings, its timestamp, and whether it is blank (`is_blank`, as its batch loads);
    with `includeFrames`, its own tags and rating as `TagFrames` gives them too.

    A Worker and pool of its own, as `TagFrames` has, holding the same `gpu` unit: so its embedding session never
    shares a process with a plain session of the same tagger."""

    what = "the frames"

    def run(self, task, ctx):
        tagger = self._tagger(task)
        thresholds, max_tags = self._settings(task, tagger)
        max_tags = max_tags or SCENE_MAX_TAGS
        grouping = self._grouping(task.settings, max_tags)
        include_frames = _switch(task.settings, "includeFrames", False)
        options = file_options(task.settings)
        path, source = self._frames(task, ctx)
        _timed(source)
        settings = {
            "tagger": tagger.id,
            "thresholds": thresholds,
            "maxTags": max_tags,
            "embeddingWeight": grouping.embedding_weight,
            "minSceneDuration": grouping.min_scene_duration,
            "ignoreTags": sorted(grouping.ignore_tags),
            "minCoverage": grouping.min_coverage,
            "includeTags": grouping.include_tags,
            "includeFrames": include_frames,
            "constants": {
                **{name: getattr(scenes, name) for name in SCENE_CONSTANTS},
                "SHOT_SPECIFIC_TAGS": sorted(scenes.SHOT_SPECIFIC_TAGS),
                "BLANK_STD": framing.BLANK_STD,
            },
        }
        return self._hand_on(
            task,
            ctx,
            options,
            path,
            source,
            settings,
            [tagger],
            lambda: self._scenes(tagger, thresholds, max_tags, grouping, include_frames, path, source, ctx),
            lists=("scenes", "frames"),
        )

    @staticmethod
    def _grouping(settings: dict, max_tags: int) -> scenes.Settings:
        """The Task's settings that grouping and pooling read, checked before anything is read or downloaded."""
        defaults = scenes.Settings()
        min_duration = settings.get("minSceneDuration")
        if min_duration is None:
            min_duration = defaults.min_scene_duration
        elif isinstance(min_duration, bool) or not isinstance(min_duration, int | float) or min_duration < 0:
            raise ValueError(f"`minSceneDuration` is not a number of seconds, 0 or more: {min_duration!r}")
        ignore = settings.get("ignoreTags") or []
        if not isinstance(ignore, list) or not all(isinstance(name, str) and name for name in ignore):
            raise ValueError(f"`ignoreTags` is not a list of tags: {ignore!r}")
        return scenes.Settings(
            embedding_weight=_fraction(settings, "embeddingWeight", defaults.embedding_weight),
            min_scene_duration=float(min_duration),
            ignore_tags=frozenset(ignore),
            min_coverage=_fraction(settings, "minCoverage", defaults.min_coverage),
            max_tags=max_tags,
            include_tags=_switch(settings, "includeTags", defaults.include_tags),
        )

    def _scenes(
        self,
        tagger: Tagger,
        thresholds: dict,
        max_tags: int,
        grouping: scenes.Settings,
        include_frames: bool,
        path: str | None,
        source: Frames,
        ctx,
    ) -> dict:
        # The tag list and the embedding session first, so that loading the model is said as such, then what the first
        # batch does.
        labels = self.models.labels(tagger, ctx)
        self.models.session(tagger, ctx, embedding=True)
        rating_rows, rating_labels = _ratings(tagger, labels)
        category_of = {category.code: category.name for category in tagger.categories}

        def run(images: list[Image.Image]):
            blank = [is_blank(image) for image in images]
            inputs = np.stack([to_input(image, tagger.preprocessing) for image in images])
            output, embeddings = self.models.scores_and_embeddings(tagger, inputs, ctx)
            ignore = grouping.ignore_tags
            own = [_own(tagger, labels, row, thresholds, max_tags, ignore) for row in output] if include_frames else []
            scores = output if tagger.sigmoided else 1 / (1 + np.exp(-output.astype(np.float64)))
            ratings = scores[:, rating_rows].astype(np.float32)
            scores[:, rating_rows] = 0  # kept whole in `ratings`, not among the tags
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            unit = (embeddings / np.maximum(norms, 1e-12)).astype(np.float32)
            # Of the batch's scores, only those of `SCORE_FLOOR` or more are kept.
            return scenes.Scores.from_dense(scores), unit, ratings, blank, own

        batches = _batch_by_batch(source, self.batch_size, run, ("Tagging", "Tagged"), ctx, TAGGING_SHARE)
        total = len(source.frames)
        tagged = scenes.TaggedFrames(
            timestamps=np.array([frame.timestamp for frame in source.frames], dtype=np.float64),
            embeddings=np.concatenate([batch[1] for batch in batches]),
            scores=scenes.Scores.stack([batch[0] for batch in batches]),
            ratings=np.concatenate([batch[2] for batch in batches]),
            blank=np.array([blank for batch in batches for blank in batch[3]], dtype=bool),
            paths=[frame.path for frame in source.frames],
            names=labels.names,
            categories=[category_of.get(code) for code in labels.codes],
            rating_labels=rating_labels,
            thresholds=thresholds,
            duration=source.duration,
        )
        own = [entry for batch in batches for entry in batch[4]]
        del batches

        _stop_if_cancelled(ctx, f"after tagging {total} frames")
        ctx.progress(TAGGING_SHARE, "Grouping")
        found = scenes.group(tagged, grouping)
        _stop_if_cancelled(ctx, f"after grouping {total} frames")
        ctx.progress(TAGGING_SHARE + (1 - TAGGING_SHARE) / 2, "Pooling")
        pooled = scenes.pool(tagged, found, grouping)

        # The video's duration where the thumbnails give it, else where the last frame's span ends.
        fields = {"tagger": tagger.id, "frameCount": total, "duration": pooled["scenes"][-1]["end"], **pooled}
        if include_frames:
            fields["frames"] = [
                {
                    **_named(frame),
                    "tags": mine["tags"],
                    "rating": mine["rating"],
                    "blank": bool(tagged.blank[i]),
                    "scene": int(found.scene_of[i]),
                }
                for i, (frame, mine) in enumerate(zip(source.frames, own, strict=True))
            ]
        result = _frames_result(path, source, fields, self._warnings(tagger, embedding=True))
        ctx.progress(1.0, f"Found {pooled['count']} scenes in {total} frames")
        return result


def _own(tagger: Tagger, labels: Labels, row: np.ndarray, thresholds: dict, max_tags: int, ignore: frozenset) -> dict:
    """A frame's own tags and rating, as `TagFrames` gives them, without the tags `ignore` names (`ignoreTags`)."""
    if not ignore:
        return tag(tagger, labels, row, thresholds, max_tags)
    own = tag(tagger, labels, row, thresholds)
    own["tags"] = [entry for entry in own["tags"] if entry["name"] not in ignore][:max_tags]
    return own


def _stop_if_cancelled(ctx, when: str) -> None:
    """A Cancel between a `TagScenes` Task's stages stops it there."""
    if getattr(ctx, "cancelled", False):
        raise Cancelled(f"Cancelled {when}")
