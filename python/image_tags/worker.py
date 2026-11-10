"""The Plugin's Python Workers, as the Host builds them: one instance per Python process, `run` once per Task."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .batcher import Batcher
from .catalogue import CLASSIFIERS, DEFAULT_CLASSIFIER, MODELS, Classifier, Model, answer
from .frames import SHEET_AS_ONE_IMAGE, read_frames, sheet_of
from .models import Cancelled, Models
from .preprocess import open_image, to_input, to_rgb

#: The most frames in one forward pass when the Profile sets no `batchSize`: a `ClassifyImage` batch, a
#: `ClassifyFrames` Task's chunk.
DEFAULT_BATCH_SIZE = 16
#: The `cpu` units each process holds while it lives: its pools' `residentGroups` in `plugin.ts`. A forward pass runs
#: on as many threads. Until the Host says what a process holds (`pool.holds`), the Worker can only assume this.
CPU_UNITS = 4
#: The most `ClassifyImage` Tasks a process runs at once: its pool's `maxConcurrentTasks` in `plugin.ts`.
TASKS_AT_ONCE = 32


def _cpu_units(pool) -> int:
    """The `cpu` units the process holds: what the Host says (`pool.holds`, which no Host gives yet), else
    `CPU_UNITS`."""
    holds = getattr(pool, "holds", None)
    return (holds or {}).get("cpu") or CPU_UNITS


class _Classify:
    """What both Workers share: the models, and how a Task names its image and classifier."""

    def __init__(self, config, pool=None):
        # Profile start-up fields: `modelDir`, the folder of an offline machine's models (then nothing downloads), and
        # `batchSize`, the most frames in one forward pass.
        self.batch_size = config.get("batchSize") or DEFAULT_BATCH_SIZE
        self.models = Models(config["host"]["dataDir"], model_dir=config.get("modelDir"), threads=_cpu_units(pool))

    @staticmethod
    def _path(task) -> str:
        path = task.input.get("path") if isinstance(task.input, dict) else None
        if not isinstance(path, str) or path == "":
            raise ValueError("The input has no `path`: the image to classify")
        return path

    @staticmethod
    def _classifier(task) -> Classifier:
        name = task.settings.get("classifier") or DEFAULT_CLASSIFIER
        classifier = CLASSIFIERS.get(name)
        if classifier is None:
            raise ValueError(f"No classifier `{name}`: one of {', '.join(CLASSIFIERS)}")
        return classifier

    def _score(self, classifier: Classifier, images: list[Image.Image], ctx) -> np.ndarray:
        """The classifier's scores for each image, `[len(images), len(labels)]`, from one batch per model."""
        inputs = {}  # by input size: models of the same size take the same tensor
        scores = {}
        for model in classifier.models:
            if model.size not in inputs:
                inputs[model.size] = np.stack([to_input(image, model.size) for image in images])
            scores[model.id] = self.models.scores(model, inputs[model.size], ctx)
        return classifier.score(scores)


class ClassifyImage(_Classify):
    """One image per Task, through one classifier of the catalogue (the `classifier` setting): a model, or a preset
    over several. Its top label is the `tag` when it reaches `minConfidence` (the classifier's own when the setting is
    left empty), else `uncertain`. A frame sheet is scored as one image, with a warning.

    The Tasks of a process run at once on the shim's event loop: each reads its image on a thread, then hands it to
    each model's batcher (`batcher.py`), so Tasks running together share forward passes, whatever their classifier."""

    def __init__(self, config, pool=None):
        super().__init__(config, pool)
        # One forward pass at a time in the process, off the event loop; ONNX Runtime spreads it over the
        # threads of the `cpu` units the process holds.
        self._forward = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-tags-forward")
        # What blocks a Task, off the event loop: reading its image, and getting each model, which may mean waiting on
        # a download. Enough threads for every Task at once waiting on every model (the event loop's own executor,
        # `asyncio.to_thread`'s, may have fewer than 32), started only as they are needed.
        self._blocking = ThreadPoolExecutor(TASKS_AT_ONCE * len(MODELS), thread_name_prefix="image-tags-task")
        self._batchers: dict[str, Batcher] = {}

    async def run(self, task, ctx):
        path = self._path(task)
        classifier = self._classifier(task)
        loop = asyncio.get_running_loop()
        sheet, inputs = await loop.run_in_executor(self._blocking, self._read, path, classifier)
        rows = await asyncio.gather(*(self._row(model, inputs[model.size], ctx) for model in classifier.models))
        scores = classifier.score({model.id: row[np.newaxis] for model, row in zip(classifier.models, rows)})[0]

        result = {"path": path}
        if sheet is not None and isinstance(sheet.get("video"), str):
            result["video"] = sheet["video"]
        result.update(answer(classifier, scores, task.settings.get("minConfidence")))
        if sheet is not None:
            result["warnings"] = [SHEET_AS_ONE_IMAGE]
        return result

    @staticmethod
    def _read(path: str, classifier: Classifier):
        """The image's frame-sheet metadata (`None` unless it is a sheet), and its input for each size the
        classifier's models take."""
        image = open_image(path)
        sheet = sheet_of(image)
        rgb = to_rgb(image)
        inputs = {model.size: to_input(rgb, model.size) for model in classifier.models}
        return sheet, inputs

    async def _row(self, model: Model, input: np.ndarray, ctx) -> np.ndarray:
        """`model`'s scores for one input, from a batch shared with the other Tasks that need the model."""
        # Its session first, downloading the model if need be: blocking, so on a thread. A Cancel ends this await at
        # once; the download, which looks at `ctx.cancelled`, stops once no Task waits on it.
        await asyncio.get_running_loop().run_in_executor(self._blocking, self.models.session, model, ctx)
        return await self._batcher(model).score(input)

    def _batcher(self, model: Model) -> Batcher:
        batcher = self._batchers.get(model.id)
        if batcher is None:
            batcher = self._batchers[model.id] = Batcher(model, self.models, self._forward, self.batch_size)
        return batcher

    def teardown(self):
        self._forward.shutdown(wait=False)
        self._blocking.shutdown(wait=False)


class ClassifyFrames(_Classify):
    """A whole video from its frame sheet, or a folder of frames, or one image as one frame: each frame through the
    classifier, in batches, then one answer from the frames' mean scores (`tag`, `confidence`, `scores`), how many
    frames agree with it (`agreement`), and each frame's own (`frames`)."""

    def run(self, task, ctx):
        path = self._path(task)
        classifier = self._classifier(task)
        min_confidence = task.settings.get("minConfidence")
        source = read_frames(path)
        total = len(source.frames)

        per_frame = []
        for start in range(0, total, self.batch_size):
            if getattr(ctx, "cancelled", False):
                raise Cancelled(f"Cancelled after {start} of {total} frames")
            batch = source.frames[start : start + self.batch_size]
            per_frame.append(self._score(classifier, [frame.load() for frame in batch], ctx))
            done = start + len(batch)
            ctx.progress(done / total, f"Classified {done} of {total} frames")
        scores = np.concatenate(per_frame)

        overall = answer(classifier, scores.mean(axis=0), min_confidence)
        # The label `confidence` is for: the tag, or the top label when the tag is uncertain.
        top = classifier.labels.index(max(overall["scores"], key=overall["scores"].get))
        frames = []
        for frame, row in zip(source.frames, scores, strict=True):
            own = answer(classifier, row, min_confidence)
            frames.append({"timestamp": frame.timestamp, "tag": own["tag"], "scores": own["scores"]})

        result = {"path": path}
        if source.video is not None:
            result["video"] = source.video
        result.update(overall)
        result["agreement"] = float(np.mean(scores.argmax(axis=1) == top))
        result["frames"] = frames
        if source.warnings:
            result["warnings"] = source.warnings
        return result
