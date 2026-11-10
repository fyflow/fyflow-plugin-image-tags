"""The Plugin's Python Workers, as the Host builds them: one instance per Python process, `run` once per Task."""

import numpy as np
from PIL import Image

from .catalogue import CLASSIFIERS, DEFAULT_CLASSIFIER, Classifier, answer
from .frames import SHEET_AS_ONE_IMAGE, read_frames, sheet_of
from .models import Cancelled, Models
from .preprocess import open_image, to_input, to_rgb

#: How many frames of a `ClassifyFrames` Task go through the models at once.
BATCH_FRAMES = 32


class _Classify:
    """What both Workers share: the models, and how a Task names its image and classifier."""

    def __init__(self, config):
        # `modelDir`, a Profile start-up field, is the folder of an offline machine's models: then nothing downloads.
        self.models = Models(config["host"]["dataDir"], model_dir=config.get("modelDir"))

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
    left empty), else `uncertain`. A frame sheet is scored as one image, with a warning."""

    def run(self, task, ctx):
        path = self._path(task)
        classifier = self._classifier(task)
        image = open_image(path)
        sheet = sheet_of(image)
        (scores,) = self._score(classifier, [to_rgb(image)], ctx)
        result = {"path": path}
        if sheet is not None and isinstance(sheet.get("video"), str):
            result["video"] = sheet["video"]
        result.update(answer(classifier, scores, task.settings.get("minConfidence")))
        if sheet is not None:
            result["warnings"] = [SHEET_AS_ONE_IMAGE]
        return result


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
        for start in range(0, total, BATCH_FRAMES):
            if getattr(ctx, "cancelled", False):
                raise Cancelled(f"Cancelled after {start} of {total} frames")
            batch = source.frames[start : start + BATCH_FRAMES]
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
