"""The Plugin's Python Workers, as the Host builds them: one instance per Python process, `run` once per Task."""

import numpy as np

from .catalogue import DEFAULT_CLASSIFIER, MODELS
from .models import Models
from .preprocess import load_image, to_input


class ClassifyImage:
    """One image per Task, through one classifier of the catalogue (the `classifier` setting)."""

    def __init__(self, config):
        # `modelDir`, a Profile start-up field, is the folder of an offline machine's models: then nothing downloads.
        self.models = Models(config["host"]["dataDir"], model_dir=config.get("modelDir"))

    def run(self, task, ctx):
        path = task.input.get("path") if isinstance(task.input, dict) else None
        if not isinstance(path, str) or path == "":
            raise ValueError("The input has no `path`: the image to classify")
        classifier = task.settings.get("classifier", DEFAULT_CLASSIFIER)
        model = MODELS.get(classifier)
        if model is None:
            raise ValueError(f"No classifier `{classifier}`: one of {', '.join(MODELS)}")

        image = load_image(path)
        scores = self.models.scores(model, to_input(image, model.size)[np.newaxis], ctx)[0]
        by_label = {label: float(score) for label, score in zip(model.labels, scores, strict=True)}
        tag = max(by_label, key=by_label.get)
        return {
            "path": path,
            "classifier": model.id,
            "labels": list(model.labels),
            "tag": tag,
            "confidence": by_label[tag],
            "scores": by_label,
            "models": [model.id],
        }
