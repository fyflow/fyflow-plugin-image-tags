"""The Plugin's Python Workers, as the Host builds them: one instance per Python process, `run` once per Task."""

import numpy as np

from .catalogue import CLASSIFIERS, DEFAULT_CLASSIFIER, answer
from .models import Models
from .preprocess import load_image, to_input


class ClassifyImage:
    """One image per Task, through one classifier of the catalogue (the `classifier` setting): a model, or a preset
    over several. Its top label is the `tag` when it reaches `minConfidence` (the classifier's own when the setting is
    left empty), else `uncertain`."""

    def __init__(self, config):
        # `modelDir`, a Profile start-up field, is the folder of an offline machine's models: then nothing downloads.
        self.models = Models(config["host"]["dataDir"], model_dir=config.get("modelDir"))

    def run(self, task, ctx):
        path = task.input.get("path") if isinstance(task.input, dict) else None
        if not isinstance(path, str) or path == "":
            raise ValueError("The input has no `path`: the image to classify")
        name = task.settings.get("classifier") or DEFAULT_CLASSIFIER
        classifier = CLASSIFIERS.get(name)
        if classifier is None:
            raise ValueError(f"No classifier `{name}`: one of {', '.join(CLASSIFIERS)}")

        image = load_image(path)
        inputs = {}  # by input size: models of the same size take the same tensor
        scores = {}
        for model in classifier.models:
            if model.size not in inputs:
                inputs[model.size] = to_input(image, model.size)[np.newaxis]
            scores[model.id] = self.models.scores(model, inputs[model.size], ctx)
        return {"path": path, **answer(classifier, classifier.score(scores)[0], task.settings.get("minConfidence"))}
