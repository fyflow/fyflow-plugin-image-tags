"""The classifiers this Plugin knows: the models, each pinned to a Hugging Face commit and its file's sha256, and the
presets, each a fixed set of labels computed from one or more models by a rule.

Two names, kept apart on purpose: `MODELS` are the files (what is downloaded, kept and run), `CLASSIFIERS` what a
Node's `classifier` setting may name (every model, and every preset). A preset's labels stay the same when the models
under it change, so a Flow that branches on them survives a model swap; a model's labels are the model's own.

Nothing outside this catalogue ever loads. A model entry's labels, input size and output are read from the model's own
files when the entry is written; `scripts/golden.py` checks them against `meta.json` and the ONNX graph, and the golden
tests hold the scores to imgutils.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

#: The tag of an answer whose top score is below the classifier's `minConfidence`: a result, never a failure.
UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class Model:
    """One ONNX classifier: `labels[i]` is the probability at `output[:, i]`, for any batch size."""

    kind: ClassVar[str] = "model"

    id: str
    title: str
    repo: str
    commit: str
    name: str
    sha256: str
    labels: tuple[str, ...]
    #: The model's input, width and height: an image is squashed to it, its aspect ignored. A model whose graph takes
    #: any height and width gets imgutils' 384×384.
    size: tuple[int, int]
    file: str = "model.onnx"
    #: A raw model never answers `uncertain` unless a Node asks for a `minConfidence`.
    min_confidence: float = 0.0

    @property
    def models(self) -> tuple["Model", ...]:
        """The models it runs: itself."""
        return (self,)

    def score(self, scores: Mapping[str, np.ndarray]) -> np.ndarray:
        """Its labels' scores per frame, `[n, len(labels)]`, from each model's (keyed by model id): its own."""
        return scores[self.id]

    @property
    def remote_path(self) -> str:
        """Where the file is on Hugging Face (or a mirror), below the endpoint."""
        return f"{self.repo}/resolve/{self.commit}/{self.name}/{self.file}"

    @property
    def local_path(self) -> str:
        """Where the file is kept, below the models' folder. The commit is cut to 12 characters, which still names
        it, to stay under Windows' 260-character paths when long paths are off."""
        return f"{self.repo}/{self.commit[:12]}/{self.name}/{self.file}"


@dataclass(frozen=True)
class Preset:
    """A fixed set of labels computed per frame from the scores of the models it needs, by its rule."""

    kind: ClassVar[str] = "preset"

    id: str
    title: str
    labels: tuple[str, ...]
    models: tuple[Model, ...]
    #: From each model's scores per frame, `[n, len(model.labels)]` keyed by model id, to the preset's,
    #: `[n, len(labels)]`.
    rule: Callable[[Mapping[str, np.ndarray]], np.ndarray]
    min_confidence: float

    def score(self, scores: Mapping[str, np.ndarray]) -> np.ndarray:
        """Its labels' scores per frame, `[n, len(labels)]`, from each model's (keyed by model id): its rule's."""
        return self.rule(scores)


type Classifier = Model | Preset


ANIME_REAL = Model(
    id="anime_real",
    title="Anime or real (deepghs anime_real_cls)",
    repo="deepghs/anime_real_cls",
    commit="097a6c1e9e62866a435810c883a45a7cb2c7077d",
    name="mobilenetv3_v1.4_dist",
    sha256="c02d68cc7f56dbcbcd4f41a10df190385c1f963f22c4eefb8d162869800ed3b0",
    labels=("anime", "real"),
    size=(384, 384),
)

# v1.5 (imgutils' default) rather than v1.3: v1.3's published accuracy is higher, but v1.5 found more anime on the
# eval set.
ANIME_CLASSIFICATION = Model(
    id="anime_classification",
    title="3D, bangumi, comic, illustration or not a painting (deepghs anime_classification)",
    repo="deepghs/anime_classification",
    commit="5ee62e06f5f4cd68a1c2f3bc5dc9805e827d37df",
    name="mobilenetv3_v1.5_dist",
    sha256="ea8f4497b0e3ee86f43b6a6fd38884e25f960979a1debb5c4e1ee6a22e80d111",
    labels=("3d", "bangumi", "comic", "illustration", "not_painting"),
    # Its graph takes any height and width: imgutils' default.
    size=(384, 384),
)

#: Below this, `anime_classification`'s four painted kinds together are taken as nothing (see `style`).
PAINTED_FLOOR = 1e-6


def style(scores: Mapping[str, np.ndarray]) -> np.ndarray:
    """anime / cg / live_action per frame.

    `live_action` is `anime_real`'s `real` to the fourth power, so only a very sure `real` counts as live action. What
    is not live action is shared out by `anime_classification`, its `not_painting` dropped and the other four made to
    sum to 1: `cg` is `3d`'s share, `anime` the rest (that of `bangumi`, `illustration` and `comic` together). So the
    three sum to 1.

    When the four sum to (all but) nothing, `anime_classification` is sure the frame is not a painting at all and says
    nothing about which kind of one: what is not live action is then split evenly between `anime` and `cg`, so such a
    frame is `uncertain` under the default threshold unless `anime_real` is very sure it is real.
    """
    live = scores[ANIME_REAL.id][:, ANIME_REAL.labels.index("real")] ** 4
    kinds = scores[ANIME_CLASSIFICATION.id]

    def kind(label):
        return kinds[:, ANIME_CLASSIFICATION.labels.index(label)]

    painted = kind("3d") + kind("bangumi") + kind("comic") + kind("illustration")
    told = painted >= PAINTED_FLOOR
    share = np.where(told, painted, 1.0)
    cg = np.where(told, kind("3d") / share, 0.5)
    anime = 1 - cg
    return np.stack([(1 - live) * anime, (1 - live) * cg, live], axis=1)


STYLE = Preset(
    id="style",
    title="Anime, CG or live action (anime_real and anime_classification)",
    labels=("anime", "cg", "live_action"),
    models=(ANIME_REAL, ANIME_CLASSIFICATION),
    rule=style,
    # Settled by the eval set: every anime video found, nothing else tagged anime.
    min_confidence=0.55,
)

#: The model files: what is downloaded, kept and run, and what the Plugin's status lists.
MODELS: dict[str, Model] = {model.id: model for model in [ANIME_REAL, ANIME_CLASSIFICATION]}

#: The presets, each over models of `MODELS`.
PRESETS: dict[str, Preset] = {preset.id: preset for preset in [STYLE]}

#: What a Node's `classifier` setting may name: every preset, and every model as it is.
CLASSIFIERS: dict[str, Classifier] = {**PRESETS, **MODELS}

DEFAULT_CLASSIFIER = STYLE.id


def answer(classifier: Classifier, scores: np.ndarray, min_confidence: float | None = None) -> dict:
    """What a classifier says of one image, from its labels' scores (`[len(labels)]`): its top label as the `tag` when
    that label's score reaches `min_confidence` (the classifier's own when `None`), else `uncertain`. `confidence` is
    the top label's score either way."""
    threshold = classifier.min_confidence if min_confidence is None else min_confidence
    by_label = {label: float(score) for label, score in zip(classifier.labels, scores, strict=True)}
    top = max(by_label, key=by_label.get)
    return {
        "classifier": classifier.id,
        "labels": list(classifier.labels),
        "tag": top if by_label[top] >= threshold else UNCERTAIN,
        "confidence": by_label[top],
        "scores": by_label,
        "models": [model.id for model in classifier.models],
    }
