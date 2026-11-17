"""The models this Plugin knows: the classifiers and the taggers, each pinned to a Hugging Face commit and its files'
sha256, and the presets, each a fixed set of labels computed from one or more classifier models by a rule.

Three names, kept apart on purpose: `MODELS` are the classifiers' files (what is downloaded, kept and run),
`CLASSIFIERS` what a Node's `classifier` setting may name (every model, and every preset), and `TAGGERS` what a Node's
`tagger` setting may name, each a model with its tag list. A preset's labels stay the same when the models under it
change, so a Flow that branches on them survives a model swap; a model's labels are the model's own.

Each entry says how its model takes an image (`Preprocessing`). Nothing outside this catalogue ever loads. A
classifier entry's labels, input size and output are read from the model's own files when the entry is written;
`scripts/golden.py` checks them against `meta.json` and the ONNX graph, and the golden tests hold the scores to
imgutils. A tagger's are checked by `scripts/golden_tags.py` (PixAI's by its golden tests), and its golden tests hold
the scores to its reference pipeline.
"""

import csv
import io
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
from PIL import Image

from .preprocess import BGR, BLACK, NCHW, NHWC, RGB, WHITE, Preprocessing

#: The tag of an answer whose top score is below the classifier's `minConfidence`: a result, never a failure.
UNCERTAIN = "uncertain"

#: How imgutils feeds its classifiers, at 384×384.
IMGUTILS_384 = Preprocessing(
    size=(384, 384),
    pad=None,
    resample=Image.Resampling.BILINEAR,
    channels=RGB,
    layout=NCHW,
    scale=255.0,
    mean=0.5,
    std=0.5,
)


class Pinned:
    """A file at a pinned commit of a Hugging Face repo: `<repo>/resolve/<commit>/[<name>/]<file>`, kept at
    `<repo>/<commit, 12 characters>/[<name>/]<file>` below the models' folder. The commit is cut to 12 characters,
    which still names it, to stay under Windows' 260-character paths when long paths are off."""

    #: The catalogue id of the model the file is, or (a tag list) of the tagger it goes with.
    id: str
    repo: str
    commit: str
    #: The folder in the repo the file is in; empty for the repo's top.
    name: str
    file: str
    sha256: str

    @property
    def subject(self) -> str:
        """What a person reads the file as, in a message: "the model anime_real"."""
        return f"the model {self.id}"

    @property
    def remote_path(self) -> str:
        """Where the file is on Hugging Face (or a mirror), below the endpoint."""
        return "/".join([self.repo, "resolve", self.commit, *([self.name] if self.name else []), self.file])

    @property
    def local_path(self) -> str:
        """Where the file is kept, below the models' folder."""
        return "/".join([self.repo, self.commit[:12], *([self.name] if self.name else []), self.file])


@dataclass(frozen=True)
class Model(Pinned):
    """One ONNX classifier: `labels[i]` is the probability at `output[:, i]`, for any batch size."""

    kind: ClassVar[str] = "model"
    #: What a Task out of GPU memory says to do.
    out_of_gpu_memory: ClassVar[str] = "use a smaller `batchSize`, or the `cpu` build"
    #: The graph's output: every classifier's is named so.
    output: ClassVar[str] = "output"

    id: str
    title: str
    repo: str
    commit: str
    name: str
    sha256: str
    labels: tuple[str, ...]
    #: How it takes an image: the classifiers squash it to their input, its aspect ignored. A model whose graph takes
    #: any height and width gets imgutils' 384×384.
    preprocessing: Preprocessing
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
    preprocessing=IMGUTILS_384,
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
    preprocessing=IMGUTILS_384,
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


@dataclass(frozen=True)
class TagList(Pinned):
    """A tagger's tag list, beside its model: `selected_tags.csv`'s format, a row per output (`tag_id`, `name`,
    `category`, `count`), in the output's order."""

    #: The tagger's.
    id: str
    repo: str
    commit: str
    name: str
    file: str
    sha256: str

    @property
    def subject(self) -> str:
        return f"the tag list of {self.id}"


@dataclass(frozen=True)
class Category:
    """A kind of tag a tagger gives: its `name` in results and settings, its `code` in the tag list's `category`
    column, and the `threshold` its tags' scores must be over when a Node sets none."""

    name: str
    code: int
    threshold: float


@dataclass(frozen=True)
class Tagger(Pinned):
    """One ONNX tagger: a score per tag of its tag list, `output[:, i]` for the list's row `i`, for any batch size. A
    tag of a category it names is handed on when its score is over that category's threshold; the tags whose code is
    `rating_code` are its ratings, of which the highest is the label (a rating has no threshold); any other tag is
    never handed on."""

    kind: ClassVar[str] = "tagger"
    #: What a Task out of GPU memory says to do.
    out_of_gpu_memory: ClassVar[str] = "use a smaller tagger, a smaller `tagBatchSize`, or the `cpu` build"

    id: str
    title: str
    repo: str
    commit: str
    sha256: str
    tags_sha256: str
    preprocessing: Preprocessing
    categories: tuple[Category, ...]
    #: The tag list's code of the ratings; `None` for a tagger without them.
    rating_code: int | None
    #: Whether the output is a probability per tag already; else a logit, sigmoided here.
    sigmoided: bool
    file: str = "model.onnx"
    tags_file: str = "selected_tags.csv"
    #: Both files are at the repo's top.
    name: str = ""
    #: The graph's output, `[batch, tags]`; its input is `input`.
    output: str = "output"
    #: The precision of its weights, `fp32` or `fp16`: fp16 on the CPU is no quicker, which its warning says.
    precision: str = "fp32"
    #: The ratings as the result names them, `(tag list's name, result's name)` in the result's order, for a tagger
    #: whose tag list names them otherwise than WD's `general`, `sensitive`, `questionable` and `explicit`: so a Flow
    #: that routes on the rating works with any tagger. Empty to keep the tag list's names and order.
    rating_names: tuple[tuple[str, str], ...] = ()

    @property
    def tag_list(self) -> TagList:
        return TagList(self.id, self.repo, self.commit, self.name, self.tags_file, self.tags_sha256)


@dataclass(frozen=True)
class Labels:
    """A tag list as read: each output's tag `names` and category `codes`, in the output's order."""

    names: tuple[str, ...]
    codes: tuple[int, ...]


def labels_from_csv(text: str) -> Labels:
    """A tag list in `selected_tags.csv`'s format, its header naming at least `name` and `category`."""
    rows = list(csv.DictReader(io.StringIO(text)))
    return Labels(tuple(row["name"] for row in rows), tuple(int(row["category"]) for row in rows))


WD_EVA02_LARGE = Tagger(
    id="wd-eva02-large",
    title="WD EVA02-large tagger v3 (SmilingWolf): general tags, characters and a rating",
    repo="SmilingWolf/wd-eva02-large-tagger-v3",
    commit="c5303bb7139430db980e4c680a778fe79d72b541",
    sha256="9e768793060c7939b277ccb382783e8670e8a042d29d77aa736be0c8cc898bfc",
    tags_sha256="298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217",
    # As the authors' Space (`SmilingWolf/wd-tagger`, `app.py`) prepares an image.
    preprocessing=Preprocessing(
        size=(448, 448),
        pad=WHITE,
        resample=Image.Resampling.BICUBIC,
        channels=BGR,
        layout=NHWC,
        scale=1.0,
        mean=0.0,
        std=1.0,
    ),
    # The Space's defaults.
    categories=(Category("general", 0, 0.35), Category("character", 4, 0.85)),
    rating_code=9,
    sigmoided=True,
)

#: PixAI v1.0, by its authors (`pixai-labs/pixai-tagger-v1.0` at `9fe10addf9326e292da8a85a98ea74cd91b41771`), who
#: ship it as safetensors only: run as Mexes's ONNX conversion, which ticket 15 checked against their own pipeline.
#: Its fp16 file is another entry, `PIXAI_FP16`, with the same thresholds: its scores differ by a few thousandths.
PIXAI = Tagger(
    id="pixai",
    title="PixAI tagger v1.0 (PixAI; ONNX by Mexes): general tags, characters, copyrights, styles, meta and a rating",
    repo="Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8",
    commit="0800778563144a0e6fdf41ddadd84aae3cb0dbcf",
    sha256="1cae6083f07be1e1757125802566c767005ae883e4504c00db4ee1ef93486dad",
    # Mexes's: the names and order of the authors' `config.json`, with Danbooru's category codes.
    tags_sha256="a9455cbf0a910d4a3890739f2f70bd986278f4594285ef1049121a03896e2a6d",
    # As the authors' `RescalePadProcessor` prepares an image (not as Mexes's README says, bicubic).
    preprocessing=Preprocessing(
        size=(1008, 1008),
        pad=BLACK,
        resample=Image.Resampling.BILINEAR,
        channels=RGB,
        layout=NCHW,
        scale=255.0,
        mean=0.5,
        std=0.5,
        long_side_first=True,
    ),
    # The authors' `category_best_threshold`. Their rating threshold (0.41) is left out: the rating's label is its
    # highest, as WD's is.
    categories=(
        Category("general", 0, 0.17),
        Category("character", 4, 0.27),
        Category("copyright", 3, 0.24),
        Category("style", 1, 0.15),
        Category("meta", 5, 0.17),
    ),
    rating_code=9,
    # The conversion ends in a Sigmoid.
    sigmoided=True,
    output="prediction",
    rating_names=(
        ("rating:g", "general"),
        ("rating:s", "sensitive"),
        ("rating:q", "questionable"),
        ("rating:e", "explicit"),
    ),
)

PIXAI_FP16 = replace(
    PIXAI,
    id="pixai-fp16",
    title="PixAI tagger v1.0 in fp16 (PixAI; ONNX by Mexes): half the size, for a GPU",
    file="model_fp16.onnx",
    sha256="cd668e6760bb58ec400ce1b0286e5c9133f83b543caac6a378406b7d56f8c303",
    precision="fp16",
)

#: What a Node's `tagger` setting may name.
TAGGERS: dict[str, Tagger] = {tagger.id: tagger for tagger in [WD_EVA02_LARGE, PIXAI, PIXAI_FP16]}

DEFAULT_TAGGER = WD_EVA02_LARGE.id


def thresholds_of(tagger: Tagger, given: Mapping[str, float] | None) -> dict[str, float]:
    """The threshold of each of the tagger's categories: `given`'s, where it names the category, else the tagger's
    default. A category the tagger lacks is an error naming those it has."""
    given = given or {}
    if not isinstance(given, Mapping):
        raise ValueError(f"`thresholds` is not an object of category: threshold, but {given!r}")
    names = [category.name for category in tagger.categories]
    for name, threshold in given.items():
        if name not in names:
            raise ValueError(f"{tagger.id} has no category `{name}`: its categories are {', '.join(names)}")
        if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not 0 <= threshold <= 1:
            raise ValueError(f"The threshold of `{name}` is not a number from 0 to 1: {threshold!r}")
    return {category.name: given.get(category.name, category.threshold) for category in tagger.categories}


def tag(
    tagger: Tagger, labels: Labels, scores: np.ndarray, thresholds: Mapping[str, float], max_tags: int | None = None
) -> dict:
    """What a tagger says of one image, from its output (`[len(labels.names)]`): the tags over their category's
    threshold, highest first, at most `max_tags` of them (all when `None`); and its rating, `None` for a tagger
    without one."""
    if not tagger.sigmoided:
        scores = 1 / (1 + np.exp(-scores.astype(np.float64)))
    codes = np.asarray(labels.codes)
    tags = []
    for category in tagger.categories:
        over = np.flatnonzero((codes == category.code) & (scores > thresholds[category.name]))
        tags += [{"name": labels.names[i], "score": float(scores[i]), "category": category.name} for i in over]
    tags.sort(key=lambda t: t["score"], reverse=True)
    if max_tags is not None:
        tags = tags[:max_tags]
    rating = None
    if tagger.rating_code is not None:
        rated = {labels.names[i]: float(scores[i]) for i in np.flatnonzero(codes == tagger.rating_code)}
        if tagger.rating_names:
            if lacking := [theirs for theirs, _ in tagger.rating_names if theirs not in rated]:
                raise ValueError(
                    f"The tag list of {tagger.id} has no rating `{lacking[0]}`: its ratings are {', '.join(rated)}"
                )
            rated = {ours: rated[theirs] for theirs, ours in tagger.rating_names}
        rating = {"label": max(rated, key=rated.get), "scores": rated}
    return {"tagger": tagger.id, "tags": tags, "rating": rating}
