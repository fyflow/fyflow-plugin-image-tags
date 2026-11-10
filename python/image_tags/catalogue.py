"""The classifiers this Plugin knows: each model pinned to a Hugging Face commit and its file's sha256.

Nothing outside this catalogue ever loads. An entry's labels, input size and output are read from the model's own
files when the entry is written; `scripts/golden.py` checks them against `meta.json` and the ONNX graph, and the golden
tests hold the scores to imgutils.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    """One ONNX classifier: `labels[i]` is the probability at `output[:, i]`, for any batch size."""

    id: str
    title: str
    repo: str
    commit: str
    name: str
    sha256: str
    labels: tuple[str, ...]
    #: The model's input, width and height: an image is squashed to it, its aspect ignored.
    size: tuple[int, int]
    file: str = "model.onnx"

    @property
    def remote_path(self) -> str:
        """Where the file is on Hugging Face (or a mirror), below the endpoint."""
        return f"{self.repo}/resolve/{self.commit}/{self.name}/{self.file}"

    @property
    def local_path(self) -> str:
        """Where the file is kept, below the models' folder. The commit is cut to 12 characters, which still names
        it, to stay under Windows' 260-character paths when long paths are off."""
        return f"{self.repo}/{self.commit[:12]}/{self.name}/{self.file}"


MODELS: dict[str, Model] = {
    model.id: model
    for model in [
        Model(
            id="anime_real",
            title="Anime or real (deepghs anime_real_cls)",
            repo="deepghs/anime_real_cls",
            commit="097a6c1e9e62866a435810c883a45a7cb2c7077d",
            name="mobilenetv3_v1.4_dist",
            sha256="c02d68cc7f56dbcbcd4f41a10df190385c1f963f22c4eefb8d162869800ed3b0",
            labels=("anime", "real"),
            size=(384, 384),
        ),
    ]
}

DEFAULT_CLASSIFIER = "anime_real"
