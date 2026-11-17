"""A stand-in tagger, so that the tagging tests need no 1.26 GB download: a tiny ONNX graph with the WD taggers' input
and output (`input` `[b, 448, 448, 3]` 0-255, `output` `[b, tags]` already sigmoided), its scores a fixed linear map of
the image's mean colour, and a tag list in `selected_tags.csv`'s format. Its catalogue entry is `wd-eva02-large`'s,
the files and their sha256 swapped.
"""

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from image_tags.catalogue import WD_EVA02_LARGE, Tagger

#: Its tag list: the four ratings first, as the WD taggers have them, then general tags (one with an underscore, which
#: stays) and a character.
TAG_LIST = """tag_id,name,category,count
9999999,general,9,100
9999998,sensitive,9,100
9999997,questionable,9,100
9999996,explicit,9,100
1,1girl,0,100
2,long_hair,0,100
3,outdoors,0,100
4,simple_background,0,100
5,hatsune_miku,4,100
6,kagamine_rin,4,100
"""
TAGS = 10


def model_bytes(seed: int = 0) -> bytes:
    """The graph: each image's mean colour (BGR, scaled to [0, 1]) times fixed weights, plus a bias, sigmoided."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 4, (3, TAGS)).astype(np.float32)
    bias = rng.normal(0, 1, TAGS).astype(np.float32)
    nodes = [
        helper.make_node("ReduceMean", ["input"], ["means"], axes=[1, 2], keepdims=0),
        helper.make_node("Div", ["means", "scale"], ["scaled"]),
        helper.make_node("MatMul", ["scaled", "weights"], ["product"]),
        helper.make_node("Add", ["product", "bias"], ["logits"]),
        helper.make_node("Sigmoid", ["logits"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "stand-in tagger",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 448, 448, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", TAGS])],
        initializer=[
            numpy_helper.from_array(np.array(255, dtype=np.float32), "scale"),
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(bias, "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


def make(folder: Path, **entry) -> Tagger:
    """Writes the stand-in's files into `folder` (`model.onnx`, `selected_tags.csv`) and gives its catalogue entry,
    with what `entry` changes of it."""
    folder.mkdir(parents=True, exist_ok=True)
    model, tags = model_bytes(), TAG_LIST.encode()
    (folder / "model.onnx").write_bytes(model)
    (folder / "selected_tags.csv").write_bytes(tags)
    return replace(
        WD_EVA02_LARGE,
        id="stand-in",
        title="A stand-in tagger",
        repo="tests/stand-in-tagger",
        commit="0123456789abcdef0123456789abcdef01234567",
        sha256=hashlib.sha256(model).hexdigest(),
        tags_sha256=hashlib.sha256(tags).hexdigest(),
        **entry,
    )


def lay_out(tagger: Tagger, files: Path, root: Path, remote: bool) -> None:
    """Copies the stand-in's files from `files` below `root`, as a hub serves them (`remote`) or as a model folder
    holds them."""
    for pinned in (tagger, tagger.tag_list):
        path = root / (pinned.remote_path if remote else pinned.local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((files / pinned.file).read_bytes())
