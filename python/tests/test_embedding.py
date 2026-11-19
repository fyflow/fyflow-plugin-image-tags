"""A tagger's embedding session (`Models.session(..., embedding=True)`): the same graph with a second output, each
image's embedding, the data input of the tagger's head.

The fast tests run the stand-in tagger (`stand_in.py`), whose head is a `Gemm` named as `wd-eva02-large`'s; the slow
ones the real taggers, only when they are in `python/.cache/models/` (as `test_tag_image_golden.py`'s).
"""

import hashlib
import threading
from dataclasses import replace

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper, shape_inference

from image_tags.catalogue import ANIME_REAL, TAGGERS
from image_tags.models import EMBEDDING, Cancelled, GraphChanged, Models, OutOfGpuMemory
from image_tags.preprocess import load_image, to_input
from tests.conftest import IMAGES, TAGGER_CACHE, Context, wait_until
from tests.stand_in import FEATURE_NODE, FEATURES, TAGS, lay_out, make
from tests.test_tag_image import NEEDS_MORE, OUT_OF_MEMORY, on_a_gpu  # noqa: F401  (`on_a_gpu` is a fixture)
from tests.test_tag_image_golden import tagger_params

#: Mean colours, BGR from 0 to 255: the stand-in's embedding starts with the colour scaled to [0, 1].
COLOURS = [(0, 0, 0), (255, 255, 255), (30, 120, 220)]


def batch_of(colours=COLOURS) -> np.ndarray:
    """Images of one colour each, as the stand-in takes them."""
    return np.stack([np.full((448, 448, 3), colour, dtype=np.float32) for colour in colours])


def models_with(tmp_path, tagger, files) -> Models:
    """Models reading a model folder that holds `tagger`'s files, from `files`."""
    root = tmp_path / "models"
    lay_out(tagger, files, root, remote=False)
    return Models(str(tmp_path / "data"), model_dir=str(root))


@pytest.fixture
def stand_in_models(stand_in, tmp_path):
    made = models_with(tmp_path, stand_in, tmp_path / "stand-in")
    yield made
    made.close()


# Found by the head's name.


def test_the_embedding_is_the_input_of_the_head_named_by_head_node(stand_in, stand_in_models):
    session = stand_in_models.session(stand_in, embedding=True)

    assert [output.name for output in session.get_outputs()] == ["output", EMBEDDING]
    _, embeddings = stand_in_models.scores_and_embeddings(stand_in, batch_of())
    # The stand-in's features start with the image's mean colour, scaled: as ONNX Runtime sums 448 × 448 pixels in
    # float32, to a few thousandths.
    assert embeddings[:, :3] == pytest.approx(np.array(COLOURS) / 255, abs=0.005)


def test_its_width_is_the_heads_input_width(stand_in, stand_in_models):
    (_, embedding) = stand_in_models.session(stand_in, embedding=True).get_outputs()
    _, embeddings = stand_in_models.scores_and_embeddings(stand_in, batch_of())

    assert (embedding.type, embedding.shape) == ("tensor(float)", ["batch", FEATURES])
    assert (embeddings.dtype, embeddings.shape) == (np.float32, (len(COLOURS), FEATURES))


def test_the_scores_are_those_of_the_plain_session(stand_in, stand_in_models):
    batch = batch_of()

    scores, _ = stand_in_models.scores_and_embeddings(stand_in, batch)

    assert scores.shape == (len(COLOURS), TAGS)
    np.testing.assert_allclose(scores, stand_in_models.scores(stand_in, batch), atol=1e-6)


def test_a_real_image_gives_the_same_scores_both_ways(stand_in, stand_in_models):
    batch = np.stack([to_input(load_image(str(IMAGES / "anime-02-train-window.jpg")), stand_in.preprocessing)])

    scores, embeddings = stand_in_models.scores_and_embeddings(stand_in, batch)

    np.testing.assert_allclose(scores, stand_in_models.scores(stand_in, batch), atol=1e-6)
    assert embeddings.shape == (1, FEATURES)


def test_an_fp16_head_input_comes_out_as_float32(tmp_path):
    tagger = make(tmp_path / "stand-in", fp16=True)
    fp16 = models_with(tmp_path, tagger, tmp_path / "stand-in")

    (_, embedding) = fp16.session(tagger, embedding=True).get_outputs()
    scores, embeddings = fp16.scores_and_embeddings(tagger, batch_of())

    assert embedding.type == "tensor(float)"
    assert (embeddings.dtype, embeddings.shape) == (np.float32, (len(COLOURS), FEATURES))
    # The head's input is the features rounded to float16, cast back: as the plain stand-in's, to a few thousandths.
    assert embeddings[:, :3] == pytest.approx(np.array(COLOURS) / 255, abs=0.005)
    # In so small a graph, the extra output moves where ONNX Runtime rounds to float16: by about 1e-4, within float16's
    # own (this stand-in's scores are 6e-4 from the float32 one's). The real `pixai-fp16`'s match to 1e-5 (below).
    np.testing.assert_allclose(scores, fp16.scores(tagger, batch_of()), atol=1e-3)
    fp16.close()


# Kept beside the plain session, made in memory.


def test_it_is_kept_apart_from_the_plain_session_which_stays_as_it_was(stand_in, stand_in_models):
    with_it = stand_in_models.session(stand_in, embedding=True)
    plain = stand_in_models.session(stand_in)

    assert [output.name for output in plain.get_outputs()] == ["output"]
    assert with_it is not plain
    assert stand_in_models.session(stand_in, embedding=True) is with_it
    assert stand_in_models.session(stand_in) is plain
    assert set(stand_in_models._sessions) == set(stand_in_models._embedding_sessions) == {stand_in.id}


def test_nothing_is_written_to_disk(stand_in, stand_in_models, tmp_path):
    before = sorted(tmp_path.rglob("*"))

    stand_in_models.scores_and_embeddings(stand_in, batch_of())

    assert sorted(tmp_path.rglob("*")) == before


def test_closing_drops_it_and_it_is_made_again(stand_in, stand_in_models):
    first = stand_in_models.session(stand_in, embedding=True)

    stand_in_models.close()

    assert stand_in_models._embedding_sessions == {}
    assert stand_in_models.session(stand_in, embedding=True) is not first


def test_only_a_tagger_has_an_embedding(data_dir):
    with pytest.raises(ValueError, match="anime_real is not a tagger"):
        Models(str(data_dir)).session(ANIME_REAL, embedding=True)


def test_where_it_runs_is_told_of_it_too(stand_in, stand_in_models):
    assert stand_in_models.runs_on_cpu(stand_in, embedding=True)
    assert stand_in_models._sessions == {}


# A graph that is not what the entry says.


def test_a_head_node_the_graph_lacks_fails_saying_the_graph_changed(stand_in, tmp_path):
    tagger = replace(stand_in, head_node="/core_model/head/Nothing")
    lacking = models_with(tmp_path, tagger, tmp_path / "stand-in")

    with pytest.raises(GraphChanged, match="The graph of the tagger stand-in changed: it has no node "):
        lacking.session(tagger, embedding=True)
    assert lacking._embedding_sessions == {}


def test_a_head_node_the_scores_do_not_come_from_fails_saying_the_graph_changed(stand_in, tmp_path):
    tagger = replace(stand_in, head_node=FEATURE_NODE)
    other = models_with(tmp_path, tagger, tmp_path / "stand-in")

    with pytest.raises(GraphChanged) as failed:
        other.session(tagger, embedding=True)
    assert "its scores come from `/core_model/head/Gemm` (Gemm), not `/features/MatMul`" in str(failed.value)
    assert "`head_node` must name the Gemm (or MatMul) that gives its scores" in str(failed.value)


def test_a_head_input_that_is_not_batch_by_width_fails(stand_in, tmp_path):
    """A MatMul head over the mean colour kept as `[batch, 1, 1, 3]`."""
    nodes = [
        helper.make_node("ReduceMean", ["input"], ["means"], axes=[1, 2], keepdims=1),
        helper.make_node("MatMul", ["means", "weights"], ["product"], name="/head/MatMul"),
        helper.make_node("Add", ["product", "bias"], ["logits"]),
        helper.make_node("Sigmoid", ["logits"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "a head over four dimensions",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 448, 448, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 1, 1, TAGS])],
        initializer=[
            numpy_helper.from_array(np.zeros((3, TAGS), dtype=np.float32), "weights"),
            numpy_helper.from_array(np.zeros(TAGS, dtype=np.float32), "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    files = tmp_path / "four"
    files.mkdir()
    (files / "model.onnx").write_bytes(data := shape_inference.infer_shapes(model).SerializeToString())
    (files / "selected_tags.csv").write_bytes((tmp_path / "stand-in" / "selected_tags.csv").read_bytes())
    tagger = replace(stand_in, head_node="/head/MatMul", sha256=hashlib.sha256(data).hexdigest())
    four = models_with(tmp_path, tagger, files)

    with pytest.raises(GraphChanged, match="`/head/MatMul`'s input `means` has 4 dimensions, not 2"):
        four.session(tagger, embedding=True)


# The same paths as the plain session.


def test_it_is_downloaded_as_the_plain_session_is(tagger_hub, stand_in, data_dir):
    fetching = Models(str(data_dir))

    _, embeddings = fetching.scores_and_embeddings(stand_in, batch_of(), Context())

    assert embeddings.shape == (len(COLOURS), FEATURES)
    assert sorted(tagger_hub.requests) == [f"/{stand_in.remote_path}"]


def test_a_cancel_while_it_downloads_stops_waiting(tagger_hub, stand_in, data_dir):
    # The stand-in is smaller than what a held hub sends before it stalls.
    tagger_hub.HEAD_BYTES = 16
    tagger_hub.hold()
    ctx = Context()
    outcome = []

    def make():
        try:
            Models(str(data_dir)).session(stand_in, ctx, embedding=True)
        except Exception as error:  # noqa: BLE001  (what it raised is the outcome)
            outcome.append(error)

    making = threading.Thread(target=make, daemon=True)
    making.start()
    wait_until(lambda: tagger_hub.requests, what="the download")
    ctx.cancelled = True
    making.join(10)

    assert not making.is_alive()
    assert [type(error) for error in outcome] == [Cancelled]


def test_a_forward_pass_out_of_gpu_memory_says_what_to_do(tagger_hub, stand_in, data_dir, on_a_gpu):
    on_a_gpu.run_with = RuntimeError(OUT_OF_MEMORY)

    with pytest.raises(OutOfGpuMemory) as failed:
        Models(str(data_dir)).scores_and_embeddings(stand_in, batch_of())
    assert str(failed.value) == NEEDS_MORE


def test_making_it_out_of_gpu_memory_says_what_to_do(tagger_hub, stand_in, data_dir, on_a_gpu):
    on_a_gpu.made_with = RuntimeError("CUDA failure 2: out of memory ; GPU=0 ; hostname=x ; expr=cudaMalloc(&p, size);")

    with pytest.raises(OutOfGpuMemory):
        Models(str(data_dir)).session(stand_in, embedding=True)


# The real taggers.


@pytest.fixture(scope="module")
def real_models(tmp_path_factory):
    made = Models(str(tmp_path_factory.mktemp("data")), model_dir=str(TAGGER_CACHE))
    yield made
    made.close()


@pytest.mark.slow
@pytest.mark.parametrize("tagger_id", tagger_params(["wd-eva02-large", "pixai-fp16"]))
def test_a_real_tagger_gives_a_1024_wide_embedding_and_the_scores_of_the_plain_session(real_models, tagger_id):
    tagger = TAGGERS[tagger_id]
    paths = [IMAGES / "anime-02-train-window.jpg", IMAGES / "cg-02-sintel.jpg"]
    batch = np.stack([to_input(load_image(str(path)), tagger.preprocessing) for path in paths])

    scores, embeddings = real_models.scores_and_embeddings(tagger, batch)
    plain = real_models.scores(tagger, batch)

    assert (embeddings.dtype, embeddings.shape) == (np.float32, (2, 1024))
    assert np.isfinite(embeddings).all()
    assert scores.shape == plain.shape
    np.testing.assert_allclose(scores, plain, atol=1e-5)
    # Two different images, two different embeddings.
    assert not np.allclose(embeddings[0], embeddings[1])
    real_models.close()
