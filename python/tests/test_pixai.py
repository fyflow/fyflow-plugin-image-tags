"""PixAI v1.0 in the catalogue: `pixai` (fp32) and `pixai-fp16`, Mexes's ONNX conversion of the authors' weights,
checked against their pipeline by ticket 15 (`.scratch/image-tags/research/pixai-onnx-check.md`).

The real models' scores are held to that pipeline's by `test_tag_image_golden_pixai.py`; these tests need no model.
"""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from image_tags.catalogue import PIXAI, PIXAI_FP16, TAGGERS, WD_EVA02_LARGE, Category, Labels, tag, thresholds_of
from image_tags.preprocess import BLACK, NCHW, RGB, Preprocessing, to_input
from image_tags.worker import TagImage, ran_on_the_cpu
from tests.conftest import IMAGES, Task, run

MEXES = "Mexes/pixai-tagger-v1.0-onnx-fp32-fp16-int8"
COMMIT = "0800778563144a0e6fdf41ddadd84aae3cb0dbcf"
DEFAULTS = {"general": 0.17, "character": 0.27, "copyright": 0.24, "style": 0.15, "meta": 0.17}


# The catalogue entries.


@pytest.mark.parametrize(
    ("tagger", "file", "sha256"),
    [
        (PIXAI, "model.onnx", "1cae6083f07be1e1757125802566c767005ae883e4504c00db4ee1ef93486dad"),
        (PIXAI_FP16, "model_fp16.onnx", "cd668e6760bb58ec400ce1b0286e5c9133f83b543caac6a378406b7d56f8c303"),
    ],
)
def test_pixai_is_pinned_to_mexes_s_conversion_each_precision_its_own_file(tagger, file, sha256):
    assert TAGGERS[tagger.id] is tagger
    assert tagger.kind == "tagger"
    assert tagger.repo == MEXES
    assert tagger.commit == COMMIT
    assert tagger.file == file
    assert tagger.remote_path == f"{MEXES}/resolve/{COMMIT}/{file}"
    assert tagger.local_path == f"{MEXES}/080077856314/{file}"
    assert tagger.sha256 == sha256
    assert tagger.tag_list.local_path == f"{MEXES}/080077856314/selected_tags.csv"
    assert tagger.tag_list.sha256 == "a9455cbf0a910d4a3890739f2f70bd986278f4594285ef1049121a03896e2a6d"


def test_the_ids_are_pixai_and_pixai_fp16_and_wd_stays_the_default():
    assert list(TAGGERS) == ["wd-eva02-large", "pixai", "pixai-fp16"]
    assert (PIXAI.id, PIXAI.precision, PIXAI_FP16.id, PIXAI_FP16.precision) == ("pixai", "fp32", "pixai-fp16", "fp16")
    assert WD_EVA02_LARGE.precision == "fp32"


@pytest.mark.parametrize("tagger", [PIXAI, PIXAI_FP16])
def test_pixai_s_categories_and_defaults_are_the_authors_for_both_precisions(tagger):
    # Danbooru's codes in the tag list's `category` column, as WD's.
    assert tagger.categories == (
        Category("general", 0, 0.17),
        Category("character", 4, 0.27),
        Category("copyright", 3, 0.24),
        Category("style", 1, 0.15),
        Category("meta", 5, 0.17),
    )
    assert thresholds_of(tagger, None) == DEFAULTS
    assert tagger.rating_code == 9


@pytest.mark.parametrize("tagger", [PIXAI, PIXAI_FP16])
def test_pixai_s_output_is_named_prediction_and_already_sigmoided(tagger):
    assert tagger.output == "prediction"
    assert tagger.sigmoided
    assert WD_EVA02_LARGE.output == "output"


@pytest.mark.parametrize("tagger", [PIXAI, PIXAI_FP16])
def test_pixai_takes_its_long_side_resized_to_1008_then_padded_black_rgb_nchw_from_minus_1_to_1(tagger):
    assert tagger.preprocessing == Preprocessing(
        size=(1008, 1008),
        pad=BLACK,
        resample=Image.Resampling.BILINEAR,
        channels=RGB,
        layout=NCHW,
        scale=255.0,
        mean=0.5,
        std=0.5,
        long_side_first=True,
    )


# Preprocessing: the long side first, in float, then the padding.

#: PixAI's recipe on an 8×8 input, so that each pixel of a small synthetic image can be looked at.
SMALL = replace(PIXAI.preprocessing, size=(8, 8))
RED = (255, 0, 0)


def test_a_wide_image_s_long_side_is_resized_to_the_input_then_it_is_centred_on_black():
    pixels = to_input(Image.new("RGB", (4, 2), RED), SMALL)

    assert pixels.shape == (3, 8, 8)
    assert pixels.dtype == np.float32
    # 4×2 to 8×4, then 2 rows of black above and 2 below: black is -1 once normalised, red (1, -1, -1).
    assert (pixels[:, :2] == -1).all()
    assert (pixels[:, 6:] == -1).all()
    assert pixels[:, 2:6].transpose(1, 2, 0).tolist() == [[[1.0, -1.0, -1.0]] * 8] * 4


def test_the_new_size_is_truncated_and_the_odd_column_of_padding_goes_after_the_image():
    # 3×7 to 8 high: 3 × 8/7 = 3.43, truncated to 3 wide; (8 - 3) // 2 = 2 columns before it, 3 after.
    pixels = to_input(Image.new("RGB", (3, 7), RED), SMALL)

    red = (pixels[0] == 1).all(axis=0)
    assert red.tolist() == [False, False, True, True, True, False, False, False]
    assert (pixels[:, :, :2] == -1).all()
    assert (pixels[:, :, 5:] == -1).all()


def test_the_resize_is_done_in_float_not_rounded_to_whole_levels():
    image = Image.new("RGB", (2, 1))
    image.putpixel((1, 0), (1, 1, 1))

    pixels = to_input(image, SMALL)

    # 2×1 to 8×4: bilinear between levels 0 and 1 gives fractions of a level, which a uint8 resize would round away.
    levels = (pixels[0, 2:6] * 0.5 + 0.5) * 255
    assert ((levels > 0.01) & (np.abs(levels - np.round(levels)) > 0.01)).any()


def test_an_image_already_the_input_s_size_is_not_resized():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)

    pixels = to_input(Image.fromarray(rgb), SMALL)

    assert pixels == pytest.approx((rgb.transpose(2, 0, 1) / 255 - 0.5) / 0.5, abs=1e-6)


def research_resize(image: Image.Image) -> np.ndarray:
    """`resize_then_pad_float` of ticket 15's `pixai-compare.py`, the preprocessing that matched the authors'
    pipeline: to float from 0 to 1, each channel resized as a PIL "F" image, centred on black, `(x - 0.5) / 0.5`."""
    size = 1008
    w, h = image.size
    r = min(size / h, size / w)
    nw, nh = int(w * r), int(h * r)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    out = np.zeros((3, size, size), dtype=np.float32)
    left, top = (size - nw) // 2, (size - nh) // 2
    for c in range(3):
        channel = Image.fromarray(np.ascontiguousarray(pixels[:, :, c]))
        if (w, h) != (size, size):
            channel = channel.resize((nw, nh), Image.Resampling.BILINEAR)
        out[c, top : top + nh, left : left + nw] = np.asarray(channel)
    return (out - 0.5) / 0.5


@pytest.mark.parametrize("name", ["anime-07-yakitori-stall-portrait.jpg", "cg-02-sintel.jpg", "tiny.jpg"])
def test_pixai_s_input_is_the_one_ticket_15_checked_against_the_authors_pipeline(name):
    with Image.open(IMAGES / name) as image:
        rgb = image.convert("RGB")

    assert np.abs(to_input(rgb, PIXAI.preprocessing) - research_resize(rgb)).max() < 1e-5


# The rating.

#: PixAI's tag list's end: a general tag, then its four ratings, in its output's order.
LABELS = Labels(names=("1girl", "rating:s", "rating:g", "rating:q", "rating:e"), codes=(0, 9, 9, 9, 9))


def test_pixai_s_ratings_are_named_as_wd_s_in_the_same_order_the_highest_the_label():
    result = tag(PIXAI, LABELS, np.array([0.9, 0.3, 0.6, 0.08, 0.02], dtype=np.float32), DEFAULTS)

    assert result["rating"]["label"] == "general"
    assert result["rating"]["scores"] == pytest.approx(
        {"general": 0.6, "sensitive": 0.3, "questionable": 0.08, "explicit": 0.02}
    )
    assert list(result["rating"]["scores"]) == ["general", "sensitive", "questionable", "explicit"]


def test_the_rating_is_the_highest_even_under_the_authors_rating_threshold():
    # Upstream's 0.41 for the ratings would leave none here; the Plugin's rating always has a label.
    result = tag(PIXAI, LABELS, np.array([0.9, 0.4, 0.3, 0.2, 0.1], dtype=np.float32), DEFAULTS)

    assert result["rating"]["label"] == "sensitive"


def test_a_rating_is_never_a_threshold_a_node_may_set():
    with pytest.raises(ValueError, match=r"^pixai has no category `rating`: its categories are general, character, "):
        thresholds_of(PIXAI, {"rating": 0.41})


# The CPU warning.


def test_the_cpu_warning_of_fp16_names_fp16_on_the_cpu_as_slow():
    warning = ran_on_the_cpu(PIXAI_FP16)

    assert warning.startswith("pixai-fp16 ran on the CPU, where a tagger is slow")
    assert "fp16 on the CPU is slow too" in warning
    assert "fp16" not in ran_on_the_cpu(PIXAI)


# A tagger whose output is named otherwise, through the Worker.


def test_a_tagger_s_output_is_read_by_the_name_its_entry_gives(tmp_path, monkeypatch, config):
    from tests import stand_in

    tagger = stand_in.make(tmp_path / "stand-in", output="prediction")
    monkeypatch.setitem(TAGGERS, tagger.id, tagger)
    config["modelDir"] = str(tmp_path / "models")
    stand_in.lay_out(tagger, tmp_path / "stand-in", tmp_path / "models", remote=False)

    result = run(TagImage(config), Task({"path": str(IMAGES / "tiny.jpg")}, {"tagger": tagger.id}))

    assert result["tagger"] == "stand-in"
    assert result["rating"]["label"] in ("general", "sensitive", "questionable", "explicit")
