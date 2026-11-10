"""The `style` preset's rule and the `minConfidence` threshold, on scores written by hand.

The expected values are worked out by hand from the spec's rule, not recomputed the way the code does.
"""

import numpy as np
import pytest

from image_tags.catalogue import CLASSIFIERS, MODELS, UNCERTAIN, answer

STYLE = CLASSIFIERS["style"]


def frames(*scores):
    """Each model's scores for the frames, `[n, len(labels)]` by model id, from one dict per frame and model:
    `{"anime_real": {...}, "anime_classification": {...}}`."""
    return {
        model: np.array([[frame[model][label] for label in MODELS[model].labels] for frame in scores])
        for model in ("anime_real", "anime_classification")
    }


def frame(real, **kinds):
    return {
        "anime_real": {"anime": 1 - real, "real": real},
        "anime_classification": {"3d": 0, "bangumi": 0, "comic": 0, "illustration": 0, "not_painting": 0, **kinds},
    }


def style_of(*scores):
    """`style`'s scores by label, one dict per frame."""
    return [dict(zip(STYLE.labels, row, strict=True)) for row in STYLE.score(frames(*scores)).tolist()]


def test_style_has_its_labels_and_needs_both_models():
    assert STYLE.labels == ("anime", "cg", "live_action")
    assert [model.id for model in STYLE.models] == ["anime_real", "anime_classification"]
    assert STYLE.min_confidence == 0.55


def test_style_shares_what_is_not_live_action_by_the_four_painted_kinds():
    # real .5: live action .5^4 = .0625, .9375 left. The four painted kinds sum to .8: renormalised, 3d .375,
    # bangumi .25, comic .125, illustration .25.
    (scores,) = style_of(frame(0.5, **{"3d": 0.3}, bangumi=0.2, comic=0.1, illustration=0.2, not_painting=0.2))

    assert scores == pytest.approx({"live_action": 0.0625, "cg": 0.9375 * 0.375, "anime": 0.9375 * 0.625})
    assert scores == pytest.approx({"live_action": 0.0625, "cg": 0.3515625, "anime": 0.5859375})


def test_style_counts_live_action_only_when_anime_real_is_very_sure():
    # real .8 is .4096 live action: a painted-looking frame stays anime (.5904).
    (scores,) = style_of(frame(0.8, bangumi=0.5, illustration=0.5))

    assert scores == pytest.approx({"live_action": 0.4096, "cg": 0.0, "anime": 0.5904})


def test_style_ignores_how_sure_anime_classification_is_that_it_is_not_a_painting():
    # The same four in the same proportions (sum .4), not_painting .6: the same answer as above.
    (scores,) = style_of(frame(0.5, **{"3d": 0.15}, bangumi=0.1, comic=0.05, illustration=0.1, not_painting=0.6))

    assert scores == pytest.approx({"live_action": 0.0625, "cg": 0.3515625, "anime": 0.5859375})


def test_style_scores_each_frame_on_its_own():
    # Frame 2: real .9, live action .6561, .3439 left; the four sum to .7, 3d 5/7, the anime kinds 2/7.
    first, second = style_of(
        frame(0.5, **{"3d": 0.3}, bangumi=0.2, comic=0.1, illustration=0.2, not_painting=0.2),
        frame(0.9, **{"3d": 0.5}, bangumi=0.1, comic=0.0, illustration=0.1, not_painting=0.3),
    )

    assert first == pytest.approx({"live_action": 0.0625, "cg": 0.3515625, "anime": 0.5859375})
    assert second == pytest.approx({"live_action": 0.6561, "cg": 0.3439 * 5 / 7, "anime": 0.3439 * 2 / 7})


def test_style_shares_cartoon_kinds_as_anime():
    (scores,) = style_of(frame(0.0, bangumi=0.25, comic=0.25, illustration=0.5))

    assert scores == pytest.approx({"live_action": 0.0, "cg": 0.0, "anime": 1.0})


@pytest.mark.parametrize("painted", [0.0, 1e-9])
def test_style_splits_what_is_not_live_action_evenly_when_no_painted_kind_is_left(painted):
    # real .5: live action .0625, the .9375 left split in two.
    (scores,) = style_of(
        frame(0.5, **{"3d": painted}, bangumi=painted, comic=painted, illustration=painted, not_painting=1.0)
    )

    assert scores == pytest.approx({"live_action": 0.0625, "cg": 0.46875, "anime": 0.46875})


def test_a_top_score_below_the_preset_s_own_threshold_is_uncertain_and_keeps_its_scores():
    result = answer(STYLE, np.array([0.5, 0.3, 0.2]))

    assert result["tag"] == UNCERTAIN
    assert result["confidence"] == 0.5
    assert result["scores"] == {"anime": 0.5, "cg": 0.3, "live_action": 0.2}
    assert result["labels"] == ["anime", "cg", "live_action"]
    assert result["classifier"] == "style"
    assert result["models"] == ["anime_real", "anime_classification"]


def test_a_top_score_that_reaches_the_threshold_is_the_tag():
    assert answer(STYLE, np.array([0.55, 0.35, 0.1]))["tag"] == "anime"
    assert answer(STYLE, np.array([0.1, 0.2, 0.7]))["tag"] == "live_action"


def test_a_min_confidence_given_replaces_the_classifier_s_own():
    assert answer(STYLE, np.array([0.5, 0.3, 0.2]), 0.4)["tag"] == "anime"
    assert answer(STYLE, np.array([0.34, 0.33, 0.33]), 0)["tag"] == "anime"
    assert answer(STYLE, np.array([0.1, 0.8, 0.1]), 0.9)["tag"] == UNCERTAIN


def test_a_raw_model_never_answers_uncertain_unless_asked():
    model = CLASSIFIERS["anime_real"]

    assert model.min_confidence == 0
    assert answer(model, np.array([0.51, 0.49]))["tag"] == "anime"
    assert answer(model, np.array([0.51, 0.49]), 0.9) == {
        "classifier": "anime_real",
        "labels": ["anime", "real"],
        "tag": UNCERTAIN,
        "confidence": 0.51,
        "scores": {"anime": 0.51, "real": 0.49},
        "models": ["anime_real"],
    }
