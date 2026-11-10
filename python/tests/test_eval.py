"""The eval script's sums (`scripts/eval.py`, run by hand): the confusion matrix, the threshold table and its pick,
the raw models' labels in `style`'s terms, the reference frames' labels and the rules compared. Hand-worked values."""

import importlib
import sys

import numpy as np
import pytest

from tests.conftest import IMAGES, PYTHON

sys.path.insert(0, str(PYTHON / "scripts"))
evaluation = importlib.import_module("eval")


def test_confusion_counts_each_truth_against_what_was_said():
    matrix = evaluation.confusion(
        ["anime", "cg", "cg", "live_action", "cg"],
        ["anime", "live_action", "cg", "live_action", "uncertain"],
        ["anime", "cg", "live_action"],
        ["anime", "cg", "live_action", "uncertain"],
    )
    assert matrix.tolist() == [[1, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0]]


def test_threshold_table_counts_answered_and_correct_and_picks_the_best_with_enough_coverage():
    truths = ["anime", "anime", "cg", "live_action"]
    tops = ["anime", "cg", "cg", "live_action"]
    confidences = [0.9, 0.5, 0.7, 0.6]

    table = evaluation.threshold_table(truths, tops, confidences, [0.4, 0.55, 0.65, 0.8, 0.95])

    assert [(row.answered, row.correct) for row in table] == [(4, 3), (3, 3), (2, 2), (1, 1), (0, 0)]
    assert table[0].accuracy == 0.75 and table[0].coverage == 1.0
    # 100% on what is answered from 0.55 up; the lowest of those answering at least half.
    assert evaluation.best_threshold(table, 0.5).threshold == 0.55
    assert evaluation.best_threshold(table, 0.9).threshold == 0.4
    assert evaluation.best_threshold(table[-1:], 0.0) is None


def test_tags_are_uncertain_below_the_threshold():
    assert evaluation.tags(["anime", "cg"], [0.6, 0.59], 0.6) == ["anime", "uncertain"]


@pytest.mark.parametrize(
    ("model", "label", "style"),
    [
        ("anime_real", "real", "live_action"),
        ("anime_real", "anime", "anime"),
        ("anime_classification", "3d", "cg"),
        ("anime_classification", "comic", "anime"),
        ("anime_classification", "illustration", "anime"),
        ("anime_classification", "bangumi", "anime"),
        ("anime_classification", "not_painting", "live_action"),
    ],
)
def test_raw_labels_map_to_style(model, label, style):
    assert evaluation.mapped(model, label) == style


def test_references_are_labelled_by_their_names():
    items = evaluation.references(IMAGES)
    assert len(items) == 17
    assert {item.label for item in items} == {"anime", "cg", "live_action"}
    assert all(item.path.name.startswith(f"{item.label}-") and item.kind == "image" for item in items)


def test_csv_rows_are_read_relative_to_the_csv(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "list.csv").write_text("path,label\nframes,cg\n\n# a comment\n", encoding="utf-8")
    (item,) = evaluation.read_csv(tmp_path / "list.csv")
    assert (item.path, item.label, item.kind) == (tmp_path / "frames", "cg", "folder")


def test_the_rules_compared_are_the_shipped_one_the_previous_one_and_3d_first_over_either_version():
    assert list(evaluation.RULES) == ["shipped", "previous", "3d_first"]
    assert evaluation.RULES["shipped"] is evaluation.STYLE.rule
    assert [model.name for model in evaluation.VERSIONS.values()] == ["mobilenetv3_v1.5_dist", "mobilenetv3_v1.3_dist"]


def test_each_rule_on_one_frame():
    scores = {
        "anime_real": np.array([[0.2, 0.8]]),
        "anime_classification": np.array([[0.9, 0.05, 0.0, 0.05, 0.0]]),
    }
    # cg = 0.9; the 0.1 left split 0.2 anime / 0.8 real.
    assert evaluation.three_d_first(scores)[0].tolist() == pytest.approx([0.02, 0.9, 0.08])
    # shipped: real^4 = 0.4096 live action; the rest 90% cg (3d's share of the painted 1.0).
    assert evaluation.RULES["shipped"](scores)[0].tolist() == pytest.approx([0.05904, 0.53136, 0.4096])
    # previous: real .8 is live action; the .2 left 90% cg (3d's share of the painted 1.0).
    assert evaluation.previous(scores)[0].tolist() == pytest.approx([0.02, 0.18, 0.8])
