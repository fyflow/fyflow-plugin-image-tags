"""A frames Task's per-frame answers written to a file (`lists: "file"`), and a file written before reused."""

import json
import os
from pathlib import Path

import pytest

from image_tags import results
from image_tags.worker import ClassifyFrames, TagFrames
from tests.conftest import IMAGES, REACH, Context, Task, run
from tests.test_classify_frames import SHEETS, frames_folder

SHEET = SHEETS / "sample.jpg"


def classify(config, input, settings=None, ctx=None, roots=REACH):
    return run(ClassifyFrames(config), Task(input, {"classifier": "anime_real", **(settings or {})}, roots=roots), ctx)


def ran_a_model(ctx):
    return any(message and message.startswith("Classified") for _, message in ctx.progress_reports)


def test_by_default_the_frames_are_in_the_result_and_nothing_is_written(hub, config, data_dir):
    result = classify(config, {"path": str(SHEET)})

    assert len(result["frames"]) == 7
    assert not {"file", "count", "reused"} & set(result)
    assert not (data_dir / results.RESULTS_FOLDER).exists()


def test_a_file_holds_the_whole_result_and_the_task_hands_on_the_rest(hub, config, data_dir):
    whole = classify(config, {"path": str(SHEET)})

    result = classify(config, {"path": str(SHEET)}, {"lists": "file"})

    file = Path(result["file"])
    assert file.parent == data_dir / results.RESULTS_FOLDER
    assert file.name.startswith("sample.") and file.suffix == ".json"  # named after the sheet's video, sample.mp4
    # The same frames on the same machine score the same: the file is the result `lists: "result"` hands on.
    assert json.loads(file.read_text(encoding="utf-8")) == whole
    assert "frames" not in result
    assert result["count"] == 7
    assert result["reused"] is False
    assert {key: value for key, value in result.items() if key not in ("file", "count", "reused")} == {
        key: value for key, value in whole.items() if key != "frames"
    }
    assert not list(file.parent.glob("*.part"))


def test_the_same_frames_and_settings_reuse_the_file_without_running_a_model(hub, config):
    first = classify(config, {"path": str(SHEET)}, {"lists": "file"})
    ctx = Context()

    again = classify(config, {"path": str(SHEET)}, {"lists": "file"}, ctx)

    assert again == {**first, "reused": True}
    assert not ran_a_model(ctx)


def test_overwrite_runs_the_models_again(hub, config):
    first = classify(config, {"path": str(SHEET)}, {"lists": "file"})
    ctx = Context()

    again = classify(config, {"path": str(SHEET)}, {"lists": "file", "overwrite": True}, ctx)

    assert again["file"] == first["file"]
    assert again["reused"] is False
    assert ran_a_model(ctx)


def test_other_settings_make_another_file(hub, config):
    first = classify(config, {"path": str(SHEET)}, {"lists": "file"})

    other = classify(config, {"path": str(SHEET)}, {"lists": "file", "minConfidence": 0.9})

    assert other["file"] != first["file"]
    assert other["reused"] is False


def test_a_changed_frame_makes_another_file(hub, config, tmp_path):
    folder = frames_folder(tmp_path, 2)
    first = classify(config, {"path": str(folder)}, {"lists": "file"})
    frame = folder / "001.jpg"
    os.utime(frame, ns=(frame.stat().st_atime_ns, frame.stat().st_mtime_ns + 1_000_000_000))

    again = classify(config, {"path": str(folder)}, {"lists": "file"})

    assert again["file"] != first["file"]
    assert Path(first["file"]).name.startswith("frames.")  # named after the folder


def test_a_list_s_file_is_named_after_its_first_frame_s_folder_and_keyed_by_its_timestamps(hub, config, tmp_path):
    folder = frames_folder(tmp_path, 2)
    frames = [{"path": str(folder / "000.jpg"), "timestamp": 1}, {"path": str(folder / "001.jpg")}]

    first = classify(config, {"frames": frames}, {"lists": "file"})
    retimed = classify(config, {"frames": [{**frames[0], "timestamp": 2}, frames[1]]}, {"lists": "file"})

    assert Path(first["file"]).name.startswith("frames.")
    assert "path" not in first
    assert retimed["file"] != first["file"]


def test_a_json_output_is_written_there_every_time(hub, config, tmp_path):
    out = tmp_path / "out"
    roots = [*REACH, {"path": str(out), "access": "write"}]
    file = out / "answers.json"

    first = classify(config, {"path": str(SHEET)}, {"lists": "file", "output": str(file)}, roots=roots)
    ctx = Context()
    again = classify(config, {"path": str(SHEET)}, {"lists": "file", "output": str(file)}, ctx, roots=roots)

    assert first["file"] == again["file"] == os.path.realpath(file)
    assert again["reused"] is False
    assert ran_a_model(ctx)
    assert len(json.loads(file.read_text(encoding="utf-8"))["frames"]) == 7


def test_an_output_folder_and_the_profile_s_output_dir_are_where_the_file_goes(hub, config, tmp_path):
    out, profile_dir = tmp_path / "out", tmp_path / "profile-out"
    roots = [*REACH, {"path": str(tmp_path), "access": "write"}]

    given = classify(config, {"path": str(SHEET)}, {"lists": "file", "output": str(out)}, roots=roots)
    profile = classify(
        config, {"path": str(SHEET)}, {"lists": "file", "profile": {"outputDir": str(profile_dir)}}, roots=roots
    )

    assert Path(given["file"]).parent == Path(os.path.realpath(out))
    assert Path(profile["file"]).parent == Path(os.path.realpath(profile_dir))


def test_an_output_without_write_access_is_refused_before_a_model_runs(hub, config, tmp_path):
    ctx = Context()

    with pytest.raises(PermissionError, match=r"outside every Allowed root with write access"):
        classify(config, {"path": str(SHEET)}, {"lists": "file", "output": str(tmp_path / "out")}, ctx)

    assert not ran_a_model(ctx)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"lists": "both"}, r'^`lists` is not "result" or "file"'),
        ({"lists": "file", "output": ""}, r"^`output` is not a path"),
        ({"lists": "file", "overwrite": "yes"}, r"^`overwrite` is not true or false"),
    ],
)
def test_bad_file_settings_fail_the_task(config, settings, message):
    with pytest.raises(Exception, match=message):
        classify(config, {"path": str(SHEET)}, settings)


def test_a_result_too_large_to_hand_on_fails_saying_to_write_a_file(hub, config, monkeypatch):
    monkeypatch.setattr(results, "MAX_RESULT_BYTES", 100)

    with pytest.raises(Exception, match=r'more than a Task may hand on: set `lists` to "file"'):
        classify(config, {"path": str(SHEET)})

    assert classify(config, {"path": str(SHEET)}, {"lists": "file"})["count"] == 7


def test_tag_frames_writes_and_reuses_its_file_too(tagger_hub, config, data_dir):
    task = Task({"path": str(IMAGES / "tiny.jpg")}, {"tagger": "stand-in", "lists": "file"})

    first = run(TagFrames(config), task)
    again = run(TagFrames(config), task)

    assert first["tagger"] == "stand-in" and first["count"] == 1 and first["reused"] is False
    assert again == {**first, "reused": True}
    assert len(json.loads(Path(first["file"]).read_text(encoding="utf-8"))["frames"]) == 1
