"""Allowed roots (ADR-0030 in fyflow-app): both Node Types read only what their Task's reach, `settings["roots"]`,
lets them, and a refused path fails the Task with what the Host's check says."""

import os
import threading

import pytest

from image_tags.worker import ClassifyFrames, ClassifyImage
from tests.conftest import IMAGES, Context, Task, run

FIXTURES_ROOT = [{"path": str(IMAGES.parent), "access": "read"}]
ANIME = IMAGES / "anime-02-train-window.jpg"
SHEET = IMAGES.parent / "sheets" / "sample.jpg"
WORKERS = [ClassifyImage, ClassifyFrames]


def link(link_path, target):
    """A symbolic link at `link_path` to `target`; the test is skipped where links cannot be made (Windows without the
    privilege)."""
    try:
        os.symlink(target, link_path, target_is_directory=os.path.isdir(target))
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"cannot make a symbolic link here: {error}")


@pytest.mark.parametrize("worker_class", WORKERS)
@pytest.mark.parametrize("path", [ANIME, SHEET, IMAGES])
def test_an_empty_reach_refuses_every_path_saying_to_add_a_root(config, worker_class, path):
    with pytest.raises(PermissionError, match=r"no Allowed roots for it yet\. Add one with read access on the Profiles"):
        run(worker_class(config), Task({"path": str(path)}, roots=[]))


@pytest.mark.parametrize("worker_class", WORKERS)
@pytest.mark.parametrize("path", [ANIME, SHEET, IMAGES])
def test_a_path_outside_the_reach_is_refused(config, worker_class, path, tmp_path):
    roots = [{"path": str(tmp_path), "access": "read"}]

    with pytest.raises(PermissionError, match=r"is outside every Allowed root with read access"):
        run(worker_class(config), Task({"path": str(path)}, roots=roots))


@pytest.mark.parametrize("worker_class", WORKERS)
def test_a_relative_path_is_refused(config, worker_class):
    with pytest.raises(PermissionError, match=r"is not an absolute path"):
        run(worker_class(config), Task({"path": "anime.jpg"}, roots=FIXTURES_ROOT))


@pytest.mark.parametrize("worker_class", WORKERS)
def test_a_path_inside_the_reach_is_classified_and_named_as_given(hub, config, worker_class):
    result = run(worker_class(config), Task({"path": str(ANIME)}, {"classifier": "anime_real"}, roots=FIXTURES_ROOT))

    assert result["path"] == str(ANIME)
    assert result["tag"] == "anime"


def test_classify_image_checks_its_path_off_the_event_loop(hub, config):
    class Recording(Context):
        def allowed_path(self, path, access="read"):
            self.checked_on = threading.current_thread().name
            return super().allowed_path(path, access)

    ctx = Recording()

    run(ClassifyImage(config), Task({"path": str(ANIME)}, {"classifier": "anime_real"}), ctx)

    assert ctx.checked_on.startswith("image-tags-task")


@pytest.mark.parametrize("worker_class", WORKERS)
def test_a_link_inside_the_reach_to_an_image_outside_it_is_refused(config, worker_class, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    link(allowed / "anime.jpg", ANIME)
    roots = [{"path": str(allowed), "access": "read"}]

    with pytest.raises(PermissionError, match=r"is outside every Allowed root"):
        run(worker_class(config), Task({"path": str(allowed / "anime.jpg")}, roots=roots))


def test_a_link_in_a_folder_to_an_image_outside_the_reach_is_skipped_with_a_warning(hub, config, tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(ANIME.read_bytes())
    link(folder / "b.jpg", IMAGES / "tiny.jpg")
    roots = [{"path": str(folder), "access": "read"}]

    result = run(ClassifyFrames(config), Task({"path": str(folder)}, {"classifier": "anime_real"}, roots=roots))

    assert len(result["frames"]) == 1
    assert result["tag"] == "anime"
    assert len(result["warnings"]) == 1
    assert "b.jpg" in result["warnings"][0]
    assert "is outside every Allowed root" in result["warnings"][0]


def test_a_folder_whose_every_image_is_refused_fails_saying_why(config, tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    link(folder / "a.jpg", ANIME)
    roots = [{"path": str(folder), "access": "read"}]

    with pytest.raises(PermissionError, match=r"a\.jpg.* is outside every Allowed root"):
        run(ClassifyFrames(config), Task({"path": str(folder)}, roots=roots))


def test_a_link_to_a_folder_inside_the_reach_is_read_where_it_leads(hub, config, tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(ANIME.read_bytes())
    link(tmp_path / "shortcut", folder)

    result = run(ClassifyFrames(config), Task({"path": str(tmp_path / "shortcut")}, {"classifier": "anime_real"}))

    assert result["path"] == str(tmp_path / "shortcut")
    assert len(result["frames"]) == 1
    assert "warnings" not in result
