"""`ClassifyImage` under a Flow's `map`: the Tasks running at once in its process share each model's forward passes.

How the models were run is seen by counting the batches `Models.scores` is given. Tasks are lined up behind a forward
pass held busy, so that their frames pile up for the next batch whatever the timing of their threads.
"""

import threading
import time

import pytest

from image_tags.worker import ClassifyImage
from tests.conftest import IMAGES, Running, Task, run, wait_until

#: The reference frames, each the input of one Task, cycled through when there are more Tasks.
FRAMES = sorted(path for path in IMAGES.glob("*-*.jpg"))


def task(image, classifier="style"):
    return Task({"path": str(image)}, {"classifier": classifier})


class ForwardPasses:
    """The batches `worker`'s models are given (`batches`: `(model id, frames)`, in the order they run). Each waits,
    as a slow forward pass would, until `let_go` lets it through: the next `n`, or all from then on."""

    def __init__(self, worker, held=True):
        self.batches = []
        self._let_through = 0
        self._all = not held
        self._turn = threading.Condition()
        scores = worker.models.scores

        def counted(model, batch, ctx=None):
            with self._turn:
                index = len(self.batches)
                self.batches.append((model.id, len(batch)))
                assert self._turn.wait_for(lambda: self._all or index < self._let_through, 30), "never let go"
            return scores(model, batch, ctx)

        worker.models.scores = counted

    def let_go(self, n=None):
        with self._turn:
            if n is None:
                self._all = True
            else:
                self._let_through += n
            self._turn.notify_all()

    def sizes(self, model):
        return [n for model_id, n in self.batches if model_id == model]


def lining_up():
    """Time for Tasks just started to read their images and hand them to the models' batches, which nothing shows
    from outside."""
    time.sleep(1)


def busy_worker(config, classifier):
    """A Worker with `classifier`'s models loaded, and a Task of that classifier in a forward pass held busy (and its
    other model's frame waiting behind it): frames arriving now wait for the next batch."""
    worker = ClassifyImage(config)
    run(worker, task(FRAMES[0], classifier))
    passes = ForwardPasses(worker)
    busy = Running(worker, task(FRAMES[0], classifier))
    wait_until(lambda: passes.batches, what="a first forward pass")
    lining_up()
    return worker, passes, busy


def alone(config, tasks):
    """Each Task's result run on its own, one after another, so each is a batch of one."""
    worker = ClassifyImage(config)
    return [run(worker, one) for one in tasks]


def test_tasks_at_once_are_scored_in_batches_of_up_to_the_profile_s_batch_size_per_model(hub, config):
    config["batchSize"] = 4
    worker, passes, busy = busy_worker(config, "style")
    tasks = [task(FRAMES[i]) for i in range(1, 7)]
    running = [Running(worker, one) for one in tasks]
    lining_up()

    passes.let_go()
    results = [one.result() for one in running]
    busy.result()

    # `style` runs both models: each model's frames go in batches of their own.
    assert passes.sizes("anime_real") == [1, 4, 2]
    assert passes.sizes("anime_classification") == [1, 4, 2]
    # Each Task gets its own image's answer, as it would alone.
    expected = alone(config, tasks)
    assert [r["path"] for r in results] == [str(FRAMES[i]) for i in range(1, 7)]
    for got, want in zip(results, expected, strict=True):
        assert got["tag"] == want["tag"]
        assert got["scores"] == pytest.approx(want["scores"], abs=1e-5)


def test_a_partial_batch_runs_once_its_window_closes(hub, config):
    worker = ClassifyImage(config)
    run(worker, task(FRAMES[0], "anime_real"))  # the model loaded
    passes = ForwardPasses(worker, held=False)

    started = time.monotonic()
    run(worker, task(FRAMES[1], "anime_real"))

    # A lone frame does not wait for a full batch (16 frames by default): its batch runs when the 20 ms window closes.
    assert passes.batches == [("anime_real", 1)]
    assert time.monotonic() - started < 1


def test_tasks_with_different_classifiers_share_the_batches_of_the_models_they_have_in_common(hub, config):
    worker, passes, busy = busy_worker(config, "style")
    tasks = [task(FRAMES[1], "style"), task(FRAMES[2], "anime_real"), task(FRAMES[3], "anime_classification")]
    running = [Running(worker, one) for one in tasks]
    lining_up()

    passes.let_go()
    results = [one.result() for one in running]
    busy.result()

    assert passes.sizes("anime_real") == [1, 2]
    assert passes.sizes("anime_classification") == [1, 2]
    expected = alone(config, tasks)
    for got, want in zip(results, expected, strict=True):
        assert got["classifier"] == want["classifier"]
        assert got["scores"] == pytest.approx(want["scores"], abs=1e-5)


def test_a_task_cancelled_in_a_running_batch_leaves_the_others_their_results(hub, config):
    worker, passes, busy = busy_worker(config, "anime_real")
    tasks = [task(FRAMES[i], "anime_real") for i in (1, 2, 3)]
    first, cancelled, last = (Running(worker, one) for one in tasks)
    lining_up()
    passes.let_go(1)
    wait_until(lambda: len(passes.batches) == 2, what="the second batch running")

    cancelled.cancel()
    with pytest.raises(BaseException):  # noqa: B017, PT011  (whatever it raises, the shim reports "Cancelled")
        cancelled.result(timeout=1)
    passes.let_go()

    assert passes.batches == [("anime_real", 1), ("anime_real", 3)]
    expected = alone(config, [tasks[0], tasks[2]])
    assert [first.result()["scores"], last.result()["scores"]] == [
        pytest.approx(want["scores"], abs=1e-5) for want in expected
    ]
    busy.result()


def test_a_task_cancelled_while_its_frame_waits_for_a_batch_is_left_out_of_it(hub, config):
    worker, passes, busy = busy_worker(config, "anime_real")
    waiting = [Running(worker, task(FRAMES[i], "anime_real")) for i in (1, 2, 3)]
    lining_up()

    waiting[1].cancel()
    with pytest.raises(BaseException):  # noqa: B017, PT011
        waiting[1].result(timeout=1)
    passes.let_go()

    assert busy.result()["path"] == str(FRAMES[0])
    assert [one.result()["path"] for one in (waiting[0], waiting[2])] == [str(FRAMES[1]), str(FRAMES[3])]
    assert passes.batches == [("anime_real", 1), ("anime_real", 2)]


def test_an_image_that_cannot_be_read_fails_only_its_own_task(hub, config, tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_text("not an image")
    worker, passes, busy = busy_worker(config, "anime_real")
    good = [Running(worker, task(FRAMES[i], "anime_real")) for i in (1, 2)]
    bad = Running(worker, task(broken, "anime_real"))

    with pytest.raises(Exception, match=r"^Not an image .*broken\.jpg"):
        bad.result()
    lining_up()
    passes.let_go()

    assert [one.result()["path"] for one in good] == [str(FRAMES[1]), str(FRAMES[2])]
    assert passes.batches == [("anime_real", 1), ("anime_real", 2)]
    busy.result()
