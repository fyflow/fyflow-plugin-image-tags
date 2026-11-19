"""Grouping a video's frames into scenes and pooling their tags (`scenes.py`), on planted videos: arrays made up with
a seeded random generator, no model and no image."""

import numpy as np
import pytest

from image_tags import scenes
from image_tags.scenes import Scores, Settings, TaggedFrames, frame_spans, group, pool

WIDTH = 64
RATINGS = ("general", "sensitive", "questionable", "explicit")
THRESHOLDS = {"general": 0.35, "character": 0.85}
LOCATIONS = [f"place_{i}" for i in range(18)]
CHARACTERS = [f"person_{i}" for i in range(4)]
SHOT = ["close-up", "looking_at_viewer", "1girl", "from_side"]
NAMES = (*LOCATIONS, *CHARACTERS, *SHOT, *RATINGS)
CATEGORIES = (*["general"] * len(LOCATIONS), *["character"] * len(CHARACTERS), *["general"] * len(SHOT), *[None] * 4)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def planted(
    parts=((40, "AB"), (30, "A")),
    noise=0.3,
    seed=1,
    every=2.0,
    character=False,
    same_tags=False,
    same_look=False,
    shot_tags=False,
    duration=None,
):
    """A made-up video and the first frame of each part. A part is `(seconds, cameras)` or `(seconds, cameras,
    name)`: its frames cycle through the cameras two at a time (`"AB"` is A/B cutting), parts of the same name are
    one planted scene, and `"_"` makes blank frames. Each scene has a random unit centre, each camera one near it, and
    each frame its camera's plus noise; each scene has three place tags around 0.8, and low scores below the
    threshold on others. `character` puts person_0 in every scene; `same_tags` gives every scene the same places;
    `same_look` gives every scene the same centre; `shot_tags` gives each camera its own shot-specific tag.

    The link threshold is a percentile of the video's pairs, so a video of only two long scenes may well be cut inside
    one of them: the cases plant three or more where they count boundaries."""
    rng = np.random.default_rng(seed)
    looks, places, cams = {}, {}, {}
    emb, rows, blank, starts, times = [], [], [], [], []
    t = every / 2
    for k, part in enumerate(parts):
        seconds, cameras, name = (*part, k) if len(part) == 2 else part
        count = round(seconds / every)
        starts.append(len(blank))
        if name not in looks:
            looks[name] = looks[0] if same_look and 0 in looks else unit(rng.normal(size=WIDTH))
            places[name] = [0, 1, 2] if same_tags else rng.choice(len(LOCATIONS), 3, replace=False)
        for i in range(count):
            times.append(t)
            t += every
            if cameras == "_":
                emb.append(np.zeros(WIDTH))
                rows.append(np.zeros(len(NAMES)))
                blank.append(True)
                continue
            camera = cameras[(i // 2) % len(cameras)]
            if (name, camera) not in cams:
                cams[name, camera] = unit(looks[name] + 0.5 * rng.normal(size=WIDTH) / np.sqrt(WIDTH))
            emb.append(unit(cams[name, camera] + noise * rng.normal(size=WIDTH) / np.sqrt(WIDTH)))
            row = np.where(rng.random(len(NAMES)) < 0.3, rng.uniform(0.0, 0.25, len(NAMES)), 0.0)
            row[list(places[name])] = np.clip(0.8 + 0.05 * rng.normal(size=3), 0, 1)
            row[len(LOCATIONS) :] = 0
            if character:
                row[NAMES.index("person_0")] = 0.95
            if shot_tags:
                row[NAMES.index(SHOT["ABCD".index(camera)])] = 0.95
            rating = np.clip([0.9, 0.08, 0.01, 0.01] + 0.01 * rng.normal(size=4), 0, 1)
            row[-4:] = rating
            rows.append(row)
            blank.append(False)
    return make(times, emb, rows, blank, duration), starts


def make(times, emb, rows, blank=None, duration=None, paths=None):
    rows = np.asarray(rows, dtype=np.float32)
    return TaggedFrames(
        timestamps=np.asarray(times, dtype=np.float64),
        embeddings=np.asarray(emb, dtype=np.float32),
        scores=Scores.from_dense(rows[:, :-4]),
        ratings=rows[:, -4:],
        blank=np.zeros(len(times), bool) if blank is None else np.asarray(blank),
        paths=paths or [f"frame_{i:04}.jpg" for i in range(len(times))],
        names=NAMES[:-4],
        categories=CATEGORIES[:-4],
        rating_labels=RATINGS,
        thresholds=THRESHOLDS,
        duration=duration,
    )


def firsts(grouping):
    """Each scene's first frame (input place)."""
    return [int(members[0]) for members in grouping.scenes]


def close_to(found, planted, within=1):
    return len(found) == len(planted) and all(abs(a - b) <= within for a, b in zip(found, planted, strict=True))


def test_three_scenes_with_a_b_cutting_are_found_within_one_frame():
    frames, starts = planted([(40, "AB"), (36, "AB"), (30, "A")])

    grouping = group(frames)

    assert close_to(firsts(grouping), starts)
    assert grouping.threshold is not None
    assert all(link is None or link < grouping.threshold for link in grouping.links[1:])


def test_scenes_with_one_character_and_the_same_tags_are_kept_apart_by_the_embedding():
    frames, starts = planted([(40, "AB"), (40, "AB")], character=True, same_tags=True)

    assert close_to(firsts(group(frames)), starts)


def test_shot_specific_tags_do_not_split_a_b_cutting():
    frames, starts = planted([(40, "AB"), (40, "AB")], shot_tags=True, same_look=True)

    assert close_to(firsts(group(frames, Settings(embedding_weight=0))), starts)


def test_blank_frames_inside_a_scene_are_bridged():
    frames, starts = planted([(20, "AB", "x"), (6, "_"), (20, "AB", "x"), (40, "A", "y"), (36, "AB", "z")])

    grouping = group(frames)

    assert close_to(firsts(grouping), [starts[0], starts[3], starts[4]])
    assert set(range(starts[1], starts[2])) <= set(grouping.scenes[0].tolist())


def test_blank_frames_between_scenes_end_the_first():
    frames, starts = planted([(40, "AB"), (6, "_"), (40, "A"), (36, "AB")])

    grouping = group(frames)

    assert firsts(grouping) == [starts[0], starts[2], starts[3]]
    assert grouping.scene_of[starts[1] : starts[2]].tolist() == [0] * 3


def test_a_long_blank_run_is_a_scene_of_its_own():
    frames, starts = planted([(40, "AB"), (20, "_"), (40, "A"), (36, "AB")])

    grouping = group(frames)

    assert firsts(grouping) == starts
    result = pool(frames, grouping)
    blank_scene = result["scenes"][1]
    assert blank_scene["tags"] == [] and blank_scene["highlights"] == []
    assert blank_scene["keyframe"]["frame"] == starts[1] + 5  # the middle frame
    assert blank_scene["rating"]["label"] == "general"


def test_frame_spans_reach_halfway_to_each_neighbour():
    start, end = frame_spans(np.array([0.0, 1, 5, 6]))

    assert start.tolist() == [0, 0.5, 3, 5.5]
    assert end.tolist() == [0.5, 3, 5.5, 6.5]  # the median gap is 1


def test_frames_at_the_same_time_share_its_span():
    start, end = frame_spans(np.array([0.0, 2, 2, 4]))

    assert start.tolist() == [0, 1, 2, 3]
    assert end.tolist() == [1, 2, 3, 5]


def test_the_last_span_ends_at_the_duration_when_known():
    assert frame_spans(np.array([1.0, 3]), duration=10)[1].tolist() == [2, 10]
    assert frame_spans(np.array([1.0, 3]))[1].tolist() == [2, 4]
    assert frame_spans(np.array([7.0]))[1].tolist() == [8]


def test_scene_times_follow_the_spans_and_the_duration():
    frames, _ = planted([(40, "AB"), (30, "A")], duration=75)

    result = pool(frames, group(frames))

    assert result["scenes"][0]["start"] == 0
    assert result["scenes"][-1]["end"] == 75
    for before, after in zip(result["scenes"], result["scenes"][1:]):
        assert before["end"] == after["start"]
    assert sum(scene["duration"] for scene in result["scenes"]) == pytest.approx(75)


def shuffled(frames, seed=7):
    """The same frames in another input order, and where each went."""
    order = np.random.default_rng(seed).permutation(len(frames))
    dense = np.zeros((len(frames), len(frames.names)), np.float32)
    row, tag, value = frames.scores.entries(np.arange(len(frames)))
    dense[row, tag] = value
    rows = np.concatenate([dense, frames.ratings], axis=1)[order]
    paths = [frames.paths[i] for i in order]
    moved = make(frames.timestamps[order], frames.embeddings[order], rows, frames.blank[order], frames.duration, paths)
    return moved, np.argsort(order)


def test_out_of_order_frames_are_grouped_in_time_order_and_named_by_their_input_place():
    frames, _ = planted([(40, "AB"), (6, "_"), (36, "AB"), (30, "A")])
    moved, place = shuffled(frames)

    grouping, again = group(frames), group(moved)

    assert [place[members].tolist() for members in grouping.scenes] == [m.tolist() for m in again.scenes]
    assert again.scene_of[place].tolist() == grouping.scene_of.tolist()
    result = pool(moved, again)
    for scene in result["scenes"]:
        times = moved.timestamps[scene["frames"]]
        assert (np.diff(times) > 0).all()
        key = scene["keyframe"]
        assert moved.paths[key["frame"]] == key["path"]
        assert moved.timestamps[key["frame"]] == key["timestamp"]


def test_a_short_scene_merges_into_the_more_similar_neighbour():
    frames, starts = planted([(40, "AB", "x"), (8, "A", "y"), (40, "AB", "z")])
    short = slice(starts[1], starts[2])
    rng = np.random.default_rng(3)
    for like, other in (("z", 2), ("x", 0)):
        look = unit(frames.embeddings[starts[other] : starts[other] + 20].mean(axis=0))
        noise = rng.normal(size=(starts[2] - starts[1], WIDTH)) / np.sqrt(WIDTH)
        frames.embeddings[short] = unit(0.6 * look + 0.8 * unit(rng.normal(size=WIDTH)) + 0.1 * noise)

        unmerged = firsts(group(frames, Settings(min_scene_duration=0)))
        assert {starts[1], starts[2]} <= set(unmerged), "the short scene is cut on both sides first"
        expected = [starts[0], starts[1]] if like == "z" else [starts[0], starts[2]]
        assert firsts(group(frames)) == expected


def test_pairs_a_hair_below_the_threshold_still_link(monkeypatch):
    # Identical frames: their pairs' similarity ties, but the arithmetic leaves some a bit below the rest. Those
    # crossing the middle are put a bit below here, which would cut there if a tie had to be met exactly.
    frames, _ = planted([(60, "A")], noise=0)
    middle = len(frames) // 2
    real = scenes._similarity

    def tied(*args):
        similarity, s_emb, _ = real(*args)
        first, second = args[3]
        similarity = np.ones(len(first))
        similarity[(first < middle) & (second >= middle)] = 1.0 - 2e-4
        return similarity, s_emb, 1.0

    monkeypatch.setattr(scenes, "_similarity", tied)

    assert firsts(group(frames, Settings(min_scene_duration=0))) == [0]


def test_one_frame_is_one_scene():
    frames = make([3.0], [unit(np.ones(WIDTH))], [[0.9] + [0] * (len(NAMES) - 5) + [0.9, 0.1, 0, 0]])

    result = pool(frames, group(frames))

    assert result["threshold"] is None
    assert result["count"] == 1
    [scene] = result["scenes"]
    assert scene["frames"] == [0] and scene["keyframe"]["frame"] == 0
    assert (scene["start"], scene["end"]) == (0, 4)
    assert [tag["name"] for tag in scene["tags"]] == ["place_0"]


def test_all_blank_frames_are_one_scene():
    frames, _ = planted([(30, "_")])

    grouping = group(frames)
    result = pool(frames, grouping)

    assert result["count"] == 1 and result["threshold"] is None
    assert result["scenes"][0]["keyframe"]["frame"] == 7
    assert result["summary"]["tags"] == []


def test_with_no_tags_left_the_embedding_alone_groups():
    frames, starts = planted([(40, "AB"), (36, "AB"), (30, "A")])
    frames.names = SHOT[:1] * len(frames.names)  # every tag shot-specific

    assert close_to(firsts(group(frames, Settings(embedding_weight=0))), starts)


def test_ignored_tags_are_left_out_like_shot_specific_ones():
    frames, starts = planted([(40, "AB"), (36, "AB"), (30, "A")])

    grouping = group(frames, Settings(embedding_weight=0, ignore_tags=frozenset(LOCATIONS)))

    assert close_to(firsts(grouping), starts)


def test_embedding_weight_0_groups_by_tags_and_1_by_the_embedding():
    by_tags, starts = planted([(40, "AB"), (36, "AB"), (30, "A")], same_look=True, noise=1.0)
    by_look, _ = planted([(40, "AB"), (36, "AB"), (30, "A")], same_tags=True)

    assert close_to(firsts(group(by_tags, Settings(embedding_weight=0))), starts)
    assert close_to(firsts(group(by_look, Settings(embedding_weight=1))), starts)


def one_scene(columns, ratings=None, times=None):
    """Frames every 2 s (spans of 2 s) of one scene, all alike, with the given scores per tag name."""
    count = len(next(iter(columns.values())))
    rows = np.zeros((count, len(NAMES)), np.float32)
    for name, values in columns.items():
        rows[:, NAMES.index(name)] = values
    rows[:, -4:] = ratings if ratings is not None else [0.9, 0.08, 0.01, 0.01]
    times = times if times is not None else 1 + 2 * np.arange(count)
    rng = np.random.default_rng(0)
    emb = unit(np.ones(WIDTH) + 0.01 * rng.normal(size=(count, WIDTH)))
    return make(times, emb, rows)


def test_hysteresis_keeps_a_tag_on_by_its_category_s_gap(monkeypatch):
    monkeypatch.setattr(scenes, "SMOOTHING", 0)
    frames = one_scene(
        {
            "place_0": [0.4, 0.27, 0.27, 0.2, 0.2],  # general: on at 0.35, off below 0.25
            # character: on at 0.85, off below 0.70; a general gap would turn it off at 0.72
            "person_0": [0.9, 0.72, 0.72, 0.6, 0.6],
        }
    )

    result = pool(frames, group(frames), Settings(min_coverage=0))

    coverage = {tag["name"]: tag["coverage"] for tag in result["scenes"][0]["tags"]}
    assert coverage == {"place_0": 0.6, "person_0": 0.6}
    assert result["summary"]["tags"][0]["name"] in coverage


def test_smoothing_never_crosses_a_boundary(monkeypatch):
    monkeypatch.setattr(scenes, "SMOOTHING", 1000)
    frames, _ = planted([(40, "AB"), (40, "A")])

    result = pool(frames, group(frames))

    first, second = ({tag["name"]: tag["coverage"] for tag in scene["tags"]} for scene in result["scenes"])
    assert len(first) == len(second) == 3  # each scene's own three places, all through it
    assert set(first.values()) == set(second.values()) == {1}
    assert not set(first) & set(second)


def test_highlights_and_rating_peaks_point_at_their_frame():
    ratings = np.tile([0.9, 0.08, 0.01, 0.01], (12, 1))
    ratings[4] = [0.3, 0.1, 0.6, 0.0]
    ratings[9] = [0.3, 0.1, 0.1, 0.55]
    frames = one_scene(
        {
            "place_0": [0.8] * 12,
            "place_5": [0] * 7 + [0.82] + [0] * 4,  # one strong frame
            "place_6": [0] * 2 + [0.55, 0.6] + [0] * 8,  # two frames over PEAK_REPEAT
            "person_2": [0] * 10 + [0.9, 0.88],  # a character twice over its threshold
            "person_3": [0.95] + [0] * 11,  # a character once: a misfire
            "close-up": [0.99] + [0] * 11,  # shot-specific
        },
        ratings,
    )
    moved, place = shuffled(frames, seed=3)

    result = pool(moved, group(moved))

    [scene] = result["scenes"]
    assert [tag["name"] for tag in scene["tags"]] == ["place_0"]
    highlights = {h["name"]: h for h in scene["highlights"]}
    assert list(highlights) == ["person_2", "place_5", "place_6"]
    assert highlights["place_5"]["frame"] == place[7]
    assert highlights["place_5"]["path"] == moved.paths[place[7]] == "frame_0007.jpg"
    assert highlights["place_5"]["timestamp"] == 15
    assert (highlights["place_6"]["frame"], highlights["place_6"]["frames"]) == (place[3], 2)
    assert highlights["person_2"]["frame"] == place[10]
    assert scene["rating"]["label"] == "general"
    peaks = {peak["label"]: peak for peak in scene["rating"]["peaks"]}
    assert list(peaks) == ["questionable", "explicit"]
    assert (peaks["questionable"]["frame"], peaks["questionable"]["score"]) == (place[4], 0.6)
    assert peaks["explicit"]["path"] == moved.paths[place[9]]


def test_ignored_tags_are_no_highlights():
    frames = one_scene({"place_5": [0] * 7 + [0.82] + [0] * 4})

    result = pool(frames, group(frames), Settings(ignore_tags=frozenset({"place_5"})))

    assert result["scenes"][0]["highlights"] == []


def test_ignored_tags_are_listed_nowhere_but_shot_specific_ones_are_still_listed():
    frames = one_scene({"place_0": [0.8] * 10, "place_1": [0.8] * 10, "close-up": [0.8] * 10})

    result = pool(frames, group(frames), Settings(ignore_tags=frozenset({"place_1"})))

    for listed in (result["scenes"][0]["tags"], result["summary"]["tags"]):
        assert {entry["name"] for entry in listed} == {"place_0", "close-up"}


def test_max_tags_caps_every_list():
    frames = one_scene(
        {
            **{f"place_{i}": [0.8] * 10 for i in range(5)},
            **{f"place_{i}": [0] * i + [0.9] + [0] * (9 - i) for i in range(5, 10)},
        }
    )

    result = pool(frames, group(frames), Settings(max_tags=2))

    assert len(result["scenes"][0]["tags"]) == 2
    assert len(result["scenes"][0]["highlights"]) == 2
    assert len(result["summary"]["tags"]) == 2


def test_without_tags_scenes_have_times_frames_rating_and_keyframe_only():
    frames, _ = planted()

    result = pool(frames, group(frames), Settings(include_tags=False))

    for scene in result["scenes"]:
        assert set(scene) <= {"index", "start", "end", "duration", "frames", "boundary", "keyframe", "rating"}
    assert set(result["summary"]) == {"rating"}


def test_the_summary_s_rating_is_the_most_severe_scene_s():
    frames, starts = planted([(40, "AB"), (40, "A")])
    frames.ratings[starts[1] + 3] = [0.02, 0.02, 0.01, 0.95]

    result = pool(frames, group(frames))

    assert [scene["rating"]["label"] for scene in result["scenes"]] == ["general", "explicit"]
    assert result["summary"]["rating"]["label"] == "explicit"
    assert max(result["summary"]["rating"]["scores"], key=result["summary"]["rating"]["scores"].get) == "general"


def test_the_result_has_the_spec_s_shape():
    frames, _ = planted([(40, "AB"), (36, "AB")], character=True)

    result = pool(frames, group(frames))

    assert set(result) == {"threshold", "summary", "count", "scenes"}
    assert result["count"] == len(result["scenes"]) == 2
    first, second = result["scenes"]
    assert "boundary" not in first and set(second["boundary"]) == {"link"}
    assert set(first["keyframe"]) == {"frame", "timestamp", "path"}
    assert set(first["rating"]) == {"label", "scores", "peaks"} and list(first["rating"]["scores"]) == list(RATINGS)
    assert set(first["tags"][0]) == {"name", "category", "coverage", "score"}
    assert {tag["name"]: tag["category"] for tag in result["summary"]["tags"]}["person_0"] == "character"
    assert sorted(first["frames"] + second["frames"]) == list(range(len(frames)))


def test_sparse_scores_stack_by_batch():
    dense = np.random.default_rng(0).random((7, 5)).astype(np.float32)

    whole = Scores.from_dense(dense, floor=0.5)
    parts = Scores.stack([Scores.from_dense(dense[:3], floor=0.5), Scores.from_dense(dense[3:], floor=0.5)])

    for field in ("rows", "tags", "values"):
        assert getattr(parts, field).tolist() == getattr(whole, field).tolist()
    row, tag, _ = whole.entries(np.array([4, 1]))
    assert sorted(zip(row.tolist(), tag.tolist())) == sorted(
        [(0, int(t)) for t in np.flatnonzero(dense[4] >= 0.5)] + [(1, int(t)) for t in np.flatnonzero(dense[1] >= 0.5)]
    )
