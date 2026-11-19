"""TagScenes' grouping and pooling: a video's tagged frames in, its scenes and their tags out. Pure numpy, no model
and no file, so the tests build the arrays directly.

`group` links frames that look alike (their image embeddings and their tags), within a window of time, and cuts the
video where no link crosses; `pool` then gives each scene its tags, highlights, rating and keyframe, and the video a
summary. Frames are taken in time order (a stable sort by timestamp), while every frame index handed on is the
frame's place in the input.

Every number not given as a setting is a constant here, so a review can vary it without one.
"""

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

#: Scores below this count as 0, and are all the worker keeps of a frame's tags.
SCORE_FLOOR = 0.05
#: Two frames further apart than this, in seconds, are never compared.
LINK_WINDOW = 30.0
#: Nor are two frames with more than this many non-blank frames between them, counting the later one.
MAX_LINK_LAG = 15
#: The share of compared pairs, from the top, that link: the link threshold is this percentile of their similarity.
LINK_PERCENTILE = 85
#: How far below the link threshold, in z-units, a pair still links. Identical frames tie, but their pairs come out of
#: the arithmetic a little apart (rounding; on Linux, onnxruntime's CPU kernels embed a frame some 2e-4 z-units
#: differently by its place in a batch), and when the threshold falls on such a tie, that must not cut between them.
_LINK_TOLERANCE = 1e-3
#: A frame counts towards a tag's idf when it scores this or more on it.
DF_SCORE = 0.2
#: Added to every tag's idf, so a tag in every frame still counts a little.
IDF_FLOOR = 0.1
#: How much more a character tag counts than a general one when comparing frames.
CHARACTER_WEIGHT = 1.5
#: The Gaussian's σ, in seconds, that smooths a frame's scores with its scene's other frames.
SMOOTHING = 4.0
#: Frames further apart than this many σ are left out of each other's smoothing.
SMOOTHING_REACH = 4.0
#: How far below its category's threshold a tag that is on stays on.
OFF_GAP = {"general": 0.10, "character": 0.15}
#: A general tag is a highlight when it reaches this in one frame…
PEAK_STRONG = 0.7
#: …or this in two. A rating reaching this in two frames is the scene's label.
PEAK_REPEAT = 0.5
#: A rating reaching this in one frame is the scene's label.
RATING_STRONG = 0.9
#: The categories whose tags are compared, pooled and listed.
CATEGORIES = ("general", "character")

#: Tags that change with every cut inside one scene (framing, camera angle, gaze, expression, head counts): left out of
#: comparing frames and of highlights. Danbooru's names, which both taggers use.
SHOT_SPECIFIC_TAGS = frozenset(
    {
        "close-up", "solo", "solo_focus", "male_focus", "upper_body", "portrait", "cowboy_shot", "full_body",
        "from_above", "from_below", "from_side", "from_behind", "straight-on", "pov", "profile",
        "looking_at_viewer", "looking_away", "looking_to_the_side", "looking_up", "looking_down",
        "head_out_of_frame", "out_of_frame", "face", "eye_focus", "blurry", "depth_of_field",
        "open_mouth", "closed_mouth", "closed_eyes", "parted_lips", "teeth", "clenched_teeth", "sweat", "hands_up",
        "1boy", "1girl", "multiple_boys", "multiple_girls", "2boys", "2girls", "3boys", "3girls", "multiple_others",
    }
)  # fmt: skip

#: Rows of embeddings handled at once, to keep copies small at 10000 frames.
_CHUNK_ROWS = 4096
#: Tags pooled at once in a scene, to keep its dense scores small.
_CHUNK_TAGS = 512


@dataclass(frozen=True)
class Scores:
    """Each frame's tag scores, sparse: frame `f`'s are `values[rows[f]:rows[f + 1]]`, for the tags (places in the
    tag list) `tags[rows[f]:rows[f + 1]]`, each tag at most once per frame. Only scores of `SCORE_FLOOR` or more need
    be kept; lower ones count as 0 anyway."""

    rows: np.ndarray
    tags: np.ndarray
    values: np.ndarray

    @classmethod
    def from_dense(cls, scores: np.ndarray, floor: float = SCORE_FLOOR) -> "Scores":
        """The scores of `scores` (`[frames, tags]`, a batch's output) of `floor` or more."""
        frame, tag = np.nonzero(scores >= floor)
        rows = np.zeros(len(scores) + 1, dtype=np.int64)
        np.cumsum(np.bincount(frame, minlength=len(scores)), out=rows[1:])
        return cls(rows, tag.astype(np.int32), scores[frame, tag].astype(np.float32))

    @classmethod
    def stack(cls, parts: Sequence["Scores"]) -> "Scores":
        """The frames of each of `parts`, in turn: the batches of a video put together."""
        rows, offset = [np.zeros(1, dtype=np.int64)], 0
        for part in parts:
            rows.append(part.rows[1:] + offset)
            offset += int(part.rows[-1])
        return cls(
            np.concatenate(rows),
            np.concatenate([np.zeros(0, np.int32), *(part.tags for part in parts)]).astype(np.int32),
            np.concatenate([np.zeros(0, np.float32), *(part.values for part in parts)]).astype(np.float32),
        )

    def entries(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The scores of `frames` as `(row, tag, value)`, `row` being the place in `frames`."""
        starts, ends = self.rows[frames], self.rows[frames + 1]
        counts = ends - starts
        row = np.repeat(np.arange(len(frames)), counts)
        at = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts) + np.repeat(starts, counts)
        return row, self.tags[at], self.values[at]


@dataclass
class TaggedFrames:
    """A video's frames as tagged and embedded, in input order, with what grouping needs to know of the tagger.

    Per frame: `timestamps` (seconds), `embeddings` (`[n, d]`, any length; a blank frame's is not read), `scores`
    (sparse, places in the tag list), `ratings` (`[n, len(rating_labels)]`), `blank` and `paths` (the frame's image,
    or None for a sheet's cell). Of the tag list: each tag's `names` and `categories` (a category's name, as
    `thresholds` keys it; tags of any category but general and character are not read), the `rating_labels` from
    least to most severe, and each category's `thresholds`. `duration` is the video's, in seconds, when known.
    """

    timestamps: np.ndarray
    embeddings: np.ndarray
    scores: Scores
    ratings: np.ndarray
    blank: np.ndarray
    paths: Sequence[str | None]
    names: Sequence[str]
    categories: Sequence[str | None]
    rating_labels: Sequence[str]
    thresholds: Mapping[str, float]
    duration: float | None = None

    def __len__(self) -> int:
        return len(self.timestamps)


@dataclass(frozen=True)
class Settings:
    """A Node's settings that grouping and pooling read, with TagScenes' defaults."""

    embedding_weight: float = 0.6
    min_scene_duration: float = 15.0
    ignore_tags: frozenset[str] = frozenset()
    min_coverage: float = 0.3
    max_tags: int = 40
    include_tags: bool = True


@dataclass
class Grouping:
    """The scenes `group` found. `scenes[k]` is scene k's frames (input places, in time order) and `links[k]` the
    strongest link across the cut before it (None for scene 0, or when no compared pair crosses). `scene_of[f]` is
    input frame f's scene. `threshold` is the link threshold in z-units, None when no pair was compared."""

    scenes: list[np.ndarray]
    links: list[float | None]
    scene_of: np.ndarray
    threshold: float | None
    #: Each input frame's span, `[start, end)` in seconds.
    start: np.ndarray = field(repr=False)
    end: np.ndarray = field(repr=False)
    #: The non-blank frames' mean embedding, and each input frame's distance from it (0 for a blank frame).
    centre: np.ndarray = field(repr=False)
    radius: np.ndarray = field(repr=False)


def frame_spans(timestamps: np.ndarray, duration: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Each frame's span, `(start, end)`, for frames in time order: from halfway to the previous frame's time to
    halfway to the next one's. The first starts at 0; the last ends at `duration` when known (and not before its own
    time), else at its own time plus half the median gap between times, or plus 1 s when there is one time. Frames at
    the same time share its span equally, in turn."""
    times = np.asarray(timestamps, dtype=np.float64)
    if not len(times):
        return np.zeros(0), np.zeros(0)
    unique, first, count = np.unique(times, return_index=True, return_counts=True)
    halfway = (unique[1:] + unique[:-1]) / 2
    if duration is not None and duration >= unique[-1]:
        last = float(duration)
    else:
        last = unique[-1] + (float(np.median(np.diff(unique))) / 2 if len(unique) > 1 else 1.0)
    lo = np.concatenate([[min(0.0, unique[0])], halfway])
    hi = np.concatenate([halfway, [last]])
    which = np.repeat(np.arange(len(unique)), count)
    within = np.arange(len(times)) - first[which]
    share = (hi - lo)[which] / count[which]
    start = lo[which] + within * share
    return start, start + share


def group(frames: TaggedFrames, settings: Settings = Settings()) -> Grouping:
    """The video's scenes: frames linked by similarity within `LINK_WINDOW` and `MAX_LINK_LAG`, cut where no link
    crosses, blank frames bridged, and scenes shorter than `min_scene_duration` merged into their more similar
    neighbour."""
    n = len(frames)
    if not n:
        raise ValueError("There are no frames to group")
    order = np.argsort(np.asarray(frames.timestamps, dtype=np.float64), kind="stable")
    times = np.asarray(frames.timestamps, dtype=np.float64)[order]
    start_t, end_t = frame_spans(times, frames.duration)
    span = end_t - start_t
    blank = np.asarray(frames.blank, dtype=bool)[order]
    seen = np.flatnonzero(~blank)  # time places of the non-blank frames
    rows = order[seen]  # their input places

    emb = np.asarray(frames.embeddings)
    centre = emb[rows].mean(axis=0, dtype=np.float64) if len(rows) else np.zeros(emb.shape[1])
    radius = np.zeros(n)
    radius[rows] = _radius(emb, rows, centre)

    pairs = _candidates(times[seen])
    similarity, s_emb, threshold = _similarity(frames, settings, rows, pairs, centre, radius)

    # Cuts, as time places: a cut k falls before the frame at time place k.
    first, second = pairs
    reach = np.arange(len(seen))
    if threshold is None:
        linked = np.zeros(0, bool)
    else:
        linked = (similarity >= threshold - _LINK_TOLERANCE) & (s_emb >= 0)
    np.maximum.at(reach, first[linked], second[linked])
    open_until = np.maximum.accumulate(reach)
    cut_after = set((np.flatnonzero(open_until[:-1] < np.arange(1, len(seen))) + 1).tolist())
    cuts = _with_blanks(seen, cut_after, n, span, settings.min_scene_duration)

    bounds = _merge(cuts, n, start_t, end_t, order, emb, centre, radius, blank, settings.min_scene_duration)

    scenes = [order[a:b] for a, b in itertools.pairwise(bounds)]
    scene_of = np.zeros(n, dtype=np.int64)
    for k, members in enumerate(scenes):
        scene_of[members] = k
    links = [None, *_links(bounds[1:-1], seen, pairs, similarity)]
    start = np.zeros(n)
    end = np.zeros(n)
    start[order], end[order] = start_t, end_t
    return Grouping(scenes, links, scene_of, threshold, start, end, centre, radius)


def _radius(emb: np.ndarray, rows: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Each of `rows`' distance from `centre`, floored so a frame at the centre divides safely."""
    out = np.zeros(len(rows))
    for at in range(0, len(rows), _CHUNK_ROWS):
        part = emb[rows[at : at + _CHUNK_ROWS]] - centre
        out[at : at + _CHUNK_ROWS] = np.sqrt(np.einsum("ij,ij->i", part, part))
    return np.maximum(out, 1e-9)


def _candidates(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pairs `(i, j)`, `i < j`, of non-blank frames (places among them, in time order) that are compared."""
    first, second = [], []
    for lag in range(1, min(MAX_LINK_LAG, len(times) - 1) + 1):
        i = np.arange(len(times) - lag)
        near = times[i + lag] - times[i] <= LINK_WINDOW
        first.append(i[near])
        second.append(i[near] + lag)
    if not first:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(first), np.concatenate(second)


def _similarity(frames, settings, rows, pairs, centre, radius):
    """Each compared pair's similarity `S` (z-units), its centred embedding cosine `S_emb`, and the link threshold
    (None with no pairs)."""
    first, second = pairs
    if not len(first):
        return np.zeros(0), np.zeros(0), None
    a, b = rows[first], rows[second]
    emb = np.asarray(frames.embeddings)
    s_emb = np.zeros(len(a))
    for at in range(0, len(a), _CHUNK_ROWS):
        x = emb[a[at : at + _CHUNK_ROWS]] - centre
        y = emb[b[at : at + _CHUNK_ROWS]] - centre
        s_emb[at : at + _CHUNK_ROWS] = np.einsum("ij,ij->i", x, y)
    s_emb /= radius[a] * radius[b]

    s_tag, any_tags = _tag_cosines(frames, settings, rows, pairs)
    weight = settings.embedding_weight if any_tags else 1.0
    similarity = weight * _z(s_emb) + (1 - weight) * _z(s_tag)
    return similarity, s_emb, float(np.percentile(similarity, LINK_PERCENTILE))


def _z(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / max(float(values.std()), 1e-6)


def _tag_cosines(frames, settings, rows, pairs):
    """The cosine of each compared pair's tag vectors (0 when either is empty), and whether any frame has a tag left:
    general and character tags only, not shot-specific or ignored, each weighted by its idf over the non-blank
    frames and characters by `CHARACTER_WEIGHT`."""
    first, second = pairs
    row, tag, value = frames.scores.entries(rows)
    kept = _kept(frames, settings, SHOT_SPECIFIC_TAGS | settings.ignore_tags)
    keep = kept[tag] & (value >= SCORE_FLOOR)
    row, tag, value = row[keep], tag[keep], value[keep].astype(np.float64)
    if not len(row):
        return np.zeros(len(first)), False
    used, column = np.unique(tag, return_inverse=True)
    df = np.bincount(column[value >= DF_SCORE], minlength=len(used))
    weight = np.log((len(rows) + 1) / (df + 1)) + IDF_FLOOR
    weight[np.asarray(frames.categories, dtype=object)[used] == "character"] *= CHARACTER_WEIGHT
    value = value * weight[column]
    norm = np.sqrt(np.bincount(row, value * value, minlength=len(rows)))

    # Dot products lag by lag: an entry of frame i meets the same tag's entry of frame i + lag.
    width = len(used)
    key = row.astype(np.int64) * width + column
    dots = np.zeros(len(first))
    lags = second - first
    for lag in np.unique(lags):
        _, ia, ib = np.intersect1d(key + int(lag) * width, key, assume_unique=True, return_indices=True)
        dot = np.bincount(row[ia], value[ia] * value[ib], minlength=len(rows))
        at = lags == lag
        dots[at] = dot[first[at]]
    norms = norm[first] * norm[second]
    return np.divide(dots, norms, out=np.zeros(len(first)), where=norms > 0), True


def _kept(frames: TaggedFrames, settings: Settings, left_out: frozenset[str]) -> np.ndarray:
    """Per tag of the tag list: whether it is general or character and not in `left_out`."""
    return np.array(
        [c in CATEGORIES and name not in left_out for name, c in zip(frames.names, frames.categories, strict=True)],
        dtype=bool,
    )


def _with_blanks(seen: np.ndarray, cut_after: set[int], n: int, span: np.ndarray, min_duration: float) -> list[int]:
    """The cuts (time places) from those between non-blank frames (`cut_after` holds `q` for a cut between the
    `q - 1`th and `q`th of them): blank frames between two linked frames stay in their scene, blank frames at a cut
    end the scene before it, and a run of blank frames longer than `min_duration` is a scene of its own. With no
    non-blank frame, the video is one scene."""
    if not len(seen):
        return []
    cuts = set()
    # Blank runs: before the first non-blank frame, between two, and after the last.
    edges = [(-1, int(seen[0]))] + [(int(a), int(b)) for a, b in itertools.pairwise(seen)]
    edges.append((int(seen[-1]), n))
    for q, (a, b) in enumerate(edges):
        if 0 < q < len(seen) and q in cut_after:
            cuts.add(b)
        if b - a > 1 and span[a + 1 : b].sum() > min_duration:
            cuts.update((a + 1, b))
    return sorted(cut for cut in cuts if 0 < cut < n)


def _merge(cuts, n, start_t, end_t, order, emb, centre, radius, blank, min_duration) -> list[int]:
    """The scene bounds (time places, from 0 to n) once every scene shorter than `min_duration` is merged into the
    neighbour whose frames' embeddings are more alike its own: the mean centred cosine over all their pairs. A
    neighbour with no non-blank frame is the less alike."""
    bounds = [0, *cuts, n]
    sums = []
    for a, b in itertools.pairwise(bounds):
        sums.append(_unit_sum(emb, order[a:b][~blank[a:b]], centre, radius))
    while len(bounds) > 2:
        durations = [end_t[b - 1] - start_t[a] for a, b in itertools.pairwise(bounds)]
        k = int(np.argmin(durations))
        if durations[k] >= min_duration:
            break
        best, alike = None, -math.inf
        for other in (k - 1, k + 1):
            if not 0 <= other < len(sums):
                continue
            (u, count), (v, other_count) = sums[k], sums[other]
            mean = float(u @ v) / (count * other_count) if count and other_count else -math.inf
            if best is None or mean > alike:
                best, alike = other, mean
        low = min(k, best)
        sums[low : low + 2] = [(sums[low][0] + sums[low + 1][0], sums[low][1] + sums[low + 1][1])]
        del bounds[low + 1]
    return bounds


def _unit_sum(emb, rows, centre, radius) -> tuple[np.ndarray, int]:
    """The sum of `rows`' centred, unit-length embeddings, and how many there are."""
    total = np.zeros(len(centre))
    for at in range(0, len(rows), _CHUNK_ROWS):
        part = rows[at : at + _CHUNK_ROWS]
        total += ((emb[part] - centre) / radius[part, None]).sum(axis=0)
    return total, len(rows)


def _links(cuts: Sequence[int], seen: np.ndarray, pairs, similarity) -> list[float | None]:
    """Per cut (time place): the strongest similarity of a compared pair across it, or None."""
    cuts = np.asarray(cuts, dtype=np.int64)
    best = np.full(len(cuts), -np.inf)
    first, second = pairs
    lo = np.searchsorted(cuts, seen[first], side="right")
    hi = np.searchsorted(cuts, seen[second], side="right")
    for step in range(int((hi - lo).max(initial=0))):
        at = lo + step < hi
        np.maximum.at(best, lo[at] + step, similarity[at])
    return [None if math.isinf(value) else round(float(value), 4) for value in best]


def pool(frames: TaggedFrames, grouping: Grouping, settings: Settings = Settings()) -> dict:
    """The result's `threshold`, `summary`, `count` and `scenes`, in the shapes TagScenes hands on.

    Means are weighted by each frame's span. Blank frames count towards a scene's time, not its tags or rating (an
    all-blank scene's rating is its blank frames'). A tag is on in a frame when its score, smoothed with the scene's
    other frames (`SMOOTHING`), turns on at its category's threshold and stays on down to `OFF_GAP` below it.
    """
    blank = np.asarray(frames.blank, dtype=bool)
    span = grouping.end - grouping.start
    pooled = np.flatnonzero(_kept(frames, settings, settings.ignore_tags))
    in_tags = np.zeros(len(frames.names), dtype=bool)
    in_tags[pooled] = True
    highlightable = _kept(frames, settings, SHOT_SPECIFIC_TAGS | settings.ignore_tags)
    on_time = np.zeros(len(frames.names))
    on_score = np.zeros(len(frames.names))
    total_time = 0.0
    labels = []
    scenes = []
    for k, members in enumerate(grouping.scenes):
        seen = members[~blank[members]]
        duration = float(grouping.end[members[-1]] - grouping.start[members[0]])
        total_time += duration
        scene = {
            "index": k,
            "start": _seconds(grouping.start[members[0]]),
            "end": _seconds(grouping.end[members[-1]]),
            "duration": _seconds(duration),
            "frames": members.tolist(),
        }
        if k:
            scene["boundary"] = {"link": grouping.links[k]}
        scene["keyframe"] = _keyframe(frames, grouping, members, seen, span)
        rating = _rating(frames, grouping, seen if len(seen) else members, span)
        if rating is not None:
            scene["rating"] = rating
            labels.append(rating["label"])
        if settings.include_tags:
            stats = _tag_stats(frames, grouping, seen, in_tags)
            on_time[stats.tags] += stats.on_time
            on_score[stats.tags] += stats.on_score
            scene["tags"] = _tag_list(frames, settings, stats.tags, stats.on_time / max(duration, 1e-9), stats)
            listed = {entry["name"] for entry in scene["tags"]}
            scene["highlights"] = _highlights(frames, grouping, settings, seen, stats, highlightable, listed)
        scenes.append(scene)

    summary = {}
    if settings.include_tags:
        used = np.flatnonzero(on_time > 0)
        stats = _Stats(used, on_time[used], on_score[used])
        summary["tags"] = _tag_list(frames, settings, used, on_time[used] / max(total_time, 1e-9), stats)
    if len(frames.rating_labels):
        everyone = np.arange(len(frames))
        seen = everyone[~blank]
        summary["rating"] = {
            "label": max(labels, key=list(frames.rating_labels).index),
            "scores": _mean_ratings(frames, seen if len(seen) else everyone, span),
        }
    threshold = None if grouping.threshold is None else round(grouping.threshold, 4)
    return {"threshold": threshold, "summary": summary, "count": len(scenes), "scenes": scenes}


def _seconds(value) -> float:
    return round(float(value), 3)


def _score(value) -> float:
    return round(float(value), 4)


def _at(frames: TaggedFrames, frame: int) -> dict:
    """A frame as the result points at it: its input place, time and image."""
    out = {"frame": int(frame), "timestamp": _seconds(frames.timestamps[frame])}
    if frames.paths[frame] is not None:
        out["path"] = frames.paths[frame]
    return out


def _keyframe(frames, grouping, members, seen, span) -> dict:
    """The non-blank frame whose centred embedding is most like the scene's mean one, or the middle frame of an
    all-blank scene."""
    if not len(seen):
        return _at(frames, members[len(members) // 2])
    emb = np.asarray(frames.embeddings)
    unit = (emb[seen] - grouping.centre) / grouping.radius[seen, None]
    mean = span[seen] @ unit
    return _at(frames, seen[int(np.argmax(unit @ mean))])


def _mean_ratings(frames, rows, span) -> dict:
    weight = np.maximum(span[rows], 1e-9)
    means = weight @ np.asarray(frames.ratings, dtype=np.float64)[rows] / weight.sum()
    return {label: _score(value) for label, value in zip(frames.rating_labels, means, strict=True)}


def _rating(frames, grouping, rows, span) -> dict | None:
    """The scene's rating: the most severe one reaching `PEAK_REPEAT` in two frames or `RATING_STRONG` in one, else
    the highest mean; the mean scores; and the `peaks` of each more severe rating reaching `PEAK_REPEAT`."""
    if not len(frames.rating_labels):
        return None
    labels = list(frames.rating_labels)
    scores = np.asarray(frames.ratings, dtype=np.float64)[rows]
    means = _mean_ratings(frames, rows, span)
    sure = np.flatnonzero(((scores >= PEAK_REPEAT).sum(axis=0) >= 2) | (scores.max(axis=0) >= RATING_STRONG))
    label = int(sure.max()) if len(sure) else labels.index(max(means, key=means.get))
    peaks = []
    for r in range(label + 1, len(labels)):
        best = int(np.argmax(scores[:, r]))
        if scores[best, r] >= PEAK_REPEAT:
            peaks.append({"label": labels[r], "score": _score(scores[best, r]), **_at(frames, rows[best])})
    return {"label": labels[label], "scores": means, "peaks": peaks}


@dataclass
class _Stats:
    """Per tag seen in a scene (`tags`, places in the tag list): the time it is on, and its score times that time;
    with `rows`, also its raw peak, the scene row of it, and how many frames reach `PEAK_REPEAT` and the tag's
    category threshold."""

    tags: np.ndarray
    on_time: np.ndarray
    on_score: np.ndarray
    peak: np.ndarray | None = None
    peak_row: np.ndarray | None = None
    repeats: np.ndarray | None = None
    over: np.ndarray | None = None


def _tag_stats(frames, grouping, seen, in_tags) -> _Stats:
    """The scene's general and character tags over its non-blank frames `seen` (in time order), a few hundred tags
    at a time so a long scene's dense scores stay small."""
    row, tag, value = frames.scores.entries(seen)
    keep = in_tags[tag] & (value >= SCORE_FLOOR)
    row, tag, value = row[keep], tag[keep], value[keep]
    tags, column = np.unique(tag, return_inverse=True)
    count = len(tags)
    stats = _Stats(
        tags,
        on_time=np.zeros(count),
        on_score=np.zeros(count),
        peak=np.zeros(count),
        peak_row=np.zeros(count, np.int64),
        repeats=np.zeros(count, np.int64),
        over=np.zeros(count, np.int64),
    )
    if not len(tags):
        return stats
    times = np.asarray(frames.timestamps, dtype=np.float64)[seen]
    span = np.maximum(grouping.end[seen] - grouping.start[seen], 1e-9)
    categories = np.asarray(frames.categories, dtype=object)[tags]
    on_at = np.array([frames.thresholds[c] for c in categories])
    off_at = on_at - np.array([OFF_GAP[c] for c in categories])
    by_column = np.argsort(column, kind="stable")
    bounds = np.searchsorted(column[by_column], np.arange(0, len(tags) + _CHUNK_TAGS, _CHUNK_TAGS))
    for chunk, lo in enumerate(range(0, len(tags), _CHUNK_TAGS)):
        hi = min(lo + _CHUNK_TAGS, len(tags))
        part = by_column[bounds[chunk] : bounds[chunk + 1]]
        raw = np.zeros((len(seen), hi - lo))
        raw[row[part], column[part] - lo] = value[part]
        on = _hysteresis(_smooth(raw, times, span), on_at[lo:hi], off_at[lo:hi])
        stats.on_time[lo:hi] = span @ on
        stats.on_score[lo:hi] = span @ (on * raw)
        stats.peak_row[lo:hi] = raw.argmax(axis=0)
        stats.peak[lo:hi] = raw.max(axis=0)
        stats.repeats[lo:hi] = (raw >= PEAK_REPEAT).sum(axis=0)
        stats.over[lo:hi] = (raw >= on_at[lo:hi]).sum(axis=0)
    return stats


def _smooth(raw: np.ndarray, times: np.ndarray, span: np.ndarray) -> np.ndarray:
    """Each row of `raw` (frames in time order) mixed with the others by a Gaussian over their times, σ =
    `SMOOTHING`, each weighted by its span; a frame keeps at least half of the weight, its own."""
    if SMOOTHING <= 0 or len(raw) < 2:
        return raw
    total = np.zeros_like(raw)
    weight = np.zeros(len(raw))
    for lag in range(1, len(raw)):
        gap = times[lag:] - times[:-lag]
        near = gap <= SMOOTHING_REACH * SMOOTHING
        if not near.any():
            break
        g = np.exp(-0.5 * (gap / SMOOTHING) ** 2) * near
        later, earlier = g * span[lag:], g * span[:-lag]
        total[:-lag] += later[:, None] * raw[lag:]
        weight[:-lag] += later
        total[lag:] += earlier[:, None] * raw[:-lag]
        weight[lag:] += earlier
    own = np.maximum(weight, span)
    return (total + own[:, None] * raw) / (weight + own)[:, None]


def _hysteresis(scores: np.ndarray, on_at: np.ndarray, off_at: np.ndarray) -> np.ndarray:
    """Per column, frame by frame: on from a score reaching `on_at`, until one falls below `off_at`."""
    on = np.zeros(scores.shape, dtype=bool)
    state = np.zeros(scores.shape[1], dtype=bool)
    for i, row in enumerate(scores):
        state = np.where(state, row >= off_at, row >= on_at)
        on[i] = state
    return on


def _tag_list(frames, settings, tags, coverage, stats) -> list[dict]:
    """The tags whose coverage reaches `min_coverage`, by coverage then score, at most `max_tags`."""
    listed = []
    for at in np.flatnonzero(coverage >= settings.min_coverage):
        score = stats.on_score[at] / stats.on_time[at] if stats.on_time[at] > 0 else 0.0
        listed.append((float(coverage[at]), float(score), int(tags[at])))
    listed.sort(key=lambda entry: (-entry[0], -entry[1]))
    return [
        {"name": frames.names[t], "category": frames.categories[t], "coverage": _score(c), "score": _score(s)}
        for c, s, t in listed[: settings.max_tags]
    ]


def _highlights(frames, grouping, settings, seen, stats, highlightable, listed) -> list[dict]:
    """Short-lived tags from raw scores: a general tag reaching `PEAK_STRONG` in one frame or `PEAK_REPEAT` in two, a
    character reaching its threshold in two. Shot-specific, ignored and listed tags are not highlights."""
    found = []
    for at, t in enumerate(stats.tags):
        name, category = frames.names[t], frames.categories[t]
        if not highlightable[t] or name in listed:
            continue
        if category == "general":
            qualifies = stats.peak[at] >= PEAK_STRONG or stats.repeats[at] >= 2
        else:
            qualifies = stats.over[at] >= 2
        if qualifies:
            found.append((float(stats.peak[at]), at))
    found.sort(key=lambda entry: -entry[0])
    return [
        {
            "name": frames.names[stats.tags[at]],
            "category": frames.categories[stats.tags[at]],
            "score": _score(peak),
            "frames": int(stats.repeats[at]),
            **_at(frames, seen[stats.peak_row[at]]),
        }
        for peak, at in found[: settings.max_tags]
    ]
