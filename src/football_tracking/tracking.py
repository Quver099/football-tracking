"""Multi-object tracking (Milestone 3).

Detection answers "what is where in THIS frame". Tracking answers "is this box the same
person as in the previous frame?" and gives every person a persistent `tracker_id`.

Design decisions
----------------
* The ball is NOT tracked with ByteTrack. It is tiny, fast and often missing, so a Kalman
  motion model does badly on it. We keep one ball per frame (highest confidence) and
  handle gaps in Milestone 5 (interpolation).
* Players, goalkeepers and referees go through class-agnostic NMS first, then ByteTrack.
  Class-agnostic NMS matters: the fine-tuned model sometimes outputs BOTH a "player" and a
  "goalkeeper" box on the same person, which would otherwise become two tracks.
* The class of a track is decided by a confidence-weighted vote over all its frames, so a
  goalkeeper that is called "player" in 8 frames out of 100 stays a goalkeeper.

The voting / metrics code is pure NumPy + pandas (no supervision needed) so it is easy to test.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import supervision as sv

NO_ID = -1


# --------------------------------------------------------------------------- tracker wrapper
@dataclass
class TrackerConfig:
    """ByteTrack settings (names follow supervision.ByteTrack).

    activation_threshold : a detection must be at least this confident to START a new track.
    lost_buffer          : frames a track survives without a match before it is deleted.
                           Bigger = bridges longer occlusions but risks wrong re-links.
    match_threshold      : max matching DISTANCE (1 - IoU) to link a box to a track. Higher = more lenient
                           (0.8 accepts IoU >= 0.2); lower = stricter (0.6 needs IoU >= 0.4).
    min_consecutive      : a track gets an ID only after it was matched this many frames in a row
                           (filters one-frame ghosts, but delays the first label).
    nms_iou              : class-agnostic NMS threshold applied before tracking.
    """

    activation_threshold: float = 0.25
    lost_buffer: int = 50
    match_threshold: float = 0.8
    min_consecutive: int = 2
    nms_iou: float = 0.6


class PlayerTracker:
    """ByteTrack for people + 'keep the best box' for the ball.

    Usage:
        tracker = PlayerTracker(fps=25, ball_id=0, config=TrackerConfig())
        for frame_idx, dets in enumerate(per_frame_detections):
            tracked = tracker.update(dets)      # sv.Detections with .tracker_id set
    """

    def __init__(self, fps: float, ball_id: int, config: TrackerConfig | None = None):
        self.cfg = config or TrackerConfig()
        self.ball_id = ball_id
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.cfg.activation_threshold,
            lost_track_buffer=self.cfg.lost_buffer,
            minimum_matching_threshold=self.cfg.match_threshold,
            frame_rate=int(round(fps)),
            minimum_consecutive_frames=self.cfg.min_consecutive,
        )

    def reset(self) -> None:
        self.tracker.reset()

    def update(self, det: sv.Detections) -> sv.Detections:
        is_ball = det.class_id == self.ball_id
        people = det[~is_ball].with_nms(threshold=self.cfg.nms_iou, class_agnostic=True)
        people = self.tracker.update_with_detections(people)
        # unconfirmed tracks may come back with tracker_id == -1; drop them
        if people.tracker_id is not None and len(people):
            people = people[people.tracker_id != NO_ID]

        balls = det[is_ball]
        if len(balls):
            balls = balls[[int(np.argmax(balls.confidence))]]
            balls.tracker_id = np.array([NO_ID])
        if len(people) == 0:
            return balls if len(balls) else people
        if len(balls) == 0:
            return people
        return sv.Detections.merge([people, balls])


# --------------------------------------------------------------------------- storing results
def tracks_to_frame(per_frame: list[sv.Detections], names: dict[int, str]) -> pd.DataFrame:
    """One row per (frame, box): the table every later milestone reads.

    Columns: frame, tracker_id, class_id, class_name, conf, x1, y1, x2, y2
    """
    rows = []
    for f, d in enumerate(per_frame):
        if len(d) == 0:
            continue
        tid = d.tracker_id if d.tracker_id is not None else np.full(len(d), NO_ID)
        for k in range(len(d)):
            x1, y1, x2, y2 = d.xyxy[k]
            rows.append((f, int(tid[k]), int(d.class_id[k]), names[int(d.class_id[k])],
                         float(d.confidence[k]), float(x1), float(y1), float(x2), float(y2)))
    return pd.DataFrame(rows, columns=["frame", "tracker_id", "class_id", "class_name",
                                       "conf", "x1", "y1", "x2", "y2"])


# --------------------------------------------------------------------------- majority-vote class
def vote_classes(df: pd.DataFrame, weight_by_conf: bool = True) -> dict[int, int]:
    """Final class per tracker_id = class with the largest (confidence-weighted) vote.

    Ball rows (tracker_id == -1) are ignored. Ties go to the lowest class id (deterministic).
    """
    result: dict[int, int] = {}
    for tid, g in df[df.tracker_id != NO_ID].groupby("tracker_id"):
        w = g.conf if weight_by_conf else pd.Series(1.0, index=g.index)
        score = w.groupby(g.class_id).sum()
        result[int(tid)] = int(score[score == score.max()].index.min())
    return result


class OnlineClassVoter:
    """Running vote for live use: the class shown for a track may change early, then settles.

    Offline (we have the whole clip) use `vote_classes` instead: it is more accurate
    because it sees every frame of the track before deciding.
    """

    def __init__(self, weight_by_conf: bool = True):
        self.weight_by_conf = weight_by_conf
        self.scores: dict[int, Counter] = defaultdict(Counter)

    def update(self, det: sv.Detections) -> np.ndarray:
        """Returns the voted class id for each box in `det` (ball rows keep their own class)."""
        out = det.class_id.copy()
        if det.tracker_id is None:
            return out
        for k, tid in enumerate(det.tracker_id):
            if tid == NO_ID:
                continue
            self.scores[int(tid)][int(det.class_id[k])] += float(det.confidence[k]) if self.weight_by_conf else 1.0
            out[k] = self.scores[int(tid)].most_common(1)[0][0]
        return out


def apply_votes(df: pd.DataFrame, votes: dict[int, int], names: dict[int, str]) -> pd.DataFrame:
    """Adds `voted_class_id` / `voted_class_name`; ball rows keep their own class."""
    out = df.copy()
    out["voted_class_id"] = [votes.get(t, c) if t != NO_ID else c for t, c in zip(out.tracker_id, out.class_id)]
    out["voted_class_name"] = out.voted_class_id.map(names)
    return out


# --------------------------------------------------------------------------- quality metrics
def class_flicker(df: pd.DataFrame, class_col: str = "class_id") -> pd.DataFrame:
    """Per track: how many times does its class change from one appearance to the next?"""
    rows = []
    for tid, g in df[df.tracker_id != NO_ID].sort_values("frame").groupby("tracker_id"):
        c = g[class_col].to_numpy()
        rows.append((int(tid), len(g), int((c[1:] != c[:-1]).sum()), len(set(c))))
    return pd.DataFrame(rows, columns=["tracker_id", "frames", "class_changes", "n_classes"])


def track_summary(df: pd.DataFrame, expected_people: int = 25) -> dict:
    """Numbers to judge a tracker run without watching the video.

    ids_created      : unique IDs. A clip with ~22 players + referees should be close to this number;
                       far more means tracks keep breaking ("fragmentation").
    short_tracks     : IDs that live < 1 s (25 frames): mostly ghosts or broken pieces.
    gaps             : times a track disappears and reappears with the same ID (good: occlusion bridged).
    people_per_frame : average number of tracked people visible per frame.
    """
    p = df[df.tracker_id != NO_ID]
    if p.empty:
        return {"ids_created": 0}
    g = p.groupby("tracker_id").frame
    length = g.count()
    span = g.max() - g.min() + 1
    gaps = int((span - length).gt(0).sum())
    return {
        "ids_created": int(length.size),
        "expected_people": expected_people,
        "median_track_len": float(length.median()),
        "short_tracks_lt_25f": int((length < 25).sum()),
        "tracks_with_gaps": gaps,
        "people_per_frame": float(p.groupby("frame").size().mean()),
    }


def id_switch_suspects(df: pd.DataFrame, max_jump_px: float = 150.0) -> pd.DataFrame:
    """Rows where a track's box centre jumps further than `max_jump_px` between consecutive frames.

    A real player moves a few pixels per frame; a huge jump usually means the ID was handed to a
    different person (an ID switch) or the camera cut. Use it to find moments worth watching.
    """
    p = df[df.tracker_id != NO_ID].sort_values(["tracker_id", "frame"]).copy()
    p["cx"] = (p.x1 + p.x2) / 2
    p["cy"] = (p.y1 + p.y2) / 2
    p["dx"] = p.groupby("tracker_id").cx.diff()
    p["dy"] = p.groupby("tracker_id").cy.diff()
    p["df"] = p.groupby("tracker_id").frame.diff()
    p["jump_per_frame"] = np.hypot(p.dx, p.dy) / p["df"]
    p = p[p["df"] > 0]
    return p[p.jump_per_frame > max_jump_px][["frame", "tracker_id", "jump_per_frame"]].reset_index(drop=True)


# --------------------------------------------------------------------------- drawing
class TrackAnnotator:
    """Boxes coloured by track ID, label '#id class', plus a short motion trail."""

    def __init__(self, trail: int = 30):
        self.box = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
        self.label = sv.LabelAnnotator(text_scale=0.5, text_padding=3, color_lookup=sv.ColorLookup.TRACK)
        self.trace = sv.TraceAnnotator(trace_length=trail, thickness=2, color_lookup=sv.ColorLookup.TRACK)

    def __call__(self, frame: np.ndarray, det: sv.Detections, names: dict[int, str],
                 class_ids: np.ndarray | None = None) -> np.ndarray:
        cls = det.class_id if class_ids is None else class_ids
        people = det[det.tracker_id != NO_ID] if det.tracker_id is not None else det[:0]
        pcls = cls[det.tracker_id != NO_ID] if det.tracker_id is not None else cls[:0]
        out = frame.copy()
        out = self.trace.annotate(out, people)
        out = self.box.annotate(out, people)
        labels = [f"#{t} {names[int(c)]}" for t, c in zip(people.tracker_id, pcls)]
        out = self.label.annotate(out, people, labels=labels)
        balls = det[det.tracker_id == NO_ID] if det.tracker_id is not None else det
        for x1, y1, x2, y2 in balls.xyxy:  # ball: small yellow marker above the box
            cx, cy = int((x1 + x2) / 2), int(y1) - 8
            pts = np.array([[cx, cy], [cx - 7, cy - 12], [cx + 7, cy - 12]], dtype=np.int32)
            import cv2
            cv2.fillPoly(out, [pts], (0, 220, 255))
        return out


# --------------------------------------------------------------------------- ghost-track cleanup
def ghost_track_ids(df: pd.DataFrame, min_len: int = 25, min_conf: float = 0.6) -> set[int]:
    """IDs of tracks that are BOTH short (< min_len frames) AND never confident (max conf < min_conf).

    Why both: a short but confident track is usually a real person whose ID broke (keep: it still
    gives positions). A long but unsure track is usually a real person far from the camera (keep).
    Only short + unsure tracks are almost certainly ghosts (a shadow, a boot, a flicker in the crowd).
    """
    p = df[df.tracker_id != NO_ID].groupby("tracker_id").agg(n=("frame", "size"), top=("conf", "max"))
    return set(int(t) for t in p[(p.n < min_len) & (p.top < min_conf)].index)


def drop_tracks(per_frame: list[sv.Detections], ids: set[int]) -> list[sv.Detections]:
    """Remove the given tracker_ids from the per-frame detections (ball rows, id -1, are kept)."""
    if not ids:
        return per_frame
    bad = np.fromiter(ids, dtype=int)
    out = []
    for d in per_frame:
        if len(d) == 0 or d.tracker_id is None:
            out.append(d)
        else:
            out.append(d[~np.isin(d.tracker_id, bad)])
    return out


# --------------------------------------------------------------------------- ID-switch repair
def split_on_class_change(df: pd.DataFrame, class_col: str = "class_name", window: int = 25,
                          min_len: int = 25) -> tuple[pd.DataFrame, list[tuple[int, int, int]]]:
    """Split a track where the DETECTOR's class changes for a long stretch.

    Example: track #15 is 'referee' for 680 frames, then 'player' for 60 frames in a row. That is
    an ID switch (the referee's ID moved onto a player), not class flicker. The class vote would
    call the whole track 'referee'. Here the second part gets a new tracker_id instead.

    window  : the per-frame class is smoothed with a rolling majority over this many frames,
              so short flicker (a few wrong frames) never causes a split.
    min_len : a new class must last at least this many frames to start a new track.
    Returns (new table, list of (old_id, new_id, first_frame)).
    """
    out = df.copy()
    next_id = int(out.tracker_id.max()) + 1
    log = []
    for tid, g in out[out.tracker_id >= 0].groupby("tracker_id"):
        g = g.sort_values("frame")
        codes = pd.Series(pd.factorize(g[class_col])[0], index=g.index)
        if codes.nunique() < 2:
            continue
        smooth = codes.rolling(window, center=True, min_periods=1).apply(
            lambda x: np.bincount(x.astype(int)).argmax(), raw=True).astype(int)
        segment = (smooth != smooth.shift()).cumsum()
        cur_cls, cur_id = None, int(tid)
        for _, s in g.groupby(segment, sort=False):
            cls = int(smooth.loc[s.index[0]])
            if len(s) >= min_len:
                if cur_cls is not None and cls != cur_cls:
                    cur_id, next_id = next_id, next_id + 1
                    log.append((int(tid), cur_id, int(s.frame.min())))
                cur_cls = cls
            out.loc[s.index, "tracker_id"] = cur_id
    return out, log
