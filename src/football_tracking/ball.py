"""Ball cleaning, interpolation and possession (Milestone 5).

Input: the table from Milestone 4 (outputs/m4_tracks_teams.csv). Ball rows have tracker_id == -1
(Milestone 3 kept the single most confident ball box per frame).

Steps
  1. ball_table          one row per frame: ball centre, size, confidence (NaN when not detected)
  2. flag_outliers       a detection far from the rolling MEDIAN of its neighbours is a false ball
  3. interpolate_ball    fill short gaps (<= max_gap frames) with straight lines; long gaps stay unknown
  4. assign_possession   per frame: the closest player whose feet are near the ball (scaled by his height)
  5. smooth_possession   ignore 1-2 frame flickers; the team in control keeps it until another team touches
  6. possession_stats / touch_events   percentages, passes and turnovers
Only NumPy, pandas and OpenCV are used, so everything is easy to test.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

NO_ID = -1


# --------------------------------------------------------------------------- 1. one row per frame
def ball_table(df: pd.DataFrame, n_frames: int, class_col: str = "voted_class_name") -> pd.DataFrame:
    """Ball position per frame. Columns: cx, cy (box centre), size (box width), conf, detected."""
    b = df[(df[class_col] == "ball") | (df.get("class_name", "") == "ball")]
    b = b.sort_values("conf", ascending=False).drop_duplicates("frame")      # at most one ball per frame
    out = pd.DataFrame(index=pd.RangeIndex(n_frames, name="frame"))
    out["cx"] = ((b.x1 + b.x2) / 2).set_axis(b.frame).reindex(out.index)
    out["cy"] = ((b.y1 + b.y2) / 2).set_axis(b.frame).reindex(out.index)
    out["size"] = (b.x2 - b.x1).set_axis(b.frame).reindex(out.index)
    out["conf"] = b.conf.set_axis(b.frame).reindex(out.index)
    out["detected"] = out.cx.notna()
    return out


# --------------------------------------------------------------------------- 2. false balls
def flag_outliers(ball: pd.DataFrame, window: int = 7, max_dev: float = 40.0) -> pd.Series:
    """True for detections that are far (> max_dev px) from the rolling median of nearby detections.

    Why the median works: if the ball moves along a straight line, the median of 7 consecutive
    positions is exactly the middle one, so real fast movement is NOT flagged. A false ball
    (a white boot, a penalty spot) is a single jump away from the path and stands out.
    """
    det = ball[ball.detected]
    if len(det) < 3:
        return pd.Series(False, index=ball.index)
    mx = det.cx.rolling(window, center=True, min_periods=3).median()
    my = det.cy.rolling(window, center=True, min_periods=3).median()
    dev = np.hypot(det.cx - mx, det.cy - my)
    return (dev > max_dev).reindex(ball.index, fill_value=False)


# --------------------------------------------------------------------------- 3. fill gaps
def gap_lengths(missing: pd.Series) -> pd.Series:
    """For every missing frame: the length of the run of missing frames it belongs to (0 if present)."""
    run_id = (missing != missing.shift()).cumsum()
    length = missing.groupby(run_id).transform("size")
    return length.where(missing, 0)


def interpolate_ball(ball: pd.DataFrame, outliers: pd.Series | None = None, max_gap: int = 25) -> pd.DataFrame:
    """Adds x, y (cleaned + interpolated position) and `source`:
    'detected', 'interpolated', 'outlier' (removed, then maybe filled), or 'missing' (left unknown).

    Only gaps with a known ball on BOTH sides and at most `max_gap` frames are filled. A long gap usually
    means the ball left the picture or flew high; drawing a straight line there would invent data.
    """
    out = ball.copy()
    bad = outliers if outliers is not None else pd.Series(False, index=out.index)
    good = out.detected & ~bad
    x, y = out.cx.where(good), out.cy.where(good)
    missing = ~good
    gl = gap_lengths(missing)
    inside = x.ffill().notna() & x.bfill().notna()                  # a known ball before AND after
    fillable = missing & inside & (gl <= max_gap)
    xi, yi = x.interpolate(limit_area="inside"), y.interpolate(limit_area="inside")
    out["x"] = x.where(~fillable, xi)
    out["y"] = y.where(~fillable, yi)
    out["source"] = np.select([good, fillable], ["detected", "interpolated"], default="missing")
    out.loc[bad & ~fillable, "source"] = "outlier"
    out.loc[bad & fillable, "source"] = "interpolated"
    return out


# --------------------------------------------------------------------------- 4. who is closest
def assign_possession(df: pd.DataFrame, ball: pd.DataFrame, class_col: str = "voted_class_name",
                      classes=("player", "goalkeeper"), max_ratio: float = 0.5) -> pd.DataFrame:
    """Per frame: the player whose FEET are closest to the ball, if close enough.

    Feet = bottom-centre of the box. Distance is divided by the player's box height so the rule works
    whether the camera is zoomed in or out: max_ratio = 0.5 means "within half a body height".
    Returns frame, holder_id, holder_team, ratio (NaN / -1 when nobody is close or the ball is unknown).
    """
    p = df[(df.tracker_id >= 0) & df[class_col].isin(classes)][["frame", "tracker_id", "team", "x1", "y1", "x2", "y2"]].copy()
    p = p.merge(ball[["x", "y"]].dropna(), left_on="frame", right_index=True)
    p["h"] = (p.y2 - p.y1).clip(lower=1)
    p["ratio"] = np.hypot((p.x1 + p.x2) / 2 - p.x, p.y2 - p.y) / p.h
    best = p.sort_values("ratio").drop_duplicates("frame").set_index("frame")
    best = best[best.ratio <= max_ratio]
    out = pd.DataFrame(index=ball.index)
    out["holder_id"] = best.tracker_id.reindex(out.index).fillna(NO_ID).astype(int)
    out["holder_team"] = best.team.reindex(out.index).fillna(NO_ID).astype(int)
    out["ratio"] = best.ratio.reindex(out.index)
    return out


# --------------------------------------------------------------------------- 5. smoothing
def smooth_possession(poss: pd.DataFrame, min_frames: int = 3) -> pd.DataFrame:
    """Two clean-up rules:
    * a new holder must keep the ball for `min_frames` frames in a row, else the frame is ignored
      (a player running past the ball for 1 frame is not a touch);
    * between touches (a pass in the air), the team that touched last stays in control.
    Adds holder_id_s, holder_team_s (debounced) and team_in_control.
    """
    out = poss.copy()
    h = out.holder_id
    run_id = (h != h.shift()).cumsum()
    run_len = h.groupby(run_id).transform("size")
    keep = (h != NO_ID) & (run_len >= min_frames)
    out["holder_id_s"] = h.where(keep, NO_ID)
    out["holder_team_s"] = out.holder_team.where(keep, NO_ID)
    out["team_in_control"] = out.holder_team_s.replace(NO_ID, np.nan).ffill().fillna(NO_ID).astype(int)
    return out


# --------------------------------------------------------------------------- 6. numbers
def possession_stats(poss: pd.DataFrame) -> pd.DataFrame:
    """Share of frames each team is in control (frames before the first touch are not counted)."""
    c = poss.team_in_control[poss.team_in_control != NO_ID]
    pct = c.value_counts(normalize=True).sort_index() * 100
    return pd.DataFrame({"frames": c.value_counts().sort_index(), "possession_%": pct.round(1)})


def touch_events(poss: pd.DataFrame) -> pd.DataFrame:
    """Sequence of touches (debounced holders) and what happened between them:
    same team, other player = 'pass'; other team = 'turnover'; same player again = 'continue'."""
    t = poss[poss.holder_id_s != NO_ID][["holder_id_s", "holder_team_s"]]
    seg_id = (t.holder_id_s != t.holder_id_s.shift()).cumsum()
    seg = t.reset_index().groupby(seg_id.values).agg(start=("frame", "min"), end=("frame", "max"),
                                                     player=("holder_id_s", "first"), team=("holder_team_s", "first"))
    prev_p, prev_t = seg.player.shift(), seg.team.shift()
    seg["event"] = np.select([prev_p.isna(), seg.player == prev_p, seg.team == prev_t],
                             ["first touch", "continue", "pass"], default="turnover")
    return seg.reset_index(drop=True)


# --------------------------------------------------------------------------- drawing
BALL_DET = (0, 220, 255)       # yellow: detected
BALL_INT = (0, 140, 255)       # orange: interpolated


class PossessionAnnotator:
    """Ball marker (filled = detected, ring = interpolated), a marker above the player on the ball,
    and a possession bar (running percentage) in the top-left corner."""

    def __init__(self, team_colours: dict[int, tuple], team_names: dict[int, str]):
        self.colours, self.names = team_colours, team_names

    def __call__(self, frame, ball_row, holder_box, holder_team, running_pct: dict[int, float]):
        out = frame.copy()
        if ball_row is not None and ball_row.source in ("detected", "interpolated") and not np.isnan(ball_row.x):
            c = (int(ball_row.x), int(ball_row.y))
            if ball_row.source == "detected":
                cv2.circle(out, c, 7, BALL_DET, -1, cv2.LINE_AA); cv2.circle(out, c, 7, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.circle(out, c, 8, BALL_INT, 2, cv2.LINE_AA)
        if holder_box is not None and holder_team >= 0:
            x1, y1, x2, _ = map(int, holder_box)
            cx = (x1 + x2) // 2
            pts = np.array([[cx, y1 - 6], [cx - 9, y1 - 22], [cx + 9, y1 - 22]], np.int32)
            cv2.fillPoly(out, [pts], self.colours.get(holder_team, (255, 255, 255)))
            cv2.polylines(out, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)
        self._panel(out, running_pct)
        return out

    def _panel(self, img, pct):
        x, y, w, h = 20, 20, 320, 70
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)          # dark panel: white kits stay visible
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.putText(img, "Possession", (x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        teams = sorted(self.colours)
        left = x + 10
        bar_w = w - 20
        for t in teams:
            seg = int(bar_w * pct.get(t, 0) / 100)
            cv2.rectangle(img, (left, y + 32), (left + seg, y + 46), self.colours[t], -1)
            left += seg
        cv2.rectangle(img, (x + 10, y + 32), (x + 10 + bar_w, y + 46), (200, 200, 200), 1)
        text = "   ".join(f"{self.names.get(t, t)} {pct.get(t, 0):.0f}%" for t in teams)
        cv2.putText(img, text, (x + 10, y + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)


def running_possession(poss: pd.DataFrame) -> pd.DataFrame:
    """Possession percentage per team up to each frame (for the on-screen bar)."""
    c = poss.team_in_control
    teams = sorted(t for t in c.unique() if t != NO_ID)
    counts = pd.DataFrame({t: (c == t).cumsum() for t in teams}, index=poss.index)
    total = counts.sum(axis=1).replace(0, np.nan)
    return (counts.div(total, axis=0) * 100).fillna(0)
