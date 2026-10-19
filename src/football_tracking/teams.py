"""Team assignment (Milestone 4).

Idea: players of one team wear the same shirt colour. So:
  1. crop the shirt area of each player box,
  2. throw away background pixels (per player: the colour group that owns the box corners;
     a fixed green range would also delete GREEN SHIRTS), take the median colour of what is left,
  3. cluster all crop colours into 2 groups with KMeans (unsupervised: no labels needed),
  4. give each TRACK the team most of its crops voted for (same idea as the class vote in M3),
  5. goalkeepers wear a different kit, so they are assigned by position (closest team), not colour,
  6. referees get no team.

Only OpenCV, NumPy, pandas and scikit-learn are used here (no supervision), so it is easy to test.
Input is the track table written by Milestone 3 (outputs/m3_tracks.csv) plus the video.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

NO_TEAM = -1
GRASS_LOWER = (30, 40, 40)      # HSV, same range as pitch.py
GRASS_UPPER = (90, 255, 255)
SPACES = {"bgr": ["B", "G", "R"], "hsv": ["H", "S", "V"], "lab": ["L", "A", "Bb"]}


# --------------------------------------------------------------------------- 1-2. one colour per crop
@dataclass
class ShirtRegion:
    """Which part of the player box is the shirt, as fractions of box height/width.

    Top 15%: head and hair. Below 50%: shorts, legs, grass. Outer 25% on each side: arms,
    background and neighbouring players. What is left is mostly shirt.
    """
    top: float = 0.15
    bottom: float = 0.50
    left: float = 0.25
    right: float = 0.75


def shirt_crop(frame: np.ndarray, box, region: ShirtRegion = ShirtRegion()) -> np.ndarray:
    """Cut the shirt area out of one player box (BGR). Returns an empty array if too small."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    a, b = int(y1 + region.top * h), int(y1 + region.bottom * h)
    c, d = int(x1 + region.left * w), int(x1 + region.right * w)
    a, b, c, d = max(a, 0), min(b, h_img), max(c, 0), min(d, w_img)
    if b - a < 3 or d - c < 3:
        return np.empty((0, 0, 3), np.uint8)
    return frame[a:b, c:d]


def non_grass_pixels(crop: np.ndarray) -> np.ndarray:
    """All pixels of the crop that are NOT grass, as an (N, 3) BGR array."""
    if crop.size == 0:
        return np.empty((0, 3), np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, np.array(GRASS_LOWER), np.array(GRASS_UPPER)) > 0
    return crop[~grass]


def background_mask_corners(img: np.ndarray, threshold: float = 18.0, k_frac: int = 6) -> np.ndarray:
    """Per-player background removal that does NOT assume grass is green.

    The four corners of a player box are almost always pitch (beside the head and beside the feet).
    So we measure THIS player's local grass colour at the 4 corners (one median colour per corner,
    which also covers light and dark pitch stripes) and remove every pixel whose Lab colour is within
    `threshold` of any corner colour. Everything else is player. Returns a boolean mask: True = player.

    Why it works for green kits: the shirt only has to differ from the grass right next to it,
    not from a fixed "green" range that would also swallow green shirts.
    """
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    c = max(1, min(h, w) // k_frac)
    corner_cols = np.array([np.median(p.reshape(-1, 3), axis=0) for p in
                            (lab[:c, :c], lab[:c, -c:], lab[-c:, :c], lab[-c:, -c:])])
    dist = np.linalg.norm(lab[:, :, None, :] - corner_cols[None, None, :, :], axis=-1).min(axis=-1)
    return dist > threshold


def shirt_pixels(frame: np.ndarray, box, region: ShirtRegion = ShirtRegion(), method: str = "corners",
                 threshold: float = 18.0):
    """Shirt pixels of one player box, plus a display crop (background painted white).

    method:
      "corners" (default): remove pixels close to the grass colour at the box corners (any kit colour works)
      "hsv"              : remove a FIXED green range (fails when a team wears green)
      "none"             : keep every pixel of the shirt region
    """
    if method != "corners":
        crop = shirt_crop(frame, box, region)
        if method == "none":
            return crop.reshape(-1, 3), crop
        px = non_grass_pixels(crop)
        disp = crop.copy()
        if crop.size:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            disp[cv2.inRange(hsv, np.array(GRASS_LOWER), np.array(GRASS_UPPER)) > 0] = 255
        return px, disp
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w_img), min(y2, h_img)
    bh, bw = y2 - y1, x2 - x1
    whole = frame[y1:y2, x1:x2]
    if bh < 10 or bw < 6:
        return np.empty((0, 3), np.uint8), np.empty((0, 0, 3), np.uint8)
    player = background_mask_corners(whole, threshold)
    a, b = int(region.top * bh), int(region.bottom * bh)
    c, d = int(region.left * bw), int(region.right * bw)
    sub, m = whole[a:b, c:d], player[a:b, c:d]
    disp = sub.copy(); disp[~m] = 255
    return sub[m], disp


def colour_features(pixels_bgr: np.ndarray) -> dict | None:
    """Median colour of the pixels in three colour spaces (BGR, HSV, Lab). None if too few pixels."""
    if len(pixels_bgr) < 30:
        return None
    px = pixels_bgr.reshape(-1, 1, 3).astype(np.uint8)
    out = {}
    for space, cols, code in [("bgr", SPACES["bgr"], None), ("hsv", SPACES["hsv"], cv2.COLOR_BGR2HSV),
                              ("lab", SPACES["lab"], cv2.COLOR_BGR2LAB)]:
        conv = px if code is None else cv2.cvtColor(px, code)
        med = np.median(conv.reshape(-1, 3), axis=0)
        out.update(dict(zip(cols, med.astype(float))))
    out["n_pixels"] = int(len(pixels_bgr))
    return out


# --------------------------------------------------------------------------- choosing good crops
def overlap_with_others(df: pd.DataFrame) -> pd.Series:
    """For every row: the largest IoU with any OTHER person box in the same frame.

    A crop of a player who overlaps someone else mixes two shirts, so we prefer isolated players.
    """
    out = pd.Series(0.0, index=df.index)
    for _, g in df.groupby("frame"):
        if len(g) < 2:
            continue
        b = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        ix1 = np.maximum(b[:, None, 0], b[None, :, 0]); iy1 = np.maximum(b[:, None, 1], b[None, :, 1])
        ix2 = np.minimum(b[:, None, 2], b[None, :, 2]); iy2 = np.minimum(b[:, None, 3], b[None, :, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / (area[:, None] + area[None, :] - inter + 1e-9)
        np.fill_diagonal(iou, 0)
        out.loc[g.index] = iou.max(axis=1)
    return out


def sample_rows(df: pd.DataFrame, class_col: str = "voted_class_name", classes=("player",),
                per_track: int = 20, max_overlap: float = 0.1, min_height: int = 40, seed: int = 0) -> pd.DataFrame:
    """Pick up to `per_track` good rows per track: right class, tall enough, not overlapping others,
    spread over the track's life (random sample, not just the first frames)."""
    p = df[(df.tracker_id >= 0) & df[class_col].isin(classes)].copy()
    p["height"] = p.y2 - p.y1
    p["overlap"] = overlap_with_others(df[df.tracker_id >= 0]).reindex(p.index).fillna(0)
    good = p[(p.height >= min_height) & (p.overlap <= max_overlap)].assign(fallback=False)
    # tracks that are NEVER isolated (always in a crowd) still need a team: take their least-overlapped rows
    missing = set(p.tracker_id) - set(good.tracker_id)
    if missing:
        fb = (p[p.tracker_id.isin(missing)].sort_values("overlap")
                .groupby("tracker_id").head(max(per_track // 2, 1)).assign(fallback=True))
        good = pd.concat([good, fb])
    shuffled = good.sample(frac=1.0, random_state=seed)                 # random order...
    keep = shuffled.groupby("tracker_id").cumcount() < per_track         # ...then first N per track
    return shuffled[keep].sort_values("frame")


def collect_features(video_path, rows: pd.DataFrame, region: ShirtRegion = ShirtRegion(),
                     keep_crops: int = 0, method: str = "corners") -> tuple[pd.DataFrame, list]:
    """Read the video ONCE and compute colour features for every selected row.

    Returns (features table, list of (tracker_id, display crop)) where at most `keep_crops`
    crops (background painted white) are kept for plotting; `feats.crop_idx` points into that list.
    """
    need = rows.groupby("frame")
    wanted = set(need.groups)
    feats, crops = [], []
    cap = cv2.VideoCapture(str(video_path))
    f = 0
    try:
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if f in wanted:
                for idx, r in need.get_group(f).iterrows():
                    px, crop = shirt_pixels(frame, (r.x1, r.y1, r.x2, r.y2), region, method)
                    fe = colour_features(px)
                    if fe is None:
                        continue
                    fe["kept_frac"] = len(px) / max(crop.shape[0] * crop.shape[1], 1)   # share of the region kept
                    fe.update(row=idx, frame=f, tracker_id=int(r.tracker_id), fallback=bool(getattr(r, "fallback", False)))
                    feats.append(fe)
                    fe["crop_idx"] = -1                         # position in `crops`, -1 = not stored
                    if len(crops) < keep_crops:
                        fe["crop_idx"] = len(crops)
                        crops.append((int(r.tracker_id), crop.copy()))
                wanted.discard(f)
            f += 1
    finally:
        cap.release()
    return pd.DataFrame(feats), crops


# --------------------------------------------------------------------------- 3. clustering
class TeamClassifier:
    """KMeans with k=2 on one colour per crop.

    KMeans numbers its clusters 0/1 at random. To make the result repeatable we RE-ORDER the clusters
    so that team 0 is always the team with the lighter shirt (higher Lab L). Then you name them.
    """

    def __init__(self, space: str = "lab", k: int = 2, seed: int = 0):
        self.space, self.cols, self.k, self.seed = space, SPACES[space], k, seed

    def fit(self, feats: pd.DataFrame) -> "TeamClassifier":
        """Fit on clean crops only (fallback crops from crowded players are predicted, not used to fit)."""
        if "fallback" in feats:
            feats = feats[~feats.fallback.astype(bool)]
        X = feats[self.cols].to_numpy(float)
        km = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed).fit(X)
        light = feats.groupby(km.labels_)["L"].median()          # lighter shirt first
        self._order = {old: new for new, old in enumerate(light.sort_values(ascending=False).index)}
        self.km = km
        self.labels_ = np.array([self._order[l] for l in km.labels_])
        self.silhouette = float(silhouette_score(X, self.labels_)) if len(set(self.labels_)) > 1 else float("nan")
        # display colour of each team = median BGR of its crops (an actual shirt colour)
        self.colours_bgr = {t: tuple(int(v) for v in feats.loc[self.labels_ == t, SPACES["bgr"]].median())
                            for t in range(self.k)}
        return self

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        raw = self.km.predict(feats[self.cols].to_numpy(float))
        return np.array([self._order[l] for l in raw])


# --------------------------------------------------------------------------- 4. one team per track
def vote_tracks(feats: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Team per track = the label most of its crops got. `agreement` = share of crops that agree."""
    t = feats[["tracker_id"]].assign(team=labels)
    rows = []
    for tid, g in t.groupby("tracker_id"):
        counts = g.team.value_counts()
        rows.append((int(tid), int(counts.index[0]), float(counts.iloc[0] / len(g)), len(g)))
    return pd.DataFrame(rows, columns=["tracker_id", "team", "agreement", "n_crops"])


# --------------------------------------------------------------------------- 5. goalkeepers
def assign_goalkeepers(df: pd.DataFrame, team_of: dict[int, int], gk_ids, class_col="voted_class_name") -> dict[int, int]:
    """A goalkeeper stands nearer his own team's half. For every frame a goalkeeper is visible, compare
    his x position with the average x of each team's outfield players in that frame, and vote.

    Approximate: positions are in PIXELS and the camera moves. Milestone 6 (pitch coordinates) makes it exact.
    """
    p = df[df.tracker_id.isin(team_of.keys())].copy()
    p["team"] = p.tracker_id.map(team_of)
    p["cx"] = (p.x1 + p.x2) / 2
    mean_x = p.groupby(["frame", "team"]).cx.mean().unstack()           # frame x team
    out = {}
    for gid in gk_ids:
        g = df[df.tracker_id == gid]
        votes = []
        for _, r in g.iterrows():
            if r.frame not in mean_x.index:
                continue
            m = mean_x.loc[r.frame].dropna()
            if len(m) < 2:
                continue
            votes.append(int((m - (r.x1 + r.x2) / 2).abs().idxmin()))
        out[int(gid)] = int(pd.Series(votes).mode()[0]) if votes else NO_TEAM
    return out


def add_team_column(df: pd.DataFrame, team_of: dict[int, int]) -> pd.DataFrame:
    """Adds `team` to the track table: 0/1 for players and goalkeepers, -1 for referees, ball and unknown."""
    out = df.copy()
    out["team"] = out.tracker_id.map(team_of).fillna(NO_TEAM).astype(int)
    return out


def team_balance(df_team: pd.DataFrame, class_col: str = "voted_class_name") -> pd.DataFrame:
    """Players per team per frame. Teams should be roughly balanced when the camera shows the whole pitch."""
    p = df_team[(df_team.team >= 0) & (df_team[class_col] == "player")]
    return p.groupby(["frame", "team"]).size().unstack(fill_value=0)


# --------------------------------------------------------------------------- drawing
REFEREE_BGR = (180, 105, 255)   # pink
BALL_BGR = (0, 220, 255)        # yellow


class TeamAnnotator:
    """Ellipse under the feet in the team's shirt colour, label '#id' (and 'GK'), referees in pink."""

    def __init__(self, team_colours: dict[int, tuple], team_names: dict[int, str] | None = None):
        self.colours = team_colours
        self.names = team_names or {t: f"team {t}" for t in team_colours}

    @staticmethod
    def _readable(col):
        b, g, r = col
        return (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 140 else (255, 255, 255)

    def __call__(self, frame: np.ndarray, rows: pd.DataFrame, class_col: str = "voted_class_name") -> np.ndarray:
        out = frame.copy()
        for r in rows.itertuples():
            cls = getattr(r, class_col)
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            cx, w = (x1 + x2) // 2, x2 - x1
            if cls == "ball":
                pts = np.array([[cx, y1 - 8], [cx - 7, y1 - 20], [cx + 7, y1 - 20]], np.int32)
                cv2.fillPoly(out, [pts], BALL_BGR)
                continue
            col = REFEREE_BGR if cls == "referee" or r.team < 0 else self.colours[r.team]
            cv2.ellipse(out, (cx, y2), (max(w // 2, 8), max(w // 6, 3)), 0, -45, 235, col, 3 if cls == "goalkeeper" else 2, cv2.LINE_AA)
            text = f"#{r.tracker_id}" + (" GK" if cls == "goalkeeper" else "") if cls != "referee" else f"REF #{r.tracker_id}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (cx - tw // 2 - 3, y2 + 6), (cx + tw // 2 + 3, y2 + 10 + th), col, -1)
            cv2.putText(out, text, (cx - tw // 2, y2 + 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self._readable(col), 1, cv2.LINE_AA)
        return out


def swatch(colours_bgr: dict[int, tuple], names: dict[int, str], size: int = 120) -> np.ndarray:
    """A small image with one coloured square per team and its name (to decide which team is which)."""
    img = np.full((size + 40, size * len(colours_bgr) + 10 * (len(colours_bgr) + 1), 3), 255, np.uint8)
    for i, t in enumerate(sorted(colours_bgr)):
        x = 10 + i * (size + 10)
        cv2.rectangle(img, (x, 10), (x + size, 10 + size), colours_bgr[t], -1)
        cv2.rectangle(img, (x, 10), (x + size, 10 + size), (60, 60, 60), 1)
        cv2.putText(img, names.get(t, str(t)), (x, size + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- 6. players mislabelled as referee
def check_referee_tracks(video_path, df: pd.DataFrame, tc: TeamClassifier, class_col: str = "voted_class_name",
                         per_track: int = 20) -> pd.DataFrame:
    """The class vote cannot fix a CONSISTENT detector mistake (a player called 'referee' in every frame).
    Colour can: compare each referee track's shirt with the two team colours and with the real referee.

    The longest referee track is taken as the reference referee (it runs through the clip). A 'referee'
    track whose shirt is closer to a team colour than to that referee is most likely a player of that team.
    Needs a TeamClassifier fitted in Lab space.
    """
    rows = sample_rows(df, class_col=class_col, classes=("referee",), per_track=per_track)
    feats, _ = collect_features(video_path, rows)
    if feats.empty:
        return pd.DataFrame()
    colour = feats.groupby("tracker_id")[SPACES["lab"]].median()
    length = df[df[class_col] == "referee"].groupby("tracker_id").size()
    main_ref = int(length.loc[colour.index].idxmax())
    ref_colour = colour.loc[main_ref].to_numpy()
    centres = np.empty_like(tc.km.cluster_centers_)
    for old, new in tc._order.items():
        centres[new] = tc.km.cluster_centers_[old]
    out = []
    for tid, c in colour.iterrows():
        d_ref = float(np.linalg.norm(c.to_numpy() - ref_colour))
        d_team = np.linalg.norm(centres - c.to_numpy(), axis=1)
        is_player = tid != main_ref and d_team.min() < d_ref
        out.append({"tracker_id": int(tid), "frames": int(length[tid]), "dist_to_referee": round(d_ref, 1),
                    "dist_team0": round(float(d_team[0]), 1), "dist_team1": round(float(d_team[1]), 1),
                    "verdict": "player" if is_player else "referee",
                    "team": int(d_team.argmin()) if is_player else NO_TEAM})
    return pd.DataFrame(out)


def apply_referee_check(df_team: pd.DataFrame, check: pd.DataFrame, class_col: str = "voted_class_name") -> pd.DataFrame:
    """Relabel the 'referee' tracks that the colour check says are players."""
    out = df_team.copy()
    for r in check[check.verdict == "player"].itertuples():
        m = out.tracker_id == r.tracker_id
        out.loc[m, class_col] = "player"
        out.loc[m, "team"] = r.team
    return out
