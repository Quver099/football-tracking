"""Assign teams to the tracks from Milestone 3 and render a video.

Usage (from repo root):
    python scripts/run_teams.py --video data/raw/08fd33_0.mp4 --tracks outputs/m3_tracks.csv \
        --names "White,Blue" --out outputs/m4_teams.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_tracking.teams import (  # noqa: E402
    TeamAnnotator, TeamClassifier, add_team_column, assign_goalkeepers, collect_features, sample_rows, vote_tracks,
)
from football_tracking.video_io import VideoWriter, iter_frames, video_info  # noqa: E402

CLS = "voted_class_name"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", default="outputs/m3_tracks.csv")
    ap.add_argument("--out", default="outputs/m4_teams.mp4")
    ap.add_argument("--csv", default="outputs/m4_tracks_teams.csv")
    ap.add_argument("--space", default="lab", choices=["lab", "bgr", "hsv"])
    ap.add_argument("--names", default="team 0,team 1", help="names for team 0 (lighter shirt) and team 1")
    a = ap.parse_args()

    df = pd.read_csv(a.tracks)
    info = video_info(a.video)
    rows = sample_rows(df, class_col=CLS, classes=("player",))
    feats, _ = collect_features(a.video, rows)
    tc = TeamClassifier(a.space).fit(feats)
    votes = vote_tracks(feats, tc.predict(feats))
    team_of = dict(zip(votes.tracker_id, votes.team))
    gk = df[(df.tracker_id >= 0) & (df[CLS] == "goalkeeper")].tracker_id.unique()
    team_of.update(assign_goalkeepers(df, team_of, gk, class_col=CLS))
    dft = add_team_column(df, team_of)
    print(f"silhouette {tc.silhouette:.3f} | tracks per team {votes.team.value_counts().sort_index().to_dict()} "
          f"| median agreement {votes.agreement.median():.2f} | low-agreement tracks {(votes.agreement < 0.7).sum()}")

    names = dict(enumerate(n.strip() for n in a.names.split(",")))
    ann = TeamAnnotator(tc.colours_bgr, names)
    by_frame = {f: g for f, g in dft.groupby("frame")}
    with VideoWriter(a.out, info["fps"], info["width"], info["height"]) as w:
        for i, frame in iter_frames(a.video):
            g = by_frame.get(i)
            w.write(ann(frame, g, class_col=CLS) if g is not None else frame)
    dft.to_csv(a.csv, index=False)
    print(f"Saved -> {a.out} and {a.csv}")


if __name__ == "__main__":
    main()
