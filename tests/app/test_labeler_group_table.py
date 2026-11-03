"""The table under a crowd movie: each group's mean scalars for that syllable.

The labeler and the comparison tool build it from a frame with one row per
group and syllable. A group that never used a syllable has no row for it, and
the rows it does have must still land under their own syllable numbers.
"""

from __future__ import annotations

import pandas as pd

from moseq2_app.viz.controller import _group_syll_info

SCALARS = ["usage", "duration", "velocity_2d_mm_mean", "velocity_3d_mm_mean",
           "height_ave_mm_mean", "dist_to_center_px_mean"]


def group_frame(rows):
    """``(group, syllable, value)`` rows, with every scalar set to ``value``."""
    return pd.DataFrame([{"group": g, "syllable": s, **{c: v for c in SCALARS}} for g, s, v in rows])


def test_a_group_missing_a_syllable_shows_each_syllable_its_own_numbers():
    df = group_frame([
        ("a", 0, 10.0), ("a", 1, 11.0), ("a", 2, 12.0),
        ("b", 0, 20.0), ("b", 2, 22.0),
    ])
    info = _group_syll_info({}, ["a", "b"], df, max_sylls=3)

    usage = {s: {g: v["usage"] for g, v in info[s]["group_info"].items()} for s in range(3)}
    assert usage == {0: {"a": 10.0, "b": 20.0}, 1: {"a": 11.0}, 2: {"a": 12.0, "b": 22.0}}
