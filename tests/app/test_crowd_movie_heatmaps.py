"""Position heatmaps next to each crowd movie in the analysis notebook's comparison.

For the syllable picked in the dropdown, each group (or session) is shown
with its crowd movie and the heatmap of where in the arena it performed that
syllable. The heatmap is found by syllable number, so a group that never used
an earlier syllable still gets its own heatmap for every later one, and a
group that never used the picked syllable has no heatmap to show.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from moseq2_app.viz.controller import _syllable_pdf
from moseq2_viz.scalars.util import get_syllable_pdfs


def marked(syllable):
    """A heatmap whose hot pixel's position says which syllable it belongs to."""
    pdf = np.zeros((3, 3))
    pdf.flat[syllable] = 1.0
    return pdf


def group_pdfs():
    """Group b never used syllable 1. The table is built the way the comparison builds it."""
    rows = [
        {"group": group, "uuid": group, "SessionName": group, "SubjectName": group, "syllable": s, "pdf": marked(s)}
        for group, syllables in (("a", [0, 1, 2]), ("b", [0, 2]))
        for s in syllables
    ]
    pdfs, _ = get_syllable_pdfs(pd.DataFrame(rows), groupby="group", syllable_key="syllable")
    return pdfs.drop("syllable", axis=1).reset_index()


def test_each_group_gets_the_heatmap_of_the_syllable_picked():
    pdfs = group_pdfs()
    for group, syllables in (("a", [0, 1, 2]), ("b", [0, 2])):
        for syllable in syllables:
            assert int(np.argmax(_syllable_pdf(pdfs, "group", group, syllable))) == syllable


def test_a_group_that_never_used_the_syllable_has_no_heatmap():
    assert _syllable_pdf(group_pdfs(), "group", "b", 1) is None
