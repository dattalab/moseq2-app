"""The unit every velocity is shown in.

Extraction's ``velocity_2d_mm`` and ``velocity_3d_mm`` are how far the
mouse's centroid moved since the previous frame, in millimetres: a
distance per frame, not per second. Nothing between extraction and the
analysis notebook rescales them, so a label beside one of these numbers
has to say mm/frame. The one place the pipeline shows a speed per second
is the fingerprint, which multiplies by 30 and says so in its heading.

moseq2-app#215 is a histogram of these values read as mm/s: a mode near
1 and a tail to 20, which is about 30 mm/s at the mode.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import moseq2_app
import moseq2_extract
import moseq2_viz
from moseq2_extract.util import scalar_attributes

#: Every package whose text a user reads.
PACKAGES = (moseq2_app, moseq2_extract, moseq2_viz)

#: A unit per second written after a velocity or speed.
PER_SECOND = re.compile(r"(?i)(velocity|speed)[^\n'\"]{0,40}\b[cm]m ?/ ?s(ec)?\b")

#: The files allowed to call a velocity mm/s, with what makes it true.
#: The fingerprint multiplies the frames' velocity by 30 before binning,
#: and tests/analysis/test_08_fingerprint.py checks its ranges are the
#: per-frame percentiles times 30.
PER_SECOND_ALLOWED = {"fingerprint_classifier.py"}


def test_velocity_is_the_centroid_displacement_between_consecutive_frames(moseq_df):
    """The number itself: millimetres moved since the previous frame.

    Recomputed from the centroid positions, session by session, the stored
    ``velocity_2d_mm`` is the distance between one frame's centroid and the
    next one's. No division by the time between them.

    The exception is the first frame of each block extraction processes at
    once (``chunk_size``, 3000 frames by default, which the bundle was
    extracted with): ``compute_scalars`` sees one block at a time, so that
    frame has no previous one and its velocity is 0.
    """
    chunk_size = 3000
    checked = 0
    for uuid, session in moseq_df.groupby("uuid", sort=False):
        dx = np.diff(session["centroid_x_mm"].to_numpy())
        dy = np.diff(session["centroid_y_mm"].to_numpy())
        displacement = np.hypot(dx, dy)
        stored = session["velocity_2d_mm"].to_numpy()[1:]
        block_start = session["frame index"].to_numpy()[1:] % chunk_size == 0
        compared = np.isfinite(displacement) & np.isfinite(stored) & ~block_start
        assert compared.sum() > 100, f"session {uuid} has {compared.sum()} frames to compare"
        np.testing.assert_allclose(stored[compared], displacement[compared], rtol=1e-4, atol=1e-4,
                                   err_msg=f"session {uuid}")
        assert (stored[block_start & np.isfinite(stored)] == 0).all(), f"session {uuid}"
        checked += 1
    assert checked > 0


def test_extraction_describes_velocity_per_frame():
    """The descriptions extraction stores beside each scalar."""
    attributes = scalar_attributes()
    for key in ("velocity_2d_mm", "velocity_3d_mm"):
        assert "(mm / frame)" in attributes[key], attributes[key]
    for key in ("velocity_2d_px", "velocity_3d_px"):
        assert "(pixels / frame)" in attributes[key], attributes[key]


def test_no_velocity_label_says_per_second_unless_it_rescales():
    """Every label that says mm/s or cm/s for a velocity is one that multiplies by a frame rate."""
    found = []
    for package in PACKAGES:
        for path in sorted(Path(package.__file__).parent.rglob("*.py")):
            if path.name in PER_SECOND_ALLOWED:
                continue
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if PER_SECOND.search(line):
                    found.append(f"{path}:{number}: {line.strip()}")
    assert found == []


def table_rows(div):
    """The row labels and values of the HTML table ``DataFrame.to_html`` wrote into a div."""
    rows = re.findall(r"<tr>\s*<th>(.*?)</th>\s*<td>(.*?)</td>\s*</tr>", div, flags=re.S)
    return pd.Series({label: value for label, value in rows})


@pytest.fixture(scope="module")
def comparison(analysis_paths, sorted_index, monkeypatch_module, tmp_path_factory):
    """The crowd-movie comparison, built on the analysis bundle, with its movies stubbed.

    Its syllable tables come from the real ``get_session_mean_syllable_info_df``;
    only the crowd movies (slow to render, and not what is tested here) are
    replaced by an empty file of known size.
    """
    import moseq2_viz.steps as viz_steps
    from moseq2_app.viz import controller
    from moseq2_app.viz.controller import CrowdMovieComparison

    movie = tmp_path_factory.mktemp("crowd_movies") / "movie.mp4"
    movie.write_bytes(b"")

    obj = CrowdMovieComparison.__new__(CrowdMovieComparison)
    obj.sorted_index = sorted_index
    obj.model_path = analysis_paths["model_path"]
    obj.index_path = analysis_paths["index_file"]
    obj.output_dir = str(movie.parent)
    obj.cm_choices = {'separate_by': 'groups'}
    obj.get_pdfs = False
    obj.max_sylls = 5
    obj.syll_info = {i: {"label": f"s{i}", "desc": "", "crowd_movie_path": "", "group_info": {}}
                     for i in range(obj.max_sylls)}
    obj.session_dict = {i: {"session_info": {}} for i in range(obj.max_sylls)}
    obj.get_session_mean_syllable_info_df()

    monkeypatch_module.setattr(viz_steps, "make_crowd_movies_wrapper",
                               lambda *args, **kwargs: {g: [str(movie)] for g in obj.groups})
    monkeypatch_module.setattr(controller, "get_video_info", lambda path: {"dims": (100, 100)})
    return obj


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def test_crowd_movie_comparison_shows_each_groups_velocity_in_mm_per_frame(comparison):
    """The table beside each group's crowd movie.

    The value is the group's mean over sessions of each session's mean
    per-frame velocity during the syllable, unscaled, and its row says
    mm/frame.
    """
    import ipywidgets as widgets

    syllable = 0
    comparison.cm_sources_dropdown = widgets.Dropdown(options=["group"], value="group")
    comparison.cm_syll_select = widgets.Dropdown(options=[f"{syllable} - s{syllable}"])
    grouped = comparison.group_syll_info[syllable]["group_info"]
    for group in grouped:
        grouped[group]["pdf"] = None

    divs, _ = comparison.generate_crowd_movie_divs(grouped)

    expected = comparison.group_df[comparison.group_df["syllable"] == syllable].set_index("group")
    assert len(divs) == len(comparison.groups) > 0
    for group, div in zip(comparison.groups, divs):
        table = table_rows(div)
        for label, column in (("2D velocity (mm/frame)", "velocity_2d_mm_mean"),
                              ("3D velocity (mm/frame)", "velocity_3d_mm_mean")):
            assert label in table.index, f"{group}: rows are {list(table.index)}"
            assert float(table[label]) == pytest.approx(expected.loc[group, column], rel=1e-5)
        assert not any("mm/s" in row for row in table.index), list(table.index)
