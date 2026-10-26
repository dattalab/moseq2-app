"""Cell 8: one picture of each recording.

    stat_type, n_bins, range_type = 'mean', 100, 'robust'
    preprocessor = MinMaxScaler()
    summary, range_dict = create_fingerprint_dataframe(moseq_df, stats_df, stat_type=stat_type,
                                                       n_bins=n_bins, range_type=range_type)
    plotting_fingerprint(summary, progress_paths['plot_path'], range_dict, preprocessor=preprocessor)

A fingerprint is one row per recording and, along it, the distribution of
four things the mouse did: how far from the centre it stayed, how fast it
moved, how tall it stood, how long it stretched. Beside those sits its
syllable usage. Stacked with the recordings grouped, it is the first look
at whether two conditions differ at all, before any test is run.

``range_type`` decides what the bins span. ``full`` runs from the
smallest value to the largest, so one bad frame, a tracking failure
reporting a mouse a metre long, stretches the axis and squeezes every
real value into the first bin. ``robust`` uses percentiles instead and
lets the outliers fall outside the bins, which is why its histograms
hold slightly less than all of a recording's frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moseq2_viz.model.fingerprint_classifier import (
    create_fingerprint_dataframe,
    plotting_fingerprint,
)

#: What the cell asks for.
SETTINGS = {"stat_type": "mean", "n_bins": 100, "range_type": "robust"}

#: The scalars the fingerprint is built from, and the syllable block that
#: sits beside them. These are ``create_fingerprint_dataframe``'s defaults,
#: which the cell takes.
SCALAR_BLOCKS = ("dist_to_center_px", "velocity_2d_mm", "height_ave_mm", "length_mm")
SYLLABLE_BLOCK = "MoSeq"

#: What ``plotting_fingerprint`` writes, in the directory it is handed.
FIGURES = ("moseq_fingerprint.png", "moseq_fingerprint.pdf")


@pytest.fixture(scope="module")
def fingerprint(moseq_df, stats_df):
    """Cell 8's first half: the binned distributions, before plotting."""
    return create_fingerprint_dataframe(moseq_df, stats_df, **SETTINGS)


@pytest.fixture(scope="module")
def plotted(analysis_paths, fingerprint):
    """Cell 8's second half: the figure, drawn where the cell says."""
    from sklearn.preprocessing import MinMaxScaler

    summary, range_dict = fingerprint
    plotting_fingerprint(
        summary, analysis_paths["plot_path"], range_dict, preprocessor=MinMaxScaler()
    )
    return Path(analysis_paths["plot_path"])


@pytest.fixture
def images(plotted):
    from PIL import Image as PILImage

    return {
        name: PILImage.open(plotted / name).convert("L")
        for name in FIGURES
        if name.endswith(".png")
    }


def test_there_is_one_fingerprint_per_recording(moseq_df, fingerprint):
    """1. A row is a recording, and it says which group it is in.

    The figure stacks the rows with the groups marked down the side, so
    the grouping has to travel with the data rather than being recovered
    at plotting time.
    """
    summary, _ = fingerprint
    assert list(summary.index.names) == ["group", "uuid"]
    assert len(summary) == moseq_df["uuid"].nunique()
    assert set(summary.index.get_level_values("uuid")) == set(moseq_df["uuid"])


def test_the_fingerprint_is_four_scalars_and_the_syllables(fingerprint):
    """2. What a column means.

    Five blocks side by side: a histogram for each of the four scalars,
    then one column per syllable. ``plotting_fingerprint`` draws them in
    that order under fixed headings, so a block that was missing or named
    differently would leave a panel of the figure empty.
    """
    summary, _ = fingerprint
    blocks = summary.columns.get_level_values(0).unique().tolist()
    assert set(blocks) == set(SCALAR_BLOCKS) | {SYLLABLE_BLOCK}


def test_n_bins_counts_the_edges_not_the_bars(fingerprint):
    """3. A hundred bins gives ninety-nine columns.

    ``n_bins`` is handed to ``linspace``, which returns that many
    boundaries and so that many minus one intervals between them. It is
    the resolution of the histogram either way, but a reader counting
    columns should know which of the two numbers they are looking at.
    """
    summary, _ = fingerprint
    for block in SCALAR_BLOCKS:
        assert summary[block].shape[1] == SETTINGS["n_bins"] - 1, block


def test_the_syllable_block_is_the_usage_cell_6_computed(stats_df, fingerprint):
    """4. The right half of the figure is cell 6's table, pivoted.

    Nothing is recomputed here. If it were, the fingerprint could
    disagree with the statistics every later cell tests, and the figure
    would be describing a different experiment from the p-values.
    """
    summary, _ = fingerprint
    syllables = summary[SYLLABLE_BLOCK]

    expected = stats_df.pivot_table(index=["group", "uuid"], values="usage", columns="syllable")
    expected = expected.reindex(index=syllables.index, columns=syllables.columns)
    np.testing.assert_allclose(syllables.to_numpy(), expected.to_numpy(), rtol=1e-9)

    np.testing.assert_allclose(syllables.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_robust_ranges_leave_the_outliers_outside_the_bins(moseq_df, stats_df, fingerprint):
    """5. The difference between the two range types, in the data.

    With the full range the bins span every value in the experiment, so a
    recording's histogram accounts for all of it bar the handful of frames
    where extraction found no mouse and recorded nothing. With robust
    ranges the extremes sit beyond the first and last bin and go
    uncounted, which is the point: a single mistracked frame no longer
    sets the axis for everything else.
    """
    robust, _ = fingerprint
    full, _ = create_fingerprint_dataframe(moseq_df, stats_df, **{**SETTINGS, "range_type": "full"})

    for block in SCALAR_BLOCKS:
        full_mass = full[block].sum(axis=1).to_numpy()
        robust_mass = robust[block].sum(axis=1).to_numpy()
        np.testing.assert_allclose(full_mass, 1.0, atol=0.01, err_msg=block)
        assert (robust_mass < full_mass).all(), f"{block} lost nothing to the tails"
        assert (robust_mass > 0.5).all(), f"{block} lost most of its frames"


def test_the_ranges_the_axes_are_labelled_from_come_back(fingerprint):
    """6. The bounds a reader needs to interpret a bin.

    A fingerprint row is a histogram with no axis of its own; the units
    live in these bounds, which the plot prints under each block. Without
    them a column is a position in an unlabelled strip.
    """
    _, range_dict = fingerprint
    assert set(range_dict.columns) == set(SCALAR_BLOCKS) | {SYLLABLE_BLOCK}
    assert len(range_dict) == 2, "a lower and an upper bound per block"

    low, high = range_dict.to_numpy()
    assert (low < high).all(), f"a block's range is empty: {range_dict}"


def test_the_figure_is_written_where_the_cell_points_it(plotted):
    """7. In the model's own plot directory, as png and as pdf.

    Cell 2 derived ``plot_path`` from the chosen model's folder, so two
    models analysed in one project keep their figures apart. The
    directory does not exist until something writes to it.
    """
    for name in FIGURES:
        path = plotted / name
        assert path.exists(), f"{name} was not written"
        assert path.stat().st_size > 0, f"{name} is empty"


def test_the_figure_is_large_enough_to_read(images):
    """8. Twenty stacked recordings across five blocks needs the room."""
    for name, image in images.items():
        width, height = image.size
        assert width >= 400 and height >= 300, f"{name} is {width}x{height}"


def test_the_figure_has_something_drawn_on_it(images):
    """9. The failure this catches is a blank canvas.

    A plotting call that silently drew nothing still writes a valid png,
    and the cell still displays it. A user sees white and cannot tell
    whether that is the result or the bug.
    """
    for name, image in images.items():
        pixels = np.asarray(image)
        ink = (pixels < 250).mean()
        assert ink > 0.01, f"{name} is {1 - ink:.1%} blank"
        assert pixels.std() > 5, f"{name} has no contrast"


def test_the_heatmap_spans_the_width_of_the_figure(images):
    """10. Five blocks side by side, not one and four empty panels.

    The blocks are laid out across the figure, so ink has to reach both
    edges. A block that failed to draw leaves a white column, which this
    would catch as a gap in coverage.
    """
    for name, image in images.items():
        pixels = np.asarray(image)
        columns_with_ink = (pixels < 250).any(axis=0)
        assert columns_with_ink.mean() > 0.5, f"{name} leaves half its width blank"
