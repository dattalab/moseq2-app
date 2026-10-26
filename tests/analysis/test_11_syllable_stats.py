"""Cell 11: compare the groups, syllable by syllable.

    from moseq2_app.main import interactive_syllable_stats
    interactive_syllable_stats(progress_paths, max_syllable=max_syllables, load_parquet=True)

The cell the experiment is read out of. It plots a statistic for every
syllable, sorted however a user asks, with the groups side by side and
error bars, and marks the syllables on which the two groups differ. It is
the only cell in either notebook that performs inference.

Four tests are offered. Kruskal-Wallis with Dunn's post-hoc compares all
groups at once and then every pair; the Z-test, t-test and Mann-Whitney
compare the two groups the dropdowns name. All four correct for testing
sixty syllables at once, which is the difference between a result and a
list of coincidences.

Which syllables come back significant is pinned rather than merely
bounded. The alternative is checking that p-values lie in [0, 1], which
no change to the arithmetic would ever violate. A pinned set reddens when
SciPy changes a default or a tie rule, which is the thing worth hearing
about, and the baseline is re-recorded deliberately.
"""

from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import pytest

from moseq2_app.main import interactive_syllable_stats
from moseq2_app.stat.controller import InteractiveSyllableStats

#: What the cell asks for. ``load_parquet`` tells the widget it may reuse
#: a cached dataframe rather than rebuilding it from the recordings.
VIEWER = {"load_parquet": True}

#: Every hypothesis test the widget offers, by the name on the dropdown.
TESTS = ("KW & Dunn's", "Z-Test", "T-Test", "Mann-Whitney")

#: The statistic the tests are run on by default, and the one a group
#: difference in behaviour would show up in first.
STATISTIC = "usage"


@pytest.fixture(scope="module")
def viewer(analysis_paths, labeled):
    """The widget behind the cell, built once."""
    return InteractiveSyllableStats(
        index_path=analysis_paths["index_file"],
        model_path=analysis_paths["model_path"],
        df_path=analysis_paths["df_info_path"],
        info_path=analysis_paths["syll_info"],
        max_sylls=labeled,
        load_parquet=VIEWER["load_parquet"],
    )


@pytest.fixture(scope="module")
def significant(viewer):
    """The syllables each test marks, on the statistic the cell opens on."""
    return {
        name: sorted(
            int(syllable)
            for syllable in viewer.run_selected_hypothesis_test(
                name, STATISTIC, viewer.ctrl_dropdown.value, viewer.exp_dropdown.value
            )
        )
        for name in TESTS
    }


def test_the_cell_displays_the_controls_and_the_plot(analysis_paths, labeled, displayed):
    """1. What a user is given: a reset, the controls, and the figure."""
    assert interactive_syllable_stats(analysis_paths, max_syllable=labeled, **VIEWER) is None

    assert [type(obj).__name__ for obj in displayed.objects] == ["Button", "VBox", "Output"]
    assert displayed.one(widgets.Button).description == "Clear Output"


def test_the_controls_offer_every_choice_the_cell_is_for(viewer):
    """2. The reason this is a widget rather than a figure.

    A user works the experiment by changing these: which statistic to
    plot, what to sort by, whether to pool into groups or show sessions,
    what the error bars mean, and which test to mark significance with.
    """
    assert "usage" in viewer.stat_dropdown.options
    assert "duration" in viewer.stat_dropdown.options
    assert "Difference" in viewer.sorting_dropdown.options
    assert list(viewer.grouping_dropdown.options) == ["group", "SessionName", "SubjectName"]
    assert set(viewer.errorbar_dropdown.options) == {"CI 95%", "None", "SEM", "STD"}
    assert tuple(viewer.hyp_test_dropdown.options) == TESTS


def test_both_groups_are_chosen_before_a_user_touches_anything(viewer, moseq_df):
    """3. A comparison needs two groups named.

    The group a test treats as control and the one it treats as
    experimental are read straight off these dropdowns and passed to the
    hypothesis tests, which have no group under ``None``. Newer
    ipywidgets leaves a dropdown unselected when its options are
    assigned, so both are set explicitly.
    """
    groups = set(moseq_df["group"])
    assert set(viewer.ctrl_dropdown.options) == groups
    assert set(viewer.exp_dropdown.options) == groups

    assert viewer.ctrl_dropdown.value in groups
    assert viewer.exp_dropdown.value in groups
    assert viewer.ctrl_dropdown.value != viewer.exp_dropdown.value, (
        "the cell opens comparing a group against itself"
    )


def test_the_plot_covers_the_syllables_cell_9_settled_on(viewer, labeled):
    """4. The same cut-off as the dendrogram and the transition graphs.

    Everything past cell 9 is limited to the syllables that cover the
    requested share of behaviour. A viewer plotting more would be showing
    syllables the rest of the analysis ignores.
    """
    assert viewer.df["syllable"].max() < labeled
    assert viewer.max_sylls == labeled


def test_every_session_is_available_to_plot_on_its_own(viewer, moseq_df):
    """5. Switching off the group average.

    A difference carried by one animal and a difference shared by five
    look the same once pooled. The grouping dropdown exists to take that
    apart, which needs every session reachable by name.
    """
    assert len(viewer.session_names) == moseq_df["uuid"].nunique()
    assert set(viewer.session_sel.options) == set(viewer.session_names)
    assert viewer.session_sel.value, "no session is selected to plot"


@pytest.mark.parametrize("test", TESTS)
def test_each_test_marks_syllables_that_exist(test, significant, labeled):
    """6. What the figure puts a mark over.

    A significant syllable is drawn with an asterisk on the plot, so the
    numbers coming back have to be syllables the plot is showing.
    """
    marked = significant[test]
    assert all(0 <= syllable < labeled for syllable in marked), marked
    assert len(set(marked)) == len(marked), f"{test} reported a syllable twice"


def test_which_syllables_differ_between_the_groups_is_unchanged(significant, data_regression):
    """7. The result the notebook exists to produce.

    Pinned rather than bounded. These are the syllables on which an
    amphetamine animal's behaviour differs from a saline animal's, under
    each of the four tests, after correction for testing every syllable.
    A change here means the arithmetic under the inference moved, which
    is worth looking at whether or not it turns out to be benign.
    """
    data_regression.check(significant)


def test_an_unknown_test_name_is_refused(viewer):
    """8. The dropdown is the only legal source of a test name.

    The name is looked up in a mapping shared with the sorting and
    statistic dropdowns, so a name from one of those reaches the lookup
    and has to fail rather than silently selecting a test.
    """
    with pytest.raises(KeyError):
        viewer.run_selected_hypothesis_test(
            "Student's", STATISTIC, viewer.ctrl_dropdown.value, viewer.exp_dropdown.value
        )


@pytest.mark.xfail(
    reason="the cell asks for the cached dataframe by setting load_parquet=True, "
    "and the progress file names it, but the two lines that would write it in "
    "SyllableLabeler.get_mean_syllable_info are commented out. Every run of "
    "cells 11 and 14 rebuilds it from the recordings instead."
)
def test_the_cached_dataframe_the_cell_asks_for_exists(analysis_paths, labeled):
    """9. What ``load_parquet=True`` is supposed to save.

    Building the syllable dataframe means reading every recording's
    scalars and joining them against the model's labels. Cells 11 and 14
    each do it, and cell 9 was to have written the result once under
    ``df_info_path`` for them to read back.
    """
    assert Path(analysis_paths["df_info_path"]).exists()
