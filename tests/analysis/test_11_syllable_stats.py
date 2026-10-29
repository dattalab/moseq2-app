"""Cell 11: compare the groups, syllable by syllable.

    from moseq2_app.main import interactive_syllable_stats
    interactive_syllable_stats(progress_paths, max_syllable=max_syllables)

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

from conftest import LABELLING

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
        info_path=analysis_paths["syll_info"],
        max_sylls=labeled,
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
    assert interactive_syllable_stats(analysis_paths, max_syllable=labeled) is None

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


@pytest.mark.parametrize("cell", [9, 11, 14])
def test_a_dataframe_left_by_an_older_version_is_not_read(analysis_paths, labeled, displayed, cell):
    """9. The syllable table is built from the recordings every time.

    Releases before 2023 saved it as ``syll_df.parquet`` in the model's
    folder, and their progress files name it as ``df_info_path``. A file
    left there holds an older version's columns and whichever model was
    current then, so no cell reads it, and ``load_parquet``, which older
    copies of the notebook still pass, changes nothing.
    """
    from moseq2_app.main import interactive_transition_graph, label_syllables

    leftover = Path(analysis_paths["model_session_path"]) / "syll_df.parquet"
    leftover.write_bytes(b"written by an older moseq2-app")
    paths = {**analysis_paths, "df_info_path": str(leftover)}
    try:
        if cell == 9:
            label_syllables(paths, **LABELLING)
        elif cell == 11:
            interactive_syllable_stats(paths, max_syllable=labeled, load_parquet=True)
        else:
            interactive_transition_graph(
                paths, max_syllables=labeled, plot_vertically=True, load_parquet=True
            )
    finally:
        leftover.unlink(missing_ok=True)


#: A scalar some sessions have no value for: a session that never used a
#: syllable has no duration for it.
SCALAR = "duration"


def test_a_scalar_of_an_unused_syllable_stays_missing(viewer):
    """10. Never used is not measured at zero.

    Usage of a syllable a session never used is 0; its duration was never
    measured, so it stays missing and each syllable's test runs on the
    sessions that used it.
    """
    unused = viewer.df["usage"] == 0
    assert unused.any(), "every session used every syllable; nothing to check"
    assert viewer.df.loc[unused, SCALAR].isna().all()
    assert (viewer.df.loc[unused, "usage"] == 0).all()


@pytest.mark.parametrize("test", TESTS)
def test_each_test_runs_on_a_scalar_with_unused_syllables(viewer, labeled, test):
    marked = viewer.run_selected_hypothesis_test(
        test, SCALAR, viewer.ctrl_dropdown.value, viewer.exp_dropdown.value
    )
    assert all(0 <= syllable < labeled for syllable in marked)


@pytest.mark.parametrize("sort", [SCALAR, "difference"])
def test_the_plot_builds_for_a_scalar_with_unused_syllables(viewer, sort):
    viewer.interactive_syll_stats_grapher(
        SCALAR, sort, "group", "SEM", viewer.session_sel.value,
        viewer.ctrl_dropdown.value, viewer.exp_dropdown.value, hyp_test="Mann-Whitney",
    )
    assert viewer.stat_fig is not None
