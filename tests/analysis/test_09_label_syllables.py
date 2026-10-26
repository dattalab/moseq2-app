"""Cell 9: make the syllables watchable, and label them.

    crowd_dir = join(progress_paths['model_session_path'], 'crowd_movies/')
    syll_infopath = join(progress_paths['model_session_path'], 'syll_info.yaml')
    explained_variance, max_syllables, max_examples = 99, None, 20
    max_syllables = label_syllables(progress_paths, max_syllables=max_syllables,
                                    n_explained=explained_variance,
                                    select_median_duration_instances=select_median_duration_instances,
                                    max_examples=max_examples)

Everything before this cell describes syllables in numbers. This is where
they become something to look at. For each syllable the cell cuts twenty
instances out of the recordings, plays them together in one frame, and
offers the result beside a box to type what the mouse is doing: rearing,
grooming, a left turn. Those names are what the rest of the analysis is
reported in.

It also settles the one number the remaining cells all take. A model fit
with a hundred states does not use a hundred: usage falls away sharply,
and the tail is syllables that occur a handful of times in twelve hours.
``n_explained=99`` keeps as many as cover ninety-nine percent of frames
and discards the rest, and cells 10, 11, 12 and 14 are all cut off there.

Generating the movies is the longest step in the notebook. A second run
of the cell finds them on disk and reuses them.
"""

from __future__ import annotations

import re
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pytest

from moseq2_app.main import label_syllables
from moseq2_viz.model.util import parse_model_results
from moseq2_viz.util import read_yaml

from tests.conftest import LABELLING
from tests.video import index_is_at_the_front, probe

#: Crowd movies are named after both numberings: where the syllable ranks
#: by usage, and what the model called it.
MOVIE_NAME = re.compile(
    r"syllable_sorted-id-(?P<sorted_id>\d+)_\((?P<sort_type>\w+)\)_original-id-(?P<original_id>\d+)"
)

#: The figure ``compute_syllable_explained_variance`` writes so a user can
#: see where the cut-off landed.
FIGURE = "syllable_needed_explained_variance"


@pytest.fixture(scope="module")
def syll_info(analysis_paths, labeled):
    """The label file the cell writes and the labeller reads back."""
    return read_yaml(analysis_paths["syll_info"])


@pytest.fixture(scope="module")
def movies(analysis_paths, labeled):
    """Every crowd movie the cell generated."""
    return sorted(Path(analysis_paths["crowd_dir"]).glob("*.mp4"))


def test_the_cell_settles_on_fewer_syllables_than_the_model_has(analysis_paths, labeled):
    """1. The number every later cell is cut off at.

    The model was fit with far more states than a mouse uses. Keeping the
    ones that cover ninety-nine percent of frames is what stops the
    dendrogram, the transition matrices and the statistics widget from
    being dominated by syllables seen a handful of times.
    """
    model = parse_model_results(analysis_paths["model_path"])
    used = len({int(label) for labels in model["labels"] for label in np.unique(labels) if label >= 0})

    assert isinstance(labeled, (int, np.integer)), f"the cell returned {labeled!r}"
    assert 0 < labeled <= used, f"{labeled} syllables kept out of {used} used"


def syllable_coverage(model_path):
    """Cumulative share of syllable bouts, syllable by syllable, best used first.

    This is the curve the cut-off is read off. ``count='usage'`` counts how
    often each syllable was entered, so it is a share of bouts rather than
    of frames.
    """
    from moseq2_viz.model.util import get_syllable_usages, relabel_by_usage

    model = parse_model_results(model_path)
    labels = relabel_by_usage(model["labels"], count="usage")[0]
    usages = np.array(list(get_syllable_usages(labels, count="usage").values()))
    return 100 * np.cumsum(usages / usages.sum()) / np.max(np.cumsum(usages / usages.sum()))


def test_the_number_is_where_the_curve_crosses_the_threshold(analysis_paths, labeled):
    """2. What the returned number is, exactly.

    It is the position on the cumulative usage curve at which the
    requested share is first reached, counting from zero: everything
    below it falls short and it is the first that does not.
    """
    coverage = syllable_coverage(analysis_paths["model_path"])

    assert coverage[labeled] >= LABELLING["n_explained"]
    assert coverage[labeled - 1] < LABELLING["n_explained"]


@pytest.mark.xfail(
    reason="the number returned is the index at which the curve crosses the "
    "threshold, and every later cell uses it as a count. Keeping that many "
    "syllables keeps the ones below the crossing, which by construction fall "
    "short of the requested share; one more would be needed to reach it."
)
def test_the_syllables_actually_kept_cover_what_was_asked_for(analysis_paths, labeled):
    """3. What a user believes they set.

    ``explained_variance = 99`` reads as an instruction to keep enough
    syllables to account for ninety-nine percent of the behaviour. Cells
    10, 11, 12 and 14 then keep syllables numbered below the returned
    value, so the share they actually cover is the one at the position
    before the crossing.
    """
    coverage = syllable_coverage(analysis_paths["model_path"])
    kept = coverage[labeled - 1]

    assert kept >= LABELLING["n_explained"], (
        f"analysing {labeled} syllables covers {kept:.2f}% of bouts, "
        f"not {LABELLING['n_explained']}%"
    )


def test_the_curve_the_cut_off_was_read_off_is_drawn(analysis_paths, labeled):
    """4. A user checks the number against this figure.

    Cumulative explained variance against syllable count, with a line
    where the cell cut. Written as png and pdf into the model's plot
    directory.
    """
    plots = Path(analysis_paths["plot_path"])
    for extension in ("png", "pdf"):
        path = plots / f"{FIGURE}.{extension}"
        assert path.exists(), f"{path.name} was not written"
        assert path.stat().st_size > 0, f"{path.name} is empty"

    from PIL import Image as PILImage

    image = PILImage.open(plots / f"{FIGURE}.png").convert("L")
    width, height = image.size
    assert width >= 400 and height >= 300, f"the figure is {width}x{height}"

    pixels = np.asarray(image)
    assert (pixels < 250).mean() > 0.01, "the figure is blank"
    columns_with_ink = (pixels < 250).any(axis=0)
    assert columns_with_ink.mean() > 0.3, "the curve does not span the plot"


def test_there_is_a_movie_for_every_syllable_being_labelled(movies, labeled):
    """5. A user steps through the dropdown and expects a movie each time.

    The labeller indexes the movies by the syllable's position in the
    usage ranking, so a gap is not a missing file at the end: it is one
    syllable showing another's movie.
    """
    assert len(movies) >= labeled, f"{len(movies)} movies for {labeled} syllables"

    sorted_ids = set()
    for movie in movies:
        match = MOVIE_NAME.search(movie.name)
        assert match, f"{movie.name} does not name the syllable it shows"
        sorted_ids.add(int(match.group("sorted_id")))

    assert set(range(labeled)) <= sorted_ids, (
        f"no movie for syllables {sorted(set(range(labeled)) - sorted_ids)}"
    )


def test_each_movie_names_both_numberings_of_its_syllable(movies):
    """6. Which syllable a movie shows, under either numbering.

    The notebook speaks in usage rank, the model file in its own
    arbitrary ids, and the two have to be matched up when a result is
    traced back to the model. The filename carries both and says which
    sort produced the rank.
    """
    for movie in movies:
        match = MOVIE_NAME.search(movie.name)
        assert match, movie.name
        assert match.group("sort_type") == "usage", movie.name


def test_the_movies_play_in_a_browser(movies):
    """7. The labeller shows these inside the notebook.

    H.264 with 4:2:0 chroma is what a browser will decode. Anything else
    leaves a user with a player that reports no error and shows nothing.
    """
    assert movies, "no crowd movies to check"
    for movie in movies:
        codec, pix_fmt = probe(movie)
        assert (codec, pix_fmt) == ("h264", "yuv420p"), movie.name


@pytest.mark.xfail(
    reason="crowd movies are written without -movflags +faststart, so the moov "
    "atom is appended after the frame data. Same defect as the extraction "
    "preview movie in the modeling notebook."
)
def test_the_movies_start_without_reading_all_of_them(movies):
    """8. The index belongs at the front.

    The labeller plays these one after another as a user moves through
    the dropdown. With the index appended, each has to be fetched whole
    before its first frame appears.
    """
    for movie in movies:
        assert index_is_at_the_front(movie), movie.name


def test_the_label_file_has_an_empty_entry_per_syllable(syll_info, labeled):
    """9. What a user is being asked to fill in.

    One entry per syllable, each with a blank name and description and a
    pointer to the movie that shows it. The blanks are the work the cell
    exists for; the movie path is how the labeller finds what to play.
    """
    assert len(syll_info) == labeled

    for syllable in range(labeled):
        entry = syll_info[syllable]
        assert entry["label"] == "", f"syllable {syllable} arrives pre-labelled"
        assert entry["desc"] == "", f"syllable {syllable} arrives pre-described"

        movie = Path(entry["crowd_movie_path"])
        assert movie.exists(), f"syllable {syllable} points at {movie}, which is not there"
        assert int(MOVIE_NAME.search(movie.name).group("sorted_id")) == syllable


def test_the_cell_offers_a_syllable_at_a_time_with_a_way_to_start_over(
    analysis_paths, labeled, displayed
):
    """10. What a user is given.

    Two things are drawn. The labelling view for the syllable currently
    chosen, which is the movie beside the boxes for its name and
    description, with previous, set and next beneath it. And the controls
    that outlive it: a dropdown holding every syllable, a button to clear
    the output, and the area the view is redrawn into.
    """
    assert label_syllables(analysis_paths, **LABELLING) == labeled

    shown = [type(obj).__name__ for obj in displayed.objects]
    assert shown[-3:] == ["Button", "Dropdown", "Output"], shown
    assert "AppLayout" in shown, f"the labelling view was not drawn: {shown}"

    assert displayed.one(widgets.Button).description == "Clear Output"
    assert len(displayed.one(widgets.Dropdown).options) == labeled

    stepper = displayed.of_type(widgets.HBox)[0]
    assert [button.description for button in stepper.children] == [
        "Prev",
        "Save Setting",
        "Next",
    ]


def test_the_dropdown_opens_on_the_first_syllable(analysis_paths, labeled, displayed):
    """11. A labeller with nothing selected shows nothing.

    Every control in the labeller indexes the syllable list by the
    dropdown's position, and there is no entry under an empty selection.
    Newer ipywidgets leaves a dropdown unselected when its options are
    assigned, so the selection is made explicitly.
    """
    label_syllables(analysis_paths, **LABELLING)

    dropdown = displayed.one(widgets.Dropdown)
    assert dropdown.index == 0, "the labeller opens with no syllable chosen"
    assert list(dropdown.options)[0].startswith("0 - ")


@pytest.mark.xfail(
    reason="interactive_syllable_labeler_wrapper defines on_syll_change and "
    "registers it with output.observe after its own return statement, so "
    "neither ever runs. Nothing is lost by it: interactive_output already "
    "redraws the view when the dropdown changes."
)
def test_the_wrapper_runs_everything_it_contains():
    """12. Statements the function can never reach.

    The last thing in the wrapper is a handler meant to redraw the view
    when a user picks a different syllable, together with the call that
    registers it. Both sit after the function has returned.
    """
    import ast
    import inspect
    import textwrap

    from moseq2_app.gui.wrappers import interactive_syllable_labeler_wrapper

    source = textwrap.dedent(inspect.getsource(interactive_syllable_labeler_wrapper))
    body = ast.parse(source).body[0].body
    returns = [i for i, node in enumerate(body) if isinstance(node, ast.Return)]

    assert not returns or returns[-1] == len(body) - 1, (
        f"{len(body) - 1 - returns[-1]} statements follow the return"
    )
