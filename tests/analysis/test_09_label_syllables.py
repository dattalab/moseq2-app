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
import shutil
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pytest

from moseq2_app.main import label_syllables
from moseq2_viz.model.util import parse_model_results
from moseq2_viz.util import read_yaml

from conftest import LABELLING
from video import index_is_at_the_front, probe

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

    It is the fewest syllables, best used first, that reach the requested
    share of the cumulative usage curve: that many reach it and one fewer
    falls short.
    """
    coverage = syllable_coverage(analysis_paths["model_path"])

    assert coverage[labeled - 1] >= LABELLING["n_explained"]
    assert coverage[labeled - 2] < LABELLING["n_explained"]


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


def test_the_wrapper_runs_everything_it_contains():
    """12. Nothing in the wrapper sits after its return.

    ``interactive_output`` redraws the view when a user picks a different
    syllable, so the wrapper needs no handler of its own for that.
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


@pytest.fixture(scope="module")
def moved(tmp_path_factory, analysis_project, labeled):
    """The labelled project copied elsewhere, holding movies a user might have.

    The copy's label file still points at the original's movies, so opening
    the labeller in the copy reads its crowd-movie directory afresh. Two
    things are put in that directory first: one movie renamed the way
    moseq2-viz named them before late 2020, with a space where the
    underscore now is, and an .mp4 that is not a crowd movie at all.

    Returns the copy's label file as the cell left it, its crowd-movie
    directory, and the renamed movie.
    """
    root = tmp_path_factory.mktemp("moved")
    shutil.copytree(analysis_project, root, dirs_exist_ok=True)
    progress = root / "progress.yaml"
    progress.write_text(progress.read_text().replace(str(analysis_project), str(root)))
    paths = read_yaml(str(progress))

    crowd = Path(paths["crowd_dir"])
    first = next(m for m in crowd.glob("*.mp4") if int(MOVIE_NAME.search(m.name)["sorted_id"]) == 0)
    ids = MOVIE_NAME.search(first.name)
    older = crowd / (
        f"syllable_sorted-id-{int(ids['sorted_id'])} ({ids['sort_type']})"
        f"_original-id-{int(ids['original_id'])}.mp4"
    )
    first.rename(older)
    (crowd / "notes.mp4").write_bytes(b"not a crowd movie")

    label_syllables(paths, **LABELLING)
    return read_yaml(paths["syll_info"]), crowd, older


def test_every_syllable_finds_its_movie_beside_an_unrelated_one(moved, labeled):
    """13. A crowd-movie directory can hold files the cell did not write.

    Each syllable's movie is found by reading its ids out of the filename,
    and a file whose name carries none is passed over rather than read.
    """
    syll_info, crowd, _ = moved
    for syllable in range(labeled):
        movie = Path(syll_info[syllable]["crowd_movie_path"])
        assert movie.parent == crowd, f"syllable {syllable} still points at {movie}"
        assert movie.exists(), f"syllable {syllable} points at {movie}, which is not there"
        assert movie.name != "notes.mp4"


def test_a_movie_named_by_an_older_moseq2_viz_is_found(moved):
    """14. Projects keep the crowd movies they were given.

    moseq2-viz put a space before the sort type until late 2020 and an
    underscore since. A project whose movies predate the change still has
    them under the old name.
    """
    syll_info, _, older = moved
    assert Path(syll_info[0]["crowd_movie_path"]) == older


def _walk(widget):
    """Every widget inside ``widget``, depth first, itself included."""
    yield widget
    for child in getattr(widget, "children", ()):
        yield from _walk(child)


def _labeller_controls(display):
    """The labelling view's name box and Save button, from what the cell drew."""
    view = display.of_type(widgets.AppLayout)[-1]
    name = next(
        w for w in _walk(view.right_sidebar)
        if isinstance(w, widgets.Text) and w.placeholder == "Syllable Name"
    )
    save = next(
        w for w in display.of_type(widgets.HBox)
        for w in w.children
        if isinstance(w, widgets.Button) and w.description == "Save Setting"
    )
    return name, save


def _typed(syllable):
    return {"label": f"name {syllable}", "desc": f"what syllable {syllable} looks like"}


@pytest.fixture(scope="module")
def relabelled(tmp_path_factory, analysis_project, labeled):
    """A labelled project reopened at a different syllable count, twice.

    A user labels fewer syllables than the cell would settle on, by typing
    a count. They then reopen the labeller at the cell's own count, which
    is also what upgrading does to a project whose count moved with the
    version, and after that at a count lower than the one they labelled.
    At the lower count they rename the first syllable and save.

    Returns the count labelled, the label file and the dropdown's options
    after each reopening, and the label file after the save.
    """
    from moseq2_app.util import write_yaml

    from widgets import capture_display

    root = tmp_path_factory.mktemp("relabelled")
    shutil.copytree(analysis_project, root, dirs_exist_ok=True)
    progress = root / "progress.yaml"
    progress.write_text(progress.read_text().replace(str(analysis_project), str(root)))
    paths = read_yaml(str(progress))

    typed = labeled - 3
    lower = typed - 4
    seen = {}
    with pytest.MonkeyPatch.context() as patch:
        display = capture_display(patch)

        label_syllables(paths, **{**LABELLING, "max_syllables": typed})
        syll_info = read_yaml(paths["syll_info"])
        for syllable in range(typed):
            syll_info[syllable].update(_typed(syllable))
        write_yaml(syll_info, paths["syll_info"])

        for name, count in (("raised", None), ("lowered", lower)):
            display.clear()
            label_syllables(paths, **{**LABELLING, "max_syllables": count})
            dropdown = display.of_type(widgets.Dropdown)[-1]
            seen[name] = read_yaml(paths["syll_info"]), list(dropdown.options)

        name, save = _labeller_controls(display)
        name.value = "renamed"
        save.click()
        seen["saved"] = read_yaml(paths["syll_info"])

    return typed, lower, seen, paths


def test_raising_the_count_keeps_every_typed_label(relabelled, labeled):
    """15. Reopening the labeller at a higher count adds blanks, nothing else.

    A project labelled under a version that settled on one syllable fewer
    is reopened here with one more. Every name and description already
    typed is shown and kept on disk, and only the syllables new to the
    count arrive blank.
    """
    typed, _, seen, _ = relabelled
    on_disk, options = seen["raised"]

    assert len(options) == labeled
    for syllable in range(typed):
        assert {k: on_disk[syllable][k] for k in ("label", "desc")} == _typed(syllable), syllable
        assert options[syllable] == f"{syllable} - name {syllable}"
    for syllable in range(typed, labeled):
        assert on_disk[syllable]["label"] == "" and on_disk[syllable]["desc"] == ""


def test_lowering_the_count_hides_labels_without_deleting_them(relabelled):
    """16. The syllables left out are left alone on disk.

    Offering fewer syllables is a choice about what to look at now. The
    names typed for the syllables beyond it stay in the file, so raising
    the count again brings them back.
    """
    typed, lower, seen, _ = relabelled
    on_disk, options = seen["lowered"]

    assert options == [f"{syllable} - name {syllable}" for syllable in range(lower)]
    for syllable in range(typed):
        assert {k: on_disk[syllable][k] for k in ("label", "desc")} == _typed(syllable), syllable


def test_saving_writes_the_shown_syllables_and_keeps_the_rest(relabelled, labeled):
    """17. A save at a lower count changes only what was edited.

    Saving writes the syllables the labeller is showing back to the file.
    Entries for syllables beyond the count are written back exactly as
    they were read.
    """
    typed, _, seen, _ = relabelled
    before, _ = seen["lowered"]
    after = seen["saved"]

    assert after[0]["label"] == "renamed"
    assert sorted(after) == sorted(before) == list(range(labeled))
    for syllable in range(1, labeled):
        assert after[syllable] == before[syllable], syllable


def test_a_count_beyond_what_the_model_uses_is_refused(relabelled, analysis_paths):
    """18. There is nothing to label past the model's last syllable.

    Every syllable the labeller offers is one the recordings contain. A
    typed count larger than the number the model used is refused with the
    number it can be, and the label file is not touched.
    """
    *_, paths = relabelled
    model = parse_model_results(analysis_paths["model_path"])
    used = len({int(label) for labels in model["labels"] for label in np.unique(labels) if label >= 0})
    before = Path(paths["syll_info"]).read_bytes()

    with pytest.raises(ValueError, match=rf"\b{used}\b"):
        label_syllables(paths, **{**LABELLING, "max_syllables": used + 1})

    assert Path(paths["syll_info"]).read_bytes() == before


def test_a_missing_movie_is_made_even_beside_an_unrelated_one(
    tmp_path, analysis_project, labeled, monkeypatch
):
    """19. A crowd-movie directory missing a syllable's movie is regenerated.

    The cell checks that every syllable it offers has a movie before it
    reuses the directory. A file that is not a crowd movie, such as a
    stray notes.mp4, does not stand in for the one that is missing.
    """
    from moseq2_app.viz import controller

    shutil.copytree(analysis_project, tmp_path, dirs_exist_ok=True)
    progress = tmp_path / "progress.yaml"
    progress.write_text(progress.read_text().replace(str(analysis_project), str(tmp_path)))
    paths = read_yaml(str(progress))

    crowd = Path(paths["crowd_dir"])
    last = next(m for m in crowd.glob("*.mp4") if int(MOVIE_NAME.search(m.name)["sorted_id"]) == labeled - 1)
    last.unlink()
    (crowd / "notes.mp4").write_bytes(b"not a crowd movie")

    made = []

    def make_movies(index_file, model_path, output_dir, config_data):
        made.append(output_dir)
        shutil.copy(next(crowd.glob("syllable_*.mp4")), last)
        return {"all": sorted(str(m) for m in crowd.glob("syllable_*.mp4"))}

    monkeypatch.setattr(controller, "make_crowd_movies_wrapper", make_movies)
    label_syllables(paths, **LABELLING)

    assert made, "the movies were not regenerated"
    movie = Path(read_yaml(paths["syll_info"])[labeled - 1]["crowd_movie_path"])
    assert movie == last and movie.exists(), movie
