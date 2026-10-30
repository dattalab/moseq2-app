"""Cell 14: the transition matrices as a graph you can prune.

    from moseq2_app.main import interactive_transition_graph
    interactive_transition_graph(progress_paths, max_syllables=max_syllables,
                                 plot_vertically=True)

Cell 12's matrices drawn as networks: a node per syllable, sized by how
much it is used, an edge per transition, weighted by how likely it is.
One graph per group, laid out identically so they can be compared.

A full graph of sixty syllables is sixty squared edges and shows nothing.
The two sliders are what makes it readable: one drops edges below a
transition probability, the other drops syllables below a usage. A user
raises them until the structure that survives is small enough to see,
which is why their ranges have to be set from the data rather than
guessed.

Hovering a node plays that syllable's crowd movie, which is why the
movies are carried into the graph rather than linked.
"""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np
import pytest

from moseq2_app.main import interactive_transition_graph
from moseq2_app.stat.controller import InteractiveTransitionGraph

#: What the cell asks for.
GRAPH = {"plot_vertically": True}


@pytest.fixture(scope="module")
def graph(analysis_paths, labeled):
    """The widget behind the cell, built once."""
    return InteractiveTransitionGraph(
        model_path=analysis_paths["model_path"],
        index_path=analysis_paths["index_file"],
        info_path=analysis_paths["syll_info"],
        max_sylls=labeled,
        **GRAPH,
    )


def test_the_cell_displays_the_thresholds_and_the_graphs(analysis_paths, labeled, displayed):
    """1. What a user is given: a reset, the two sliders, and the figure."""
    assert (
        interactive_transition_graph(analysis_paths, max_syllables=labeled, **GRAPH) is None
    )

    assert [type(obj).__name__ for obj in displayed.objects] == ["Button", "HBox", "Output"]
    assert displayed.one(widgets.Button).description == "Clear Output"


def test_there_is_a_graph_for_every_group(graph, transitions):
    """2. One network per condition, built from the same syllables.

    The groups come from the model rather than the index, the same way
    cell 12 takes them, so the two figures describe the same split.
    """
    assert graph.group == sorted(transitions["group"])
    assert len(graph.trans_mats) == len(graph.group)
    assert len(graph.usages) == len(graph.group)


def test_the_graphs_cover_the_syllables_cell_9_settled_on(graph, labeled):
    """3. The same cut-off as every cell since the labeller."""
    assert graph.max_sylls == labeled
    for matrix in graph.trans_mats:
        assert matrix.shape == (labeled, labeled)
    assert graph.df["syllable"].max() < labeled


def test_each_group_contributes_one_row_per_syllable(graph, labeled):
    """4. What the node sizes and the hover text are read from.

    The per-session statistics are averaged within each group, so the
    graph shows a group's typical behaviour rather than any one animal's.
    """
    assert len(graph.df) == len(graph.group) * labeled
    counts = graph.df.groupby("group")["syllable"].nunique()
    assert (counts == labeled).all()


def test_the_edge_slider_spans_the_transition_probabilities_present(graph):
    """5. A slider that cannot reach the data is useless.

    Edges are dropped below the value a user sets, so the range has to
    run from nothing dropped up to the strongest transition in either
    group. It opens with everything shown.
    """
    strongest = float(np.max(graph.trans_mats))

    assert graph.edge_thresholder.min == 0
    assert graph.edge_thresholder.max == pytest.approx(strongest)
    assert graph.edge_thresholder.value == pytest.approx((0, strongest))


def test_the_usage_slider_spans_the_usages_present(graph):
    """6. The same, for dropping whole syllables.

    Raising this leaves only the syllables a group actually spends time
    in, which is usually what makes the graph readable.
    """
    assert graph.usage_thresholder.min == 0
    assert graph.usage_thresholder.max > 0
    assert graph.usage_thresholder.value[0] == 0
    assert graph.usage_thresholder.value[1] == pytest.approx(graph.usage_thresholder.max)


def test_the_layout_choice_the_cell_makes_is_carried(graph):
    """7. ``plot_vertically`` stacks the groups rather than placing them side by side.

    With sixty-syllable graphs the horizontal arrangement runs off the
    notebook's width, which is why the cell overrides the default.
    """
    assert graph.plot_vertically is True


def test_each_syllable_carries_its_movie_into_the_graph(graph, labeled):
    """8. Hovering a node plays the syllable.

    The movie is base64-encoded into the syllable's metadata rather than
    linked, so the tooltip works in a notebook that has been moved or
    sent to someone else. By the time the graph is built the path has
    been replaced by the encoded movie.
    """
    import base64

    for syllable in range(labeled):
        encoded = graph.syll_info[syllable]["crowd_movie_path"]
        assert encoded, f"syllable {syllable} carries no movie"
        assert base64.b64decode(encoded)[4:8] == b"ftyp", (
            f"syllable {syllable} does not carry an mp4"
        )


def test_the_layout_options_cover_the_arrangements_offered(graph):
    """9. How the nodes are placed.

    The arrangement decides whether the structure is visible at all, and
    it is the first thing a user changes when a graph comes out as a
    hairball.
    """
    assert len(graph.graph_layout_dropdown.options) > 1
    assert graph.graph_layout_dropdown.value in graph.graph_layout_dropdown.options
