"""The width legend on the analysis notebook's interactive transition graphs.

Edges are drawn at a width proportional to their transition probability, or
on a difference graph to the size of the difference between the two groups.
The legend shows the thinnest and thickest edge of each kind with the
probability it stands for, so it has to undo the same scale the edges were
drawn with, and on a difference graph it reads the signed differences to
tell the two directions apart.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from bokeh.plotting import figure

from moseq2_app.stat.view import format_trans_graph_edges, get_legend_items, get_minmax_tp


def test_the_legend_reads_back_the_probabilities_the_edges_were_drawn_from():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([("a", "b", 0.004), ("b", "c", 0.012)])
    _, widths, _ = format_trans_graph_edges(graph, {e: "black" for e in graph.edges}, False)
    np.testing.assert_allclose(get_minmax_tp(widths, diff=False), [0.004, 0.012])


def test_a_difference_legend_shows_both_directions():
    """A transition less likely in the first group is listed with its own size."""
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, -0.01), (1, 2, 0.02)])
    _, widths, _ = format_trans_graph_edges(graph, {e: "black" for e in graph.edges}, True)
    weights = {e: graph.edges[e]["weight"] for e in graph.edges}
    np.testing.assert_allclose(get_minmax_tp(widths, True, signed_weights=weights), [0.01, 0.01, 0.02, 0.02])

    _, info = get_legend_items(figure(), widths, "b - a", True, signed_weights=weights)
    red = [item for item in info.items if item.renderers[0].glyph.line_color == "red"]
    assert len(red) == 2
    for item in red:
        assert "0.0100" in item.label.value
        assert item.renderers[0].glyph.line_width == 4


def test_a_difference_legend_names_both_groups():
    """Each colour says which group is higher than which, and by how much."""
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, -0.01), (1, 2, 0.02)])
    _, widths, _ = format_trans_graph_edges(graph, {e: "black" for e in graph.edges}, True)
    weights = {e: graph.edges[e]["weight"] for e in graph.edges}

    main, info = get_legend_items(figure(), widths, "b - a", True, signed_weights=weights)
    labels = [item.label.value for item in main.items]
    assert labels[-4:] == [
        "Usage higher in b than a",
        "Usage lower in b than a",
        "Transition more likely in b than a",
        "Transition less likely in b than a",
    ]
    assert [item.label.value for item in info.items] == [
        "Thinnest blue edge: b higher by 0.0200",
        "Thickest blue edge: b higher by 0.0200",
        "Thinnest red edge: b lower by 0.0100",
        "Thickest red edge: b lower by 0.0100",
    ]


def draw(monkeypatch, groups, differences):
    """Draw two groups and their difference the way the notebook's transition-graph cell does.

    ``differences`` maps each edge to its signed difference, second group
    minus first; the group graphs share one edge set. Returns the legend
    labels of each panel, keyed by the panel's title.
    """
    from bokeh.models import Legend

    from moseq2_app.stat import view

    shown = []
    monkeypatch.setattr(view, "show", shown.append)

    nodes = sorted({n for edge in differences for n in edge})
    group_graph = nx.DiGraph()
    group_graph.add_weighted_edges_from((u, v, 0.05) for u, v in differences)
    difference = nx.DiGraph()
    difference.add_weighted_edges_from((u, v, w) for (u, v), w in differences.items())
    pos = nx.circular_layout(nodes)
    usage = dict.fromkeys(nodes, 1 / len(nodes))
    scalar = {key: [dict.fromkeys(nodes, 1.0)] * 3
              for key in ("duration", "speeds_2d", "speeds_3d", "heights", "dists")}
    syll_info = {n: {"label": f"s{n}", "desc": "", "crowd_movie_path": ""} for n in nodes}
    title = f"{groups[1]} - {groups[0]}"

    view.plot_interactive_transition_graph(
        [group_graph, group_graph.copy(), difference], pos, list(groups), [*groups, title],
        [usage, usage, dict.fromkeys(nodes, 0.01)], syll_info, [], [], scalars=scalar,
    )

    panels = {}
    for child in shown[0].children:
        plot = child[0]
        panels[plot.title.text] = [
            item.label.value
            for legend in plot.renderers if isinstance(legend, Legend)
            for item in legend.items
        ]
    return panels


def test_a_group_name_with_a_spaced_hyphen_is_named_whole(monkeypatch):
    """The difference legend names the groups it compares, not pieces of its title.

    The title of a difference panel joins the two group names with " - ",
    so a group named "WT - saline" makes the title "WT - saline - KO",
    which reads three ways. The legend takes the names themselves.
    """
    panels = draw(monkeypatch, ("KO", "WT - saline"), {(0, 1): -0.01, (1, 2): 0.02})

    labels = panels["WT - saline - KO"]
    assert "Usage higher in WT - saline than KO" in labels, labels
    assert "Thickest blue edge: WT - saline higher by 0.0200" in labels, labels


def test_the_drawn_difference_legend_tells_the_directions_apart(monkeypatch):
    """The notebook's graph hands the legend each edge's signed difference.

    Edge widths on a difference graph are sizes, so only the signed
    weights say which edges went down. Drawn through the notebook's own
    path, the red entries carry the edge that is lower in b.
    """
    panels = draw(monkeypatch, ("a", "b"), {(0, 1): -0.01, (1, 2): 0.02})

    labels = panels["b - a"]
    assert "Thinnest red edge: b lower by 0.0100" in labels, labels
    assert "Thickest blue edge: b higher by 0.0200" in labels, labels
