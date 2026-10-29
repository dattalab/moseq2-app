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
