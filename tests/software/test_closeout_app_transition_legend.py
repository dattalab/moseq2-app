"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd
import h5py


def test_transition_legend_inverts_the_drawn_scale():
    from moseq2_app.stat.view import get_minmax_tp
    actual = get_minmax_tp({("a", "b"): .004 * 250, ("b", "c"): .012 * 250}, diff=False)
    np.testing.assert_allclose(actual, [.004, .012])


def test_difference_legend_retains_negative_edge_probabilities():
    import networkx as nx
    from bokeh.plotting import figure
    from moseq2_app.stat.view import format_trans_graph_edges, get_minmax_tp, get_legend_items
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([(0, 1, -.01), (1, 2, .02)])
    _, widths, _ = format_trans_graph_edges(graph, {e: 'black' for e in graph.edges}, True)
    weights = {e: graph.edges[e]['weight'] for e in graph.edges}
    np.testing.assert_allclose(get_minmax_tp(widths, True, signed_weights=weights), [.01, .01, .02, .02])
    _, info = get_legend_items(figure(), widths, 'a-b', True, signed_weights=weights)
    down = [item for item in info.items if 'Down-regulated' in item.label.value]
    assert len(down) == 2
    for item in down:
        assert '0.0100' in item.label.value
        assert item.renderers[0].glyph.line_width == 4
