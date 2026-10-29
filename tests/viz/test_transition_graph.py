"""Transition graphs drawn by moseq2-viz.

``moseq2-viz plot-transition-graph`` draws one graph per group and one per
pair of groups showing how the two differ. A difference graph keeps only the
syllables with an edge above the threshold, so its nodes are a subset of the
syllables and not numbered from zero.
"""

from __future__ import annotations

import numpy as np

from moseq2_viz.model.trans_graph import make_difference_graphs


def test_difference_nodes_are_sized_and_coloured_by_their_own_usage():
    """Each node shows the usage difference of the syllable it stands for."""
    a, b = np.zeros((6, 6)), np.zeros((6, 6))
    b[3, 5] = 0.1
    usage_a = dict.fromkeys(range(6), 0.1)
    usage_b = {**usage_a, 0: 0.9, 1: 0.8, 3: 0.2, 5: 0.05}

    result = make_difference_graphs(
        [a, b], [usage_a, usage_b], ["a", "b"], ["a", "b"], {}, [], {}, [],
        node_sizes=[], difference_graphs=[], usage_scale=1,
    )
    graph, node_sizes, node_colours = result[2][0], result[4][0], result[5][0]

    assert list(graph.nodes) == [3, 5]
    np.testing.assert_allclose(node_sizes, [0.1, 0.05])
    assert node_colours == ["b", "r"]


def test_the_difference_legend_names_the_direction_each_colour_means():
    """Blue is higher in the first group of a 'b - a' panel title, red is lower."""
    from moseq2_viz.model.trans_graph import compute_and_graph_grouped_TMs

    labels = [np.tile([0, 0, 2, 2, 1, 1, 2, 2], 50), np.tile([0, 0, 1, 1, 2, 2], 60)]
    fig = compute_and_graph_grouped_TMs(
        {"max_syllable": 3, "normalize": "bigram", "scale_node_by_usage": True},
        labels, ["a", "b"], ["a", "b"],
    )
    assert "b - a" in [ax.get_title() for ax in fig.axes]

    legend = fig.legends[0]
    colours = {
        text.get_text(): (handle.get_color(), handle.get_markeredgecolor())
        for handle, text in zip(legend.legend_handles, legend.get_texts())
    }
    assert colours == {
        "Transition more likely in the first group of the panel title": ("b", "b"),
        "Transition less likely in the first group of the panel title": ("r", "r"),
        "Usage higher in the first group of the panel title": ("w", "b"),
        "Usage lower in the first group of the panel title": ("w", "r"),
    }
