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
