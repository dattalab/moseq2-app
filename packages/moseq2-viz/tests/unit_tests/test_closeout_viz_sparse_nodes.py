"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd





def test_difference_graph_uses_node_identity_after_thresholding():
    from moseq2_viz.model.trans_graph import make_difference_graphs
    a, b = np.zeros((6, 6)), np.zeros((6, 6))
    b[3, 5] = .1
    ua = dict.fromkeys(range(6), .1)
    ub = dict(ua, **{})
    ub[0], ub[1], ub[3], ub[5] = .9, .8, .2, .05
    result = make_difference_graphs([a, b], [ua, ub], ['a', 'b'], ['a', 'b'], {}, [], {}, [],
                                    node_sizes=[], difference_graphs=[], usage_scale=1)
    graph = result[2][0]
    assert list(graph.nodes) == [3, 5]
    np.testing.assert_allclose(result[4][0], [.1, .05])
    assert result[5][0] == ['b', 'r']
