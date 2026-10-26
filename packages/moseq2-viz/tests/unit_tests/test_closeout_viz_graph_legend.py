"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd


def test_difference_graph_legend_matches_edge_sign_colors(monkeypatch):
    import moseq2_viz.model.trans_graph as graph
    import matplotlib.pyplot as plt
    captured = {}
    monkeypatch.setattr(graph, "get_group_trans_mats", lambda *a: ([np.zeros((2, 2))] * 2, None))
    monkeypatch.setattr(graph, "graph_transition_matrix", lambda *a, **kw: (plt, None, None))
    monkeypatch.setattr(plt, "legend", lambda **kw: captured.update(kw))
    graph.compute_and_graph_grouped_TMs(
        {"max_syllable": 2, "normalize": None, "scale_node_by_usage": False},
        [], [], ["synthetic-a", "synthetic-b"])
    handles = {item.get_label(): item for item in captured["handles"]}
    up = next(value for label, value in handles.items() if label.startswith("Up-regulated trans"))
    down = next(value for label, value in handles.items() if label.startswith("Down-regulated trans"))
    from matplotlib.colors import to_rgba
    assert to_rgba(up.get_color()) == to_rgba("blue")
    assert to_rgba(down.get_color()) == to_rgba("red")
