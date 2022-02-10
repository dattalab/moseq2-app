import numpy as np
from unittest import TestCase
from bokeh.plotting import figure
from moseq2_app.stat.view import colorscale, format_graphs, get_difference_legend_items

class TestStatView(TestCase):

    def test_graph_dendrogram(self):
        pass

    def test_colorscale(self):
        ret = colorscale('#111111', -1)
        assert ret == '111111'

    def test_format_graphs(self):

        graphs = [ 'a', 'b', 'c',  'b-a', 'c-b', 'c-a' ]

        group = ['a', 'b', 'c']

        output = format_graphs(graphs, group)

        expected_output = np.array([
                            np.array(['a', 'b-a', 'c-b'], dtype=object),
                            np.array([None, 'b', 'c-a'], dtype=object),
                            np.array([None, None, 'c'], dtype=object)
        ])

        for i in range(3):
            assert all(output[i] == expected_output[i])

    def test_get_difference_legend_items(self):

        fig = figure()

        edge_widths = {i: x for i, x in enumerate(range(10, 1))}

        diff_main_items, diff_width_items = get_difference_legend_items(fig, edge_widths, 'default')

        assert len(diff_main_items) == 4
        assert len(diff_width_items) == 4

    def test_bokeh_plotting(self):
        pass

    def test_draw_stats(self):
        pass

    def test_get_neighbors_and_entropies(self):
        pass

    def test_plot_interactive_transition_graph(self):
        pass