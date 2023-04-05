import os
import shutil
import bokeh.io
from os.path import exists
from unittest import TestCase
from moseq2_app.main import label_syllables
from moseq2_viz.model.util import relabel_by_usage
from moseq2_viz.model.trans_graph import get_trans_graph_groups
from moseq2_app.stat.controller import InteractiveSyllableStats, InteractiveTransitionGraph

# class TestSyllableStatController(TestCase):

#     @classmethod
#     def setUpClass(cls):
#         progress_paths = {
#             'base_dir': 'data/',
#             'config_file': 'data/config.yaml',
#             'session_config': 'data/session_config.yaml',
#             'index_file': 'data/test_index.yaml',
#             'model_path': 'data/test_model.p',
#             'crowd_dir': 'data/crowd_movies/',
#             'syll_info': 'data/syll_info.yaml',
#             'df_info_path': 'data/syll_df.parquet',
#             'plot_path': 'data/plot'
#         }

#         if exists(progress_paths['syll_info']):
#             os.remove(progress_paths['syll_info'])

#         if exists(progress_paths['df_info_path']):
#             os.remove(progress_paths['df_info_path'])

#         if exists('data/moseq_scalar_dataframe.parquet'):
#             os.remove('data/moseq_scalar_dataframe.parquet')

#         if exists(progress_paths['crowd_dir']):
#             shutil.rmtree(progress_paths['crowd_dir'])

#         label_syllables(progress_paths, max_syllables=None, n_explained=99)

#     def setUp(self):

#         self.progress_paths = {
#             'base_dir': 'data/',
#             'config_file': 'data/config.yaml',
#             'session_config': 'data/session_config.yaml',
#             'index_file': 'data/test_index.yaml',
#             'model_path': 'data/test_model.p',
#             'crowd_dir': 'data/crowd_movies/',
#             'syll_info': 'data/syll_info.yaml',
#             'df_info_path': 'data/syll_df.parquet',
#             'plot_path': 'data/plot'
#         }


#         bokeh.io.output_notebook()
#         self.gui = InteractiveSyllableStats(index_path=self.progress_paths['index_file'],
#                                             model_path=self.progress_paths['model_path'],
#                                             df_path=self.progress_paths['df_info_path'],
#                                             info_path=self.progress_paths['syll_info'],
#                                             max_sylls=None,
#                                             load_parquet=False)


#     def tearDown(self):
#         del self.gui

#     @classmethod
#     def tearDownClass(cls):
#         group_movie_dir = 'data/grouped_crowd_dir/'
#         if exists(group_movie_dir):
#             shutil.rmtree(group_movie_dir)

#         if exists('data/syll_info.yaml'):
#             os.remove('data/syll_info.yaml')

#         if exists('data/syll_df.parquet'):
#             os.remove('data/syll_df.parquet')

#         if exists('data/moseq_scalar_dataframe.parquet'):
#             os.remove('data/moseq_scalar_dataframe.parquet')

#         if exists('data/crowd_movies/'):
#             shutil.rmtree('data/crowd_movies/')

#     ## Syllable Stat Tests
#     def test_init_stats(self):
#         assert len(self.gui.session_names) == 1

#     def test_clear_on_click(self):
#         self.gui.clear_on_click()

#     def test_on_grouping_update(self):
#         class Event:
#             def __init__(self, new='SessionName'):
#                 self.new = new

#         event = Event()

#         self.gui.on_grouping_update(event)
#         assert list(self.gui.session_sel.options) == self.gui.session_names

#         event = Event('SubjectName')

#         self.gui.on_grouping_update(event)
#         assert list(self.gui.session_sel.options) == self.gui.subject_names

#         event = Event('Group')

#         self.gui.on_grouping_update(event)
#         assert list(self.gui.session_sel.value) == [self.gui.session_sel.options[0]]
#         assert self.gui.session_sel.layout.display == "none"

#     def test_compute_dendrogram(self):

#         self.gui.cladogram = None
#         self.gui.compute_dendrogram()

#         assert list(self.gui.results.keys()) == ['icoord', 'dcoord', 'ivl', 'leaves', 'color_list']
#         assert len(self.gui.icoord) == 9
#         assert len(self.gui.dcoord) == 9

#     def test_interactive_stat_helper(self):

#         self.gui.interactive_stat_helper()

#         assert isinstance(self.gui.sorted_index, dict)
#         assert len(self.gui.df.columns) == 34

#         assert len(self.gui.df.syllable.unique()) == 10

#         self.gui.df_path = None
#         self.gui.interactive_stat_helper()
#         assert len(self.gui.df.columns) == 34

#         assert len(self.gui.df.syllable.unique()) == 10

#     def test_interactive_syll_stats_grapher(self):

#         stat = 'usage'
#         sort = 'usage'
#         groupby = 'group'
#         errorbar = 'CI 95%'
#         sessions = self.gui.session_names
#         ctrl_group = 'default'
#         exp_group = 'default'

#         self.gui.interactive_syll_stats_grapher(stat, sort, groupby, errorbar, sessions, ctrl_group, exp_group)

#         sort = 'similarity'
#         errorbar = 'SEM'
#         self.gui.interactive_syll_stats_grapher(stat, sort, groupby, errorbar, sessions, ctrl_group, exp_group, 'Z-Test')

#         sort = 'difference'
#         errorbar = 'STD'
#         self.gui.interactive_syll_stats_grapher(stat, sort, groupby, errorbar, sessions, ctrl_group, exp_group, 'T-Test')

# class TestTransitionGraphController(TestCase):

#     @classmethod
#     def setUpClass(cls):
#         progress_paths = {
#             'base_dir': 'data/',
#             'config_file': 'data/config.yaml',
#             'session_config': 'data/session_config.yaml',
#             'index_file': 'data/test_index.yaml',
#             'model_path': 'data/test_model.p',
#             'crowd_dir': 'data/crowd_movies/',
#             'syll_info': 'data/syll_info.yaml',
#             'df_info_path': 'data/syll_df.parquet',
#             'plot_path': 'data/plot'
#         }

#         if exists(progress_paths['syll_info']):
#             os.remove(progress_paths['syll_info'])

#         if exists(progress_paths['df_info_path']):
#             os.remove(progress_paths['df_info_path'])

#         if exists('data/moseq_scalar_dataframe.parquet'):
#             os.remove('data/moseq_scalar_dataframe.parquet')

#         if exists(progress_paths['crowd_dir']):
#             shutil.rmtree(progress_paths['crowd_dir'])

#         label_syllables(progress_paths, max_syllables=None, n_explained=99)

#     def setUp(self):

#         self.progress_paths = {
#             'base_dir': 'data/',
#             'config_file': 'data/config.yaml',
#             'session_config': 'data/session_config.yaml',
#             'index_file': 'data/test_index.yaml',
#             'model_path': 'data/test_model.p',
#             'crowd_dir': 'data/crowd_movies/',
#             'syll_info': 'data/syll_info.yaml',
#             'df_info_path': 'data/syll_df.parquet',
#             'plot_path': 'data/plot'
#         }

#         bokeh.io.output_notebook()
#         self.gui = InteractiveTransitionGraph(model_path=self.progress_paths['model_path'],
#                                               index_path=self.progress_paths['index_file'],
#                                               info_path=self.progress_paths['syll_info'],
#                                               df_path=self.progress_paths['df_info_path'],
#                                               max_sylls=None,
#                                               plot_vertically=True,
#                                               load_parquet=False)

#     def tearDown(self):
#         del self.gui

#     @classmethod
#     def tearDownClass(cls):
#         group_movie_dir = 'data/grouped_crowd_dir/'
#         if exists(group_movie_dir):
#             shutil.rmtree(group_movie_dir)

#         if exists('data/syll_info.yaml'):
#             os.remove('data/syll_info.yaml')

#         if exists('data/syll_df.parquet'):
#             os.remove('data/syll_df.parquet')

#         if exists('data/moseq_scalar_dataframe.parquet'):
#             os.remove('data/moseq_scalar_dataframe.parquet')

#         if exists('data/crowd_movies/'):
#             shutil.rmtree('data/crowd_movies/')

#     def test_init_trans_graph(self):

#         scalar_dict = {
#             'Default': 'speeds_2d',
#             'Duration': 'duration',
#             '2D velocity': 'speeds_2d',
#             '3D velocity': 'speeds_3d',
#             'Height': 'heights',
#             'Distance to Center': 'dists'
#         }

#         assert self.gui.scalar_dict == scalar_dict
#         assert self.gui.df_path == None
#         assert len(self.gui.syll_info) == 10

#         assert len(self.gui.df.syllable.unique()) == 10
#         assert self.gui.group == ['default']
#         assert len(self.gui.trans_mats) == 1
#         assert len(self.gui.incoming_transition_entropy) == 1
#         assert len(self.gui.outgoing_transition_entropy) == 1

#     def test_clear_on_click(self):
#         self.gui.clear_on_click()

#     def test_set_range_widget_values(self):

#         self.gui.set_range_widget_values()

#         assert self.gui.edge_thresholder.value == (0, 0.125)
#         assert self.gui.usage_thresholder.value == (0, 0.6744186046511628)
#         assert self.gui.speed_thresholder.value == (0, 5.0545125007629395)

#     def test_compute_entropies(self):

#         # Get labels and optionally relabel them by usage sorting
#         labels = self.gui.model_fit['labels']
#         label_group, _ = get_trans_graph_groups(self.gui.model_fit)
#         labels = relabel_by_usage(labels, count='usage')[0]

#         self.gui.compute_entropies(labels, label_group)

#         assert len(self.gui.incoming_transition_entropy) == 1
#         assert len(self.gui.outgoing_transition_entropy) == 1

#         assert len(self.gui.incoming_transition_entropy[0]) == 10
#         assert len(self.gui.outgoing_transition_entropy[0]) == 10

#     def test_compute_entropy_differences(self):

#         # Get labels and optionally relabel them by usage sorting
#         labels = self.gui.model_fit['labels']
#         label_group, _ = get_trans_graph_groups(self.gui.model_fit)
#         labels = relabel_by_usage(labels, count='usage')[0]

#         self.gui.compute_entropies(labels, label_group)

#         self.gui.compute_entropy_differences()

#         assert len(self.gui.incoming_transition_entropy) == 1
#         assert len(self.gui.outgoing_transition_entropy) == 1

#     def test_initialize_transition_data(self):

#         self.gui.initialize_transition_data()

#         assert len(self.gui.df.syllable.unique()) == 10
#         assert len(self.gui.incoming_transition_entropy) == 1
#         assert len(self.gui.outgoing_transition_entropy) == 1
#         assert len(self.gui.trans_mats) == 1

#     def test_interactive_transition_graph_helper(self):

#         layout = 'spring'
#         scalar_color = 'speeds_2d'

#         self.gui.initialize_transition_data()
#         self.gui.interactive_transition_graph_helper(layout,
#                                                      scalar_color,
#                                                      self.gui.edge_thresholder.value,
#                                                      self.gui.usage_thresholder.value,
#                                                      self.gui.speed_thresholder.value)
