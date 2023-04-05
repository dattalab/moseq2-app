# import os
# import shutil
# import bokeh.io
# import numpy as np
# import pandas as pd
# from copy import deepcopy
# from os.path import exists
# from unittest import TestCase
# from moseq2_app.main import label_syllables
# from moseq2_viz.model.util import parse_model_results
# from moseq2_app.viz.controller import SyllableLabeler, CrowdMovieComparison

# class TestSyllableLabeler(TestCase):

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

#         self.model_fit = parse_model_results(self.progress_paths['model_path'])
#         self.gui = SyllableLabeler(self.model_fit,
#                                    self.progress_paths['model_path'],
#                                    self.progress_paths['index_file'],
#                                    self.progress_paths['config_file'],
#                                    6, # max syll
#                                    False, # not choose the instances closer to median
#                                    20, # set number of examples
#                                    self.progress_paths['crowd_dir'],
#                                    self.progress_paths['syll_info'])

#     def tearDown(self):
#         if exists('data/crowd_movies/'):
#             shutil.rmtree('data/crowd_movies/')

#         del self.gui


#     @classmethod
#     def tearDownClass(cls):
#         if exists('data/syll_info.yaml'):
#             os.remove('data/syll_info.yaml')

#         if exists('data/syll_df.parquet'):
#             os.remove('data/syll_df.parquet')

#         if exists('data/moseq_scalar_dataframe.parquet'):
#             os.remove('data/moseq_scalar_dataframe.parquet')

#         if exists('data/crowd_movies/'):
#             shutil.rmtree('data/crowd_movies/')

#     # Syllable Labeler Tests
#     def test_labeler_init(self):

#         # asserting that the input arguments into the init function were properly set.
#         # the following files in the assertions are newly generated
#         assert len(self.gui.syll_info) == 6
#         assert exists('data/syll_info.yaml')
#         assert exists('data/syll_df.parquet')
#         assert exists('data/moseq_scalar_dataframe.parquet')

#         # asserting that upon reloading pre-existing files with max_syllables, the correct number of
#         # syllables are returned.
#         self.gui = SyllableLabeler(self.model_fit,
#                                    self.progress_paths['model_path'],
#                                    self.progress_paths['index_file'],
#                                    self.progress_paths['config_file'],
#                                    5, # max syll
#                                    False, # not choose the instances closer to median
#                                    20, # set number of examples
#                                    self.progress_paths['crowd_dir'],
#                                    self.progress_paths['syll_info'])

#         assert len(self.gui.syll_info) == 5
#         assert exists('data/syll_info.yaml')
#         assert exists('data/syll_df.parquet')
#         assert exists('data/moseq_scalar_dataframe.parquet')

#         os.remove('data/syll_info.yaml')

#         # asserting that the syll_info.yaml file will be regenerated if it didn't exist
#         self.gui = SyllableLabeler(self.model_fit,
#                                    self.progress_paths['model_path'],
#                                    self.progress_paths['index_file'],
#                                    self.progress_paths['config_file'],
#                                    5, # max syll
#                                    False, # not choose the instances closer to median
#                                    20, # set number of examples
#                                    self.progress_paths['crowd_dir'],
#                                    self.progress_paths['syll_info'])

#         # assert that file is being regenerated
#         assert exists('data/syll_info.yaml')
#         assert len(list(self.gui.syll_select.options.keys())) == len(list(range(self.gui.max_sylls)))

#     def test_clear_on_click(self):

#         self.gui.clear_on_click()

#     def test_write_syll_info(self):

#         if exists('data/syll_info.yaml'):
#             os.remove('data/syll_info.yaml')

#         self.gui.write_syll_info()
#         assert exists('data/syll_info.yaml')

#     def test_on_next(self):

#         curr_idx = self.gui.syll_select.index
#         self.gui.lbl_name_input.value = 'test'
#         self.gui.desc_input.value = 'test desc'
#         self.gui.on_next()

#         assert self.gui.syll_select.index == curr_idx+1
#         assert self.gui.syll_info[curr_idx]['label'] == 'test'
#         assert self.gui.syll_info[curr_idx]['desc'] == 'test desc'

#         assert self.gui.lbl_name_input.value == ''
#         assert self.gui.desc_input.value == ''

#     def test_on_prev(self):

#         curr_idx = self.gui.syll_select.index
#         self.gui.lbl_name_input.value = 'test'
#         self.gui.desc_input.value = 'test desc'
#         self.gui.on_prev()

#         assert self.gui.syll_select.index == self.gui.max_sylls-1
#         assert self.gui.syll_info[curr_idx]['label'] == 'test'
#         assert self.gui.syll_info[curr_idx]['desc'] == 'test desc'

#         assert self.gui.lbl_name_input.value == ''
#         assert self.gui.desc_input.value == ''

#     def test_on_set(self):

#         curr_idx = self.gui.syll_select.index
#         self.gui.lbl_name_input.value = 'test'
#         self.gui.desc_input.value = 'test desc'
#         self.gui.on_set()

#         assert self.gui.syll_select.index == curr_idx
#         assert self.gui.syll_info[curr_idx]['label'] == 'test'
#         assert self.gui.syll_info[curr_idx]['desc'] == 'test desc'

#         assert self.gui.set_button.button_style == 'success'

#     def test_get_mean_group_dict(self):
#         df = pd.read_parquet(self.gui.df_output_file, engine='fastparquet')

#         group_df = df.groupby(['group', 'syllable'], as_index=False).mean()

#         tmp_group_syll_info = deepcopy(self.gui.syll_info)

#         self.gui.get_mean_group_dict(group_df)

#         assert self.gui.group_syll_info != tmp_group_syll_info

#     def test_get_mean_syllable_info(self):

#         self.gui.get_mean_syllable_info()

#         if exists('data/syll_df.parquet'):
#             os.remove('data/syll_df.parquet')

#         self.gui.get_mean_syllable_info()

#     def test_set_group_info_widgets(self):

#         self.gui.set_group_info_widgets(self.gui.group_syll_info[self.gui.syll_select.index]['group_info'])

#         assert len(self.gui.info_boxes.children) == 2

#     def test_interactive_syllable_labeler(self):

#         self.gui.interactive_syllable_labeler(self.gui.syll_info[0])
#         assert self.gui.cm_lbl.text == f'Crowd Movie {self.gui.syll_select.index + 1}/{len(self.gui.syll_select.options)}'

#     def test_set_default_cm_parameters(self):
#         self.gui.config_data['max_syllable'] = 10

#         config_data = self.gui.set_default_cm_parameters(self.gui.config_data)

#         assert config_data['max_syllable'] == 6
#         assert config_data['raw_size'] == (512, 424)
#         assert config_data['max_dur'] == 60
#         assert config_data['min_dur'] == 3

#     def test_get_crowd_movie_paths(self):

#         self.gui.get_crowd_movie_paths(self.progress_paths['index_file'],
#                                        self.progress_paths['model_path'],
#                                        self.gui.config_data,
#                                        self.progress_paths['crowd_dir'])

#         assert exists(self.progress_paths['crowd_dir'])
#         assert len([f for f in os.listdir(self.progress_paths['crowd_dir']) if f.endswith('.mp4')]) == 6

# class TestCrowdMovieComparison(TestCase):

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

#         label_syllables(progress_paths, max_syllables=6)

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

#         self.grouped_data_dir = 'data/grouped_crowd_movies/'

#         self.gui = CrowdMovieComparison(self.progress_paths['config_file'],
#                                         self.progress_paths['index_file'],
#                                         self.progress_paths['df_info_path'],
#                                         self.progress_paths['model_path'],
#                                         self.progress_paths['syll_info'],
#                                         self.grouped_data_dir,
#                                         get_pdfs=True,
#                                         load_parquet=True)

#     ## Crowd Movie Comparison tests
#     def test_compare_init(self):

#         assert self.gui.sessions == ['66e77b85-f5fa-4e31-a61c-8952394ff441',
#                                      'ae8a9d45-7ad9-4048-963f-ca4931125fcd']

#         assert self.gui.cm_session_sel.options == ('012517',)

#         assert self.gui.config_data['separate_by'] == 'groups'
#         assert self.gui.config_data['gaussfilter_space'] == [0, 0]

#     def test_clear_on_click(self):
#         self.gui.clear_on_click()

#     def test_set_default_cm_parameters(self):

#         self.gui.set_default_cm_parameters()

#         assert self.gui.config_data['max_syllable'] == self.gui.max_sylls
#         assert self.gui.config_data['separate_by'] == 'groups'

#         assert self.gui.config_data['gaussfilter_space'] == [0, 0]
#         assert self.gui.config_data['medfilter_space'] == [0]

#     def test_show_session_select(self):

#         class Change:
#             def __init__(self, new='SessionName'):
#                 self.new = new

#         change = Change()
#         self.gui.show_session_select(change)
#         assert self.gui.config_data['separate_by'] == 'sessions'

#         change = Change('SubjectName')
#         self.gui.show_session_select(change)
#         assert self.gui.config_data['separate_by'] == 'subjects'

#         change = Change('group')
#         self.gui.show_session_select(change)
#         assert self.gui.config_data['separate_by'] == 'groups'

#     def test_select_session(self):

#         self.gui.select_session()

#         assert self.gui.config_data['session_names'] == list(self.gui.cm_session_sel.value)

#     def test_get_mean_group_dict(self):

#         df = pd.read_parquet(self.progress_paths['df_info_path'], engine='fastparquet')
#         group_df = df.groupby(['group', 'syllable'], as_index=False).mean()

#         self.gui.get_mean_group_dict(group_df)

#         # expecting a 6 key dictionary for data to pass to displayed table
#         assert len(self.gui.group_syll_info[0]['group_info']['default']) == 6

#     def test_get_session_mean_syllable_info_df(self):

#         self.gui.get_session_mean_syllable_info_df()

#         assert len(self.gui.df.syllable.unique()) == 11
#         assert self.gui.groups == ['default']
#         assert len(self.gui.df.columns) != len(self.gui.group_df.columns)

#     def test_get_selected_session_syllable_info(self):

#         self.gui.select_session()
#         self.gui.get_selected_session_syllable_info(self.gui.config_data['session_names'])

#     def test_get_pdf_plot(self):

#         df = pd.read_parquet(self.progress_paths['df_info_path'], engine='fastparquet')
#         group_df = df.groupby(['group', 'syllable'], as_index=False).mean()

#         self.gui.get_session_mean_syllable_info_df()
#         self.gui.get_mean_group_dict(group_df)

#         pdf_df = np.mean(self.gui.df['pdf'].values)

#         self.gui.get_pdf_plot(pdf_df, 'default')

#     def test_generate_crowd_movie_divs(self):

#         self.gui.grouped_syll_dict = self.gui.group_syll_info[0]['group_info']
#         divs, bk_plots = self.gui.generate_crowd_movie_divs(self.gui.grouped_syll_dict)
#         assert self.gui.curr_label == ''
#         assert self.gui.curr_desc == ''
#         assert len(divs) == 1
#         assert len(bk_plots) == 1

#     def test_on_click_trigger_button(self):

#         self.gui.cm_sources_dropdown.value = 'SessionName'

#         self.gui.grouped_syll_dict = self.gui.group_syll_info[0]['group_info']

#         self.gui.grouped_syll_dict = self.gui.group_syll_info[0]['group_info']
#         self.gui.on_click_trigger_button()

#     def test_crowd_movie_preview(self):

#         self.gui.crowd_movie_preview('0 - ', 'group', 2)

#         assert self.gui.widget_box is not None

#         self.gui.crowd_movie_preview('0 - ', 'sessions', 2)

#         assert self.gui.widget_box is not None