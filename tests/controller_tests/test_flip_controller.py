# import os
# import h5py
# import shutil
# import bokeh.io
# import numpy as np
# import ruamel.yaml as yaml
# from moseq2_viz.util import read_yaml
# from os.path import exists
# from unittest import TestCase
# from moseq2_app.flip.controller import FlipRangeTool
# from moseq2_extract.helpers.wrappers import extract_wrapper
# from moseq2_app.gui.progress import generate_missing_metadata

# class TestFlipController(TestCase):

#     @classmethod
#     def setUpClass(cls):
#         # generate sample metadata json for each session that is missing one
#         generate_missing_metadata('data/azure_test/', 'azure_test')

#         config_data = read_yaml('data/config.yaml')
#         clean_parameters = {'prefilter_space': config_data['spatial_filter_size'], # median filter kernel sizes 
#                             'prefilter_time': config_data['temporal_filter_size'], # temporal filter kernel sizes
#                             'strel_tail': strels['strel_tail'], # struc. element for filtering tail
#                             'iters_tail': config_data['tail_filter_iters'], # number of iters for morph. opening to filter tail
#                             'frame_dtype': config_data['frame_dtype'], # frame dtype
#                             'strel_min':strels['strel_min'], # structuring element for erosion
#                             'iters_min': config_data['cable_filter_iters']}# number of iterations for erosion

#         extract_wrapper('data/azure_test/nfov_test.mkv',
#                         None,
#                         config_data,
#                         num_frames=60,
#                         skip=True)
#         assert exists('data/azure_test/proc/results_00.mp4')

#         bokeh.io.output_notebook()

#     def setUp(self):
#         self.gui = FlipRangeTool(input_dir='data/',
#                                  max_frames=50,
#                                  output_file='data/output-classifier.pkl',
#                                  clean_parameters=clean_parameters,
#                                  launch_gui=True,
#                                  continuous_slider_update=True)

#     def tearDown(self):
#         del self.gui

#     @classmethod
#     def tearDownClass(cls):
#         shutil.rmtree('data/azure_test/proc/')

#     def test_init(self):

#         assert self.gui.sessions == {'azure_test': 'data/azure_test/proc/results_00.mp4'}

#         assert self.gui.start == self.gui.frame_num_slider.value
#         assert self.gui.stop == 0

#         assert self.gui.frame_ranges == []
#         assert self.gui.display_frame_ranges == []

#     def test_changed_selected_session(self):

#         self.gui.start_button.description = 'End Range'

#         self.gui.changed_selected_session()
#         assert self.gui.start_button.description == 'Start Range'
#         assert self.gui.start_button.button_style == 'info'

#     def test_on_selected_range_value(self):

#         initial_value = self.gui.delete_selection_button.layout.visibility

#         self.gui.on_selected_range_value()

#         new_value = self.gui.delete_selection_button.layout.visibility

#         assert initial_value != new_value
#         assert new_value == 'visible'

#     def test_on_delete_selection_clicked(self):

#         self.gui.selected_ranges.options = ['L - sessionName1 - range(100,200)',
#                                             'R - sessionName2 - range(200,500)',
#                                             'L - sessionName3 - range(600,700)']

#         self.gui.selected_frame_ranges_dict = {'sessionName1': [(True, range(100, 200))],
#                                                'sessionName2': [(True, range(200, 500))],
#                                                'sessionName3': [(True, range(600, 700))]}

#         self.gui.frame_ranges = [range(100, 200),
#                                  range(200, 500),
#                                  range(600, 700)]

#         self.gui.display_frame_ranges = list(self.gui.selected_ranges.options)

#         initial_value = self.gui.selected_ranges.value

#         self.gui.on_delete_selection_clicked()

#         new_value = 'R - 200-500 - sessionName2'

#         # updated current selected value
#         assert initial_value != new_value

#         # updated number of selected ranges, post-selection deletion
#         assert len(self.gui.selected_ranges.options) == 2

#         # number of total viewed sessions is unchanged
#         assert len(self.gui.selected_frame_ranges_dict.keys()) == 3

#         # selections within session keys have been updated
#         assert self.gui.selected_frame_ranges_dict == {'sessionName1': [],
#                                                        'sessionName2': [(True, range(200, 500))],
#                                                        'sessionName3': [(True, range(600, 700))]}


#         # updated number of frame ranges to display
#         assert len(self.gui.frame_ranges) == 2
#         assert len(list(self.gui.display_frame_ranges)) == 2

#     def test_curr_frame_update(self):
#         input_event = {'new': 1}

#         self.gui.curr_frame_update(input_event)

#         assert self.gui.frame_num_slider.value == 1

#     def test_facing_range_callback(self):

#         class Event:
#             description = 'left'

#         self.gui.stop = 30

#         event = Event()
#         self.gui.facing_range_callback(event)

#         assert self.gui.face_left_button.layout.visibility == 'hidden'
#         assert self.gui.face_right_button.layout.visibility == 'hidden'

#         assert self.gui.start_button.description == 'Start Range'
#         assert self.gui.start_button.button_style == 'info'

#     def test_update_state_on_selected_range(self):

#         left = True
#         self.gui.start, self.gui.stop = 0, 40
#         prev_indicator_value = self.gui.curr_total_label.value

#         self.gui.update_state_on_selected_range(left)
#         assert self.gui.curr_total_label.value != prev_indicator_value
#         assert self.gui.curr_total_label.value == '<center><h4><font color="black";>Current Total Selected Frames: 40</h4></center>'

#         assert self.gui.frame_ranges == [range(0, 40)]
#         assert self.gui.display_frame_ranges == ['L - azure_test - range(0, 40)']

#         left = False
#         self.gui.start, self.gui.stop = 41, 51
#         self.gui.update_state_on_selected_range(left)
#         assert self.gui.curr_total_label.value == '<center><h4><font color="green";>Current Total Selected Frames: 50</h4></center>'
#         assert self.gui.frame_ranges == [range(0, 40), range(41, 51)]
#         assert self.gui.display_frame_ranges == ['L - azure_test - range(0, 40)', 'R - azure_test - range(41, 51)']

#     def test_start_stop_frame_range(self):

#         self.gui.start_button.description = 'Start Range'
#         self.gui.face_left_button.layout.visibility = 'hidden'
#         self.gui.face_right_button.layout.visibility = 'hidden'

#         self.gui.start_stop_frame_range()

#         assert self.gui.start_button.description == 'Cancel Select'
#         assert self.gui.face_left_button.layout.visibility == 'visible'
#         assert self.gui.face_right_button.layout.visibility == 'visible'

#         self.gui.start_stop_frame_range()

#         assert self.gui.start_button.description == 'Start Range'
#         assert self.gui.face_left_button.layout.visibility == 'hidden'
#         assert self.gui.face_right_button.layout.visibility == 'hidden'

#     def test_load_sessions(self):
#         path_dict = self.gui.load_sessions()

#         with h5py.File(path_dict['azure_test'], mode='r') as f:
#             assert f['frames'].shape == (60, 80, 80)
#         assert path_dict['azure_test'] == 'data/azure_test/proc/results_00.h5'

#     def test_interactive_launch_frame_selector(self):

#         self.gui.interactive_launch_frame_selector()

#     def test_get_corrected_data(self):
#         left = True
#         self.gui.start, self.gui.stop = 0, 40
#         self.gui.update_state_on_selected_range(left)

#         left = False
#         self.gui.start, self.gui.stop = 41, 51
#         self.gui.update_state_on_selected_range(left)

#         self.gui.get_corrected_data()

#         assert self.gui.corrected_dataset.shape == (50, 80, 80)

#     def test_plot_xy_examples(self):

#         left = True
#         self.gui.start, self.gui.stop = 0, 40
#         self.gui.update_state_on_selected_range(left)

#         left = False
#         self.gui.start, self.gui.stop = 41, 51
#         self.gui.update_state_on_selected_range(left)

#         self.gui.get_corrected_data()

#         # Get flipped data
#         data_xflip = np.flip(self.gui.corrected_dataset, axis=2)
#         data_yflip = np.flip(self.gui.corrected_dataset, axis=1)
#         data_xyflip = np.flip(data_yflip, axis=2)

#         self.gui.plot_xy_examples(data_xflip, data_yflip, data_xyflip, selected_frame=0)

#     def test_augment_dataset(self):
#         left = True
#         self.gui.start, self.gui.stop = 0, 40
#         self.gui.update_state_on_selected_range(left)

#         left = False
#         self.gui.start, self.gui.stop = 41, 51
#         self.gui.update_state_on_selected_range(left)

#         self.gui.get_corrected_data()

#         self.gui.augment_dataset()

#         assert self.gui.x.shape == (200, 6400)
#         assert self.gui.y.shape == (200,)

#     def test_prepare_datasets(self):
#         left = True
#         self.gui.start, self.gui.stop = 0, 40
#         self.gui.update_state_on_selected_range(left)

#         left = False
#         self.gui.start, self.gui.stop = 41, 51
#         self.gui.update_state_on_selected_range(left)

#         self.gui.prepare_datasets(test_size=20)

#         assert self.gui.x_train.shape == (160, 6400)
#         assert self.gui.y_train.shape == (160,)

#         assert self.gui.x_test.shape == (40, 6400)
#         assert self.gui.y_test.shape == (40,)

#     def test_train_and_evaluate_model(self):

#         left = True
#         self.gui.start, self.gui.stop = 0, 40
#         self.gui.update_state_on_selected_range(left)

#         left = False
#         self.gui.start, self.gui.stop = 41, 51
#         self.gui.update_state_on_selected_range(left)

#         self.gui.prepare_datasets(test_size=20)

#         self.gui.train_and_evaluate_model()

#         assert self.gui.clf.criterion == 'gini'

#         y_predict = self.gui.clf.predict(self.gui.x_test)

#         percent_correct = np.mean(self.gui.y_test == y_predict) * 1e2

#         assert percent_correct == 45.0

#         assert exists(self.gui.output_file)
#         os.remove(self.gui.output_file)