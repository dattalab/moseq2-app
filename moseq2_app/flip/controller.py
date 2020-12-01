'''

Interactive Flip classifier frame selection functionality. This module utilizes the widgets from
the widgets.py file to facilitate the real-time interaction.

'''

import re
import cv2
import h5py
import joblib
import warnings
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
import ipywidgets as widgets
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from os.path import dirname, join, exists
from IPython.display import display, clear_output
from moseq2_app.roi.view import bokeh_plot_helper
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from moseq2_extract.extract.proc import clean_frames
from moseq2_app.gui.progress import get_session_paths
from moseq2_app.flip.widgets import FlipClassifierWidgets
from moseq2_extract.util import recursive_find_h5s, h5_to_dict

class FlipRangeTool(FlipClassifierWidgets):

    def __init__(self, input_dir, max_frames, output_file, tail_filter_iters, prefilter_kernel_size, continuous_slider_update):
        '''

        Initialization for the Flip Classifier Training tool.
         Finds all the extracted sessions within the given input path, and opens their h5
         files, storing their references to read frames from one by one at display time.

        Parameters
        ----------
        input_dir (str): Path to base directory containing extraction session folders
        max_frames (int): Maximum number of frames to include in the dataset.
        output_file (str): Path to save the outputted flip classifier.
        tail_filter_iters (int): Number of tail filtering iterations
        prefilter_kernel_size (int): Size of the median spatial filter.
        continuous_slider_update (bool): Indicates whether to continuously update the view upon slider edits.
        '''

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            super().__init__(continuous_update=continuous_slider_update)

            # User input parameters
            self.input_dir = input_dir
            self.max_frames = max_frames
            self.output_file = output_file

            # initialize frame cleaning parameter dict
            self.clean_parameters = {
                'iters_tail': tail_filter_iters,
                'strel_tail': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                'prefilter_space': (prefilter_kernel_size,)
            }

            # get input session paths
            self.sessions = get_session_paths(input_dir, extracted=True)
            if len(self.sessions) == 0:
                if 'aggregate_results/' not in input_dir:
                    found_agg = exists(join(input_dir, 'aggregate_results/'))
                    if found_agg:
                        self.input_dir = join(input_dir, 'aggregate_results/')
                        print(f'Loading data from: {self.input_dir}')
                        self.sessions = get_session_paths(self.input_dir, extracted=True)

                    if not found_agg or len(self.sessions) == 0:
                        print('Error: No extracted sessions were found.')

            # open h5 files and get reference dict
            self.data_dict, self.path_dict = self.load_sessions()

            # initialize selected frame range dictionary
            self.selected_frame_ranges_dict = {k: [] for k in self.data_dict}
            self.curr_total_selected_frames = 0

            # observe dropdown value changes
            self.session_select_dropdown.observe(self.changed_selected_session, names='value')
            self.session_select_dropdown.options = self.path_dict

            # Widget values
            self.frame_ranges = []
            self.display_frame_ranges = []

            self.start = self.frame_num_slider.value
            self.stop = 0

            self.selected_ranges.options = self.frame_ranges

            # Callbacks
            self.clear_button.on_click(self.clear_on_click)
            self.start_button.on_click(self.start_stop_frame_range)
            self.frame_num_slider.observe(self.curr_frame_update, names='value')

    def changed_selected_session(self, event):
        '''
        Callback function to load newly selected session.

        Parameters
        ----------
        event (ipywidgets Event): self.session_select_dropdown.value is changed

        Returns
        -------
        '''

        # check if button is in middle range selection
        if self.start_button.description == 'End Range':
            self.start_button.description = 'Start Range'
            self.start_button.button_style = 'info'

            self.start, self.stop = 0, 0

        # if so reset the button and start stop values
        self.frame_num_slider.max = self.data_dict[self.session_select_dropdown.label].shape[0] - 1
        self.frame_num_slider.value = 0
        clear_output(wait=True)
        self.interactive_launch_frame_selector()

    def curr_frame_update(self, event):
        '''
        Updates the currently displayed frame when the slider is moved.

        Parameters
        ----------
        event (ipywidgets Event): self.frame_num_slider.value is changed.

        Returns
        -------
        '''

        self.frame_num_slider.value = event['new']
        clear_output(wait=True)
        self.interactive_launch_frame_selector()

    def update_state_on_selected_range(self):
        '''
        Helper function that updates the view upon a correct frame range addition (stop > start).
         Callback function to update the table of selected frame ranges upon
         button click. Function will will add the selected ranges to the table
          and session dict to train the model downstream.

        Returns
        -------
        '''

        # Updating list of displayed session + selected frame ranges
        selected_range = range(self.start, self.stop)
        display_selected_range = f'{self.session_select_dropdown.label} - {selected_range}'
        self.curr_total_selected_frames += len(selected_range)

        # Update the current frame selector indicator
        old_lbl = self.curr_total_label.value
        old_val = re.findall(r': \d+', old_lbl)[0]
        new_val = old_lbl.replace(old_val, f': {str(self.curr_total_selected_frames)}')

        # Change indicator color to green if number of total selected
        # frames exceeds selected max number of frames
        if self.curr_total_selected_frames >= self.max_frames:
            new_val = f'<center><h4><font color="green";>{new_val}</h4></center>'
        self.curr_total_label.value = new_val

        # appending session list to get frames from for the flip classifier later on
        if selected_range not in self.selected_frame_ranges_dict[self.session_select_dropdown.label]:
            self.selected_frame_ranges_dict[self.session_select_dropdown.label] += [selected_range]

        # appending to frame ranges to display in table
        self.frame_ranges.append(selected_range)
        self.display_frame_ranges.append(display_selected_range)
        self.selected_ranges.options = self.display_frame_ranges

    def start_stop_frame_range(self, b):
        '''
        Callback function that triggers the "Add Range" functionality.
         If user clicks the button == 'Start Range', then the function will start including frames
         in the correct flip set. Else, it will end the included range and truncate the slider range
         from the start index.

        Parameters
        ----------
        b (button click): User clicks on "Start or Stop" Range button.

        Returns
        -------
        '''

        if self.start_button.description == 'Start Range':
            self.start_button.button_style = 'success'
            self.start = self.frame_num_slider.value
            self.start_button.description = 'End Range'
        elif self.start_button.description == 'End Range':
            self.stop = self.frame_num_slider.value
            if self.stop > self.start:
                self.update_state_on_selected_range()

            # Update button based on current selection state
            self.start_button.description = 'Start Range'
            self.start_button.button_style = 'info'

    def clear_on_click(self, b):
        '''
        Clears the output.

        Parameters
        ----------
        b (button click)

        Returns
        -------
        '''

        clear_output()

    def load_sessions(self):
        '''
        Recursively searches for completed h5 extraction files, and loads total_frames=max_frames to
         include in the total dataset. Additionally applies some image filtering prior to returning the data.

        Parameters
        ----------

        Returns
        -------
        data_dict (dict): dict of
        path_dict (dict): dict of session names and paths filtered for sessions missing an h5 or mp4 files.
        '''

        path_dict = {}

        # Get h5 paths
        for key, path in self.sessions.items():
            h5_path = join(dirname(path), 'proc/', 'results_00.h5')
            if exists(h5_path):
                path_dict[key] = h5_path
            elif exists(path) and path.endswith('.h5'):
                path_dict[key] = path
            elif exists(path) and path.endswith('.mp4') and exists(path.replace('.mp4', '.h5')):
                path_dict[key] = path.replace('.mp4', '.h5')
            else:
                del path_dict[key]

        # Get references to h5 files
        data_dict = {}
        for key, path in path_dict.items():
            dset = h5py.File(path, mode='r')['frames']
            data_dict[key] = dset

        return data_dict, path_dict

    def interactive_launch_frame_selector(self):
        '''

        Interactive tool that displays the frame to display with the selected data box.
        Users will use the start_range button to add frame ranges to the range box list.

        Returns
        -------
        '''

        tools = 'pan, box_zoom, wheel_zoom, reset'

        num = self.frame_num_slider.value

        # plot current frame
        bg_fig = figure(title=f"Current Frame: {num}",
                        tools=tools,
                        output_backend="webgl")

        displayed_frame = clean_frames(np.array([self.data_dict[self.session_select_dropdown.label][num]]), **self.clean_parameters)[0]

        data = dict(image=[displayed_frame],
                    x=[0],
                    y=[0],
                    dw=[displayed_frame.shape[1]],
                    dh=[displayed_frame.shape[0]])

        bokeh_plot_helper(bg_fig, data)

        output = widgets.Output(layout=widgets.Layout(align_items='center', height='600px'))
        with output:
            show(bg_fig)

        output_box = widgets.HBox([output, self.range_box])

        # Display centered grid plot
        display(self.clear_button, self.session_select_dropdown, output_box, self.button_box)

    def get_corrected_data(self):
        '''
        Apply the selected flip orientation ranges to the entire dataset to correct the
         incorrectly oriented frames.

        Returns
        -------
        '''

        corrected_dataset = []
        # Get corrected frame ranges
        for session, frs in tqdm(self.selected_frame_ranges_dict.items(), desc='Computing Corrected Dataset'):
            if len(frs) > 0:
                flips = frs
                # remove frames possibly selected twice
                correct_idx = sorted(set(np.concatenate([list(flip) for flip in flips])))

                # get the session
                cleaned_data = clean_frames(self.data_dict[session][correct_idx], **self.clean_parameters)

                # Get list of frame indices where the mouse is facing east
                flip_idx = np.setdiff1d(np.arange(cleaned_data.shape[0]), correct_idx)

                corrected_data = deepcopy(cleaned_data)
                corrected_data[flip_idx] = np.flip(corrected_data[flip_idx], axis=2)
                corrected_dataset.append(corrected_data)

        self.corrected_dataset = np.concatenate(corrected_dataset, axis=0)

    def plot_xy_examples(self, data_xflip, data_yflip, data_xyflip, selected_frame=0):
        '''
        Plots 2 columns of examples for the correct and incorrect examples being used to train
         the flip classifier.

         Inputted 3D array shapes are all as follows: (nframes x nrows x ncols)

        Parameters
        ----------
        data_xflip (3D np.ndarray): Single frame of the corrected dataset flipped on the x-axis (class 1)
        data_yflip (3D np.ndarray): Single frame of the corrected dataset flipped on the x-axis (class 0)
        data_xyflip (3D np.ndarray): Single frame of the corrected dataset flipped on the x and y-axis (class 1)

        Returns
        -------
        '''

        cols = ['0 - Correctly Flipped (Facing East)', '1 - Incorrectly Flipped (Facing West)']
        rows = ['', 'y-flipped']

        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

        pad = 5

        for ax, col in zip(axes[0], cols):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')

        for ax, row in zip(axes[:, 0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')

        axes[0, 0].imshow(self.corrected_dataset[selected_frame], vmin=0, vmax=80)
        axes[1, 0].imshow(data_yflip[selected_frame], vmin=0, vmax=80)
        axes[0, 1].imshow(data_xflip[selected_frame], vmin=0, vmax=80)
        axes[1, 1].imshow(data_xyflip[selected_frame], vmin=0, vmax=80)

        fig.tight_layout()

    def augment_dataset(self):
        '''
        Augments the selected correct dataset with 3 rotated versions of the truth values:
         1. xflip -> incorrect case; 2. yflip -> correct case; 3. xyflip -> incorrect case;
         and creates the X and Y train/test sets.
          The first half of X contains the incorrect cases (1), and the second half contains the correct cases (0).
          Equivalently, the first half of the y variable is composed of 1s, and the latter half is composed of 0s.

        Returns
        -------
        '''

        # Get flipped data
        data_xflip = np.flip(self.corrected_dataset, axis=2)
        data_yflip = np.flip(self.corrected_dataset, axis=1)
        data_xyflip = np.flip(data_yflip, axis=2)

        # Pack X and y training sets
        npixels = self.corrected_dataset.shape[1] * self.corrected_dataset.shape[2]
        ntrials = self.corrected_dataset.shape[0]

        self.x = np.vstack((data_xflip.reshape((-1, npixels)), data_xyflip.reshape((-1, npixels)),
                       data_yflip.reshape((-1, npixels)), self.corrected_dataset.reshape((-1, npixels))))

        # Plot examples of class 0: correctly flipped, and class 1: incorrectly flipped
        self.plot_xy_examples(data_xflip, data_yflip, data_xyflip, selected_frame=0)

        # class 1 is x facing west
        self.y = np.concatenate((np.ones((ntrials * 2,)), np.zeros((ntrials * 2,))))

    def prepare_datasets(self, test_size, random_state=0):
        '''
        Correct data after the appropriate flip ranges have been selected, augment and create X,y training sets,
         and split the data to training and testing splits.

        Parameters
        ----------
        test_size (int): Test dataset percent split size
        random_state (int): Seed value to randomly sort the split data

        Returns
        -------
        '''

        # Correct flips
        self.get_corrected_data()

        # Augment the data
        self.augment_dataset()

        # Split the datasets into Train and Test Sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,
                                                                                self.y,
                                                                                test_size=test_size/100,
                                                                                random_state=random_state)

    def train_and_evaluate_model(self,
                                 n_estimators=100,
                                 criterion="gini",
                                 n_jobs=4,
                                 max_depth=6,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 oob_score=False,
                                 random_state=0,
                                 verbose=0):
        '''

        Trains the flip classifier the pre-augmented dataset given some optionally adjustable
         model initialization parameters.

        Parameters
        ----------
        n_estimators (int): The number of trees in the forest.
        criterion (str): The function to measure the quality of a split. ['gini', mse', 'mae']
        n_jobs (int): The number of jobs to run in parallel for both `fit` and `predict`.
        max_depth (int): The maximum depth of the tree. If None, then nodes are expanded until
         all leaves are pure. (This will use a lot of memory, and may take a while.)
        min_samples_split (int): The minimum number of samples required to split an internal node.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        oob_score (bool): whether to use out-of-bag samples to estimate the R^2 on unseen data.
        random_state (int): The seed used by the random number generator.
        verbose (int): Controls the verbosity when fitting and predicting.

        Returns
        -------
        '''

        # Flip Classifier Model to train
        self.clf = RandomForestClassifier(n_estimators=n_estimators,
                                          criterion=criterion,
                                          min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf,
                                          oob_score=oob_score,
                                          max_depth=max_depth,
                                          random_state=random_state,
                                          n_jobs=n_jobs,
                                          verbose=verbose)

        self.clf.fit(self.x_train, self.y_train)

        y_predict = self.clf.predict(self.x_test)

        percent_correct = np.mean(self.y_test == y_predict) * 1e2
        print('Performance: {0:3f}% correct'.format(percent_correct))

        if percent_correct > 90:
            print('You have achieved acceptable model accuracy.')
            print('Re-extract the data with the newly saved model, and continue to the PCA step.')
        else:
            print('Model performance is not high enough to extract a valid dataset. '
                  'Either try selecting more accurate frame indices,\n'
                  'or re-extract the data with your latest flip classifier (with highest accuracy) '
                  'and retry selecting frame ranges with less random flips.')

        joblib.dump(self.clf, self.output_file)
        print(f'Saved model in {self.output_file}')