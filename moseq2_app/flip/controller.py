"""
Interactive Flip classifier frame selection functionality.
"""

import cv2
import h5py
import joblib
import warnings
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
import ipywidgets as widgets
import matplotlib.pyplot as plt
from collections import OrderedDict
from IPython.display import display
from bokeh.plotting import figure, show
from os.path import dirname, join, exists
from moseq2_app.roi.view import bokeh_plot_helper
from moseq2_extract.util import gen_batch_sequence
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from moseq2_app.gui.progress import get_session_paths
from moseq2_extract.io.video import write_frames_preview
from moseq2_app.flip.widgets import FlipClassifierWidgets
from moseq2_extract.extract.proc import clean_frames, get_flips

class FlipRangeTool(FlipClassifierWidgets):

    def __init__(self, input_dir, max_frames, output_file, clean_parameters,
                 launch_gui=True, continuous_slider_update=True):
        """
        Find all the extracted sessions within the given input path, and prepare for GUI display.

        Args:
        input_dir (str): Path to base directory containing extraction session folders
        max_frames (int): Maximum number of frames to include in the dataset.
        output_file (str): Path to save the outputted flip classifier.
        clean_parameters (dict): Parameters passed to moseq2_extract.extract.proc.clean_frames 
        launch_gui (bool): Indicates whether to launch the labeling gui or just create the FlipClassifier instance.
        continuous_slider_update (bool): Indicates whether to continuously update the view upon slider edits.
        
        """

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            # User input parameters
            self.input_dir = input_dir
            self.output_file = output_file
            self.clf = None

            # get input session paths
            self.sessions = get_session_paths(input_dir, extracted=True, flipped=False)
            if len(self.sessions) == 0:
                if 'aggregate_results/' not in input_dir:
                    found_agg = exists(join(input_dir, 'aggregate_results/'))
                    if found_agg:
                        self.input_dir = join(input_dir, 'aggregate_results/')
                        print(f'Loading data from: {self.input_dir}')
                        self.sessions = get_session_paths(self.input_dir, extracted=True, flipped=False)

                    if not found_agg or len(self.sessions) == 0:
                        print('Error: No extracted sessions were found.')

            # open h5 files and get reference dict
            self.path_dict = self.load_sessions()

            # initialize widgets and their callbacks.
            # passing additional state variables to consolidate
            super().__init__(path_dict=self.path_dict,
                             max_frames=max_frames,
                             continuous_update=continuous_slider_update,
                             launch_gui=launch_gui)

            # initialize frame cleaning parameter dict
            self.clean_parameters = clean_parameters

    def load_sessions(self):
        """
        Recursively searche for completed h5 extraction files, and loads total_frames=max_frames to include in the total dataset.

        Args:

        Returns:
        path_dict (dict): dict of session names and paths filtered for sessions missing an h5 or mp4 files.
        """

        path_dict = OrderedDict()

        # Get h5 paths
        for key, path in self.sessions.items():
            h5_path = join(dirname(path), 'proc/', 'results_00.h5')
            if '_flipped' not in key:
                if exists(h5_path):
                    path_dict[key] = h5_path
                elif exists(path) and path.endswith('.h5'):
                    path_dict[key] = path
                elif exists(path) and path.endswith('.mp4') and exists(path.replace('.mp4', '.h5')):
                    path_dict[key] = path.replace('.mp4', '.h5')
                else:
                    del path_dict[key]

        return path_dict

    def interactive_launch_frame_selector(self):
        """
        display the frame to display with the selected data box.
        """

        tools = 'pan, box_zoom, wheel_zoom, reset'

        num = self.frame_num_slider.value

        # plot current frame
        bg_fig = figure(title=f"Current Frame: {num}",
                        tools=tools,
                        output_backend="webgl")

        with h5py.File(self.path_dict[self.session_select_dropdown.label], mode='r') as f:
            displayed_frame = clean_frames(np.array([f['frames'][num]]), **self.clean_parameters)[0]

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
        """
        Apply the selected flip orientation ranges to the entire dataset to correct the incorrectly oriented frames.
        """

        corrected_dataset = []
        # Get corrected frame ranges
        for session, frs in tqdm(self.selected_frame_ranges_dict.items(), desc='Computing Corrected Dataset'):
            if len(frs) > 0:
                # get the indicated directions
                directions = [f[0] for f in frs]
                
                # get the separate lists of flip ranges to correct
                correct_flips = [f[1] for f, d in zip(frs, directions) if d is False]
                incorrect_flips = [f[1] for f, d in zip(frs, directions) if d is True]
                
                # handle frames that are indicated as correctly flipped
                if len(correct_flips) > 0:
                    # remove frames possibly selected twice
                    correct_idx = sorted(set(np.concatenate([list(flip) for flip in correct_flips])))

                    # get the session and load only the selected frame range and apply filtering
                    with h5py.File(self.path_dict[session], mode='r') as f:
                        correct_cleaned_data = clean_frames(f['frames'][correct_idx], **self.clean_parameters)
                    
                    # add the data to the dataset
                    corrected_dataset.append(deepcopy(correct_cleaned_data))
                
                # handle frames indicated incorrectly flipped
                if len(incorrect_flips) > 0:
                    incorrect_idx = sorted(set(np.concatenate([list(flip) for flip in incorrect_flips])))
                    with h5py.File(self.path_dict[session], mode='r') as f:
                        incorrect_cleaned_data = clean_frames(f['frames'][incorrect_idx], **self.clean_parameters)
                    
                    # flip the data that is facing left
                    flip_corrected_data = np.flip(incorrect_cleaned_data, axis=2)

                    # add the data to the dataset
                    corrected_dataset.append(deepcopy(flip_corrected_data))

        self.corrected_dataset = np.concatenate(corrected_dataset, axis=0)

    def plot_xy_examples(self, data_xflip, data_yflip, data_xyflip, selected_frame=0):
        """
        Plots 2 columns of examples for the correct and incorrect examples being used to train the flip classifier.

        Args:
        data_xflip (np.ndarray): Single frame of the corrected dataset flipped on the x-axis (class 1)
        data_yflip (np.ndarray): Single frame of the corrected dataset flipped on the x-axis (class 0)
        data_xyflip (np.ndarray): Single frame of the corrected dataset flipped on the x and y-axis (class 1)

        Returns:
        """

        cols = ['0 - Correctly oriented (Facing Right)', '1- Incorrectly oriented (Facing Left)']
        rows = ['', 'x-flipped']

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

    def augment_dataset(self, plot_examples=False):
        """
        Augment the selected correct dataset with 3 rotated versions of the truth values.

        Args:
        plot_examples (bool): Indicates whether to display the 2x2 preview grid of dataset examples.
        Returns:
        """

        # Get flipped data
        # 1. xflip -> incorrect case; 2. yflip -> correct case; 3. xyflip -> incorrect case;
        data_xflip = np.flip(self.corrected_dataset, axis=2)
        data_yflip = np.flip(self.corrected_dataset, axis=1)
        data_xyflip = np.flip(data_yflip, axis=2)

        # Pack X and y training sets
        npixels = self.corrected_dataset.shape[1] * self.corrected_dataset.shape[2]
        ntrials = self.corrected_dataset.shape[0]

        self.x = np.vstack((data_xflip.reshape((-1, npixels)), data_xyflip.reshape((-1, npixels)),
                       data_yflip.reshape((-1, npixels)), self.corrected_dataset.reshape((-1, npixels))))

        if plot_examples:
            # Plot examples of class 0: correctly flipped, and class 1: incorrectly flipped
            self.plot_xy_examples(data_xflip, data_yflip, data_xyflip, selected_frame=0)

        # class 1 is x facing west
        self.y = np.concatenate((np.ones((ntrials * 2,)), np.zeros((ntrials * 2,))))

    def prepare_datasets(self, test_size, random_state=0, plot_examples=False):
        """
        correct data with user input, augment and create X,y training sets, and split the data to training and testing splits.

        Args:
        test_size (int): Test dataset percent split size
        random_state (int): Seed value to randomly sort the split data
        plot_examples (bool): Indicates whether to display the 2x2 preview grid of dataset examples
        """

        # Correct flips
        self.get_corrected_data()

        # Augment the data
        self.augment_dataset(plot_examples=plot_examples)

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
                                 verbose=0,
                                 train=True):
        """
        Train the flip classifier the pre-augmented dataset given some optionally adjustable model initialization parameters.

        Args:
        n_estimators (int): The number of trees in the forest.
        criterion (str): The function to measure the quality of a split. ['gini', mse', 'mae']
        n_jobs (int): The number of jobs to run in parallel for both fit and predict.
        max_depth (int): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        oob_score (bool): whether to use out-of-bag samples to estimate the R^2 on unseen data.
        random_state (int): The seed used by the random number generator.
        verbose (int): Controls the verbosity when fitting and predicting.
        train (bool): If True, trains or retrains a model, if False only tests the model on the test set.
        """

        if not exists(self.output_file):
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
        else:
            print('Loading pre-existing flip classifier')
            try:
                self.clf = joblib.load(self.output_file)
            except Exception as e:
                print(f'Error could not load existing flip classifier: {e}')
                print('Creating and training a new flip classifier.')
                train = True
                self.clf = RandomForestClassifier(n_estimators=n_estimators,
                                                  criterion=criterion,
                                                  min_samples_split=min_samples_split,
                                                  flip_indicesmin_samples_leaf=min_samples_leaf,
                                                  oob_score=oob_score,
                                                  max_depth=max_depth,
                                                  random_state=random_state,
                                                  n_jobs=n_jobs,
                                                  verbose=verbose)

        if train:
            self.clf.fit(self.x_train, self.y_train)

        # get model test set predictions
        y_predict = self.clf.predict(self.x_test)

        # compute performace
        percent_correct = np.mean(self.y_test == y_predict) * 1e2
        print('Performance: {0:3f}% correct'.format(percent_correct))

        if percent_correct > 90:
            print('You have achieved acceptable model accuracy.')
            print('Correct the extracted data in the next cell using your new model, and continue to the PCA step.')
        else:
            print('Model performance is not high enough to extract a valid dataset. '
                  'Either try selecting more accurate frame indices,\n'
                  'or re-extract the data with your latest flip classifier (with highest accuracy) '
                  'and retry selecting frame ranges with less random flips.')

        # save model
        joblib.dump(self.clf, self.output_file)
        print(f'Saved model in {self.output_file}')

    def apply_flip_classifier(self, chunk_size=4000, chunk_overlap=0,
                              smoothing=51, frame_path='frames', fps=30,
                              write_movie=False, verbose=True):
        """
        Apply a trained flip classifier on previously extracted data to flip the mice to the correct orientation.

        Args:
        chunk_size (int): size of frame chunks to process in batches.
        chunk_overlap (int): number of frames to overlap between chunks to improve classification precision between chunks.
        smoothing (int): kernel size of the applied median filter on the flip classifier results
        verbose (bool): displays the tqdm progress bars for each session.
        """

        if self.clf is None:
            try:
                joblib.load(self.output_file)
            except Exception as e:
                print('Could not load provided classifier.')
                return

        video_pipe = None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            for key, path in tqdm(self.path_dict.items(), desc='Flipping extracted sessions...'):
                # Open h5 file to stream and correct/update stored frames and scalar angles.
                with h5py.File(path, mode='a') as f:
                    output_movie = path.replace('.h5', '_flipped.mp4')

                    frames = f[frame_path]
                    frame_batches = gen_batch_sequence(len(frames)-1, chunk_size, chunk_overlap)

                    for batch in tqdm(frame_batches, desc=f'Adjusting flips: {key}', disable=not verbose):
                        frame_batch = frames[batch]

                        # apply flip classifier on each batch to find which frames to flip
                        flips = get_flips(frame_batch.copy(), flip_file=self.output_file, smoothing=smoothing)
                        flip_indices = np.where(flips)

                        # rewrite the frames with the newly classified orientation
                        frame_batch[flip_indices] = np.rot90(f[frame_path][batch][flip_indices], k=2, axes=(1, 2))
                        f[frame_path][batch] = frame_batch

                        # augment recorded scalar value to reflect orientation switches
                        f['scalars/angle'][flip_indices] += np.pi

                        if write_movie:
                            try:
                                # Writing frame batch to mp4 file
                                video_pipe = write_frames_preview(output_movie,
                                                                  frame_batch,
                                                                  pipe=video_pipe,
                                                                  close_pipe=False,
                                                                  depth_min=0,
                                                                  depth_max=100,
                                                                  fps=fps,
                                                                  progress_bar=verbose)
                            except AttributeError as e:
                                warnings.warn(f'Could not generate flipped movie for {key}:{path}. Skipping...')
                                print(e)
                                print(e.__traceback__)
                                break
                    # Check if video is done writing. If not, wait.
                    if video_pipe is not None:
                        video_pipe.communicate()
                        video_pipe = None
