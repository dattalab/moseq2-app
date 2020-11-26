import cv2
import h5py
import joblib
import warnings
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
import ipywidgets as widgets
from bokeh.plotting import figure, show
from IPython.display import display, clear_output
from moseq2_app.roi.view import bokeh_plot_helper
from moseq2_extract.util import recursive_find_h5s
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from moseq2_extract.extract.proc import clean_frames
from moseq2_app.flip.widgets import FlipClassifierWidgets

class FlipRangeTool(FlipClassifierWidgets):

    def __init__(self, input_dir, max_frames, output_file, tail_filter_iters, prefilter_kernel_size):
        '''

        Initialization for the Flip Classifier Training tool

        Parameters
        ----------
        input_dir (str): Path to base directory containing extraction session folders
        max_frames (int): Maximum number of frames to include in the dataset.
        output_file (str): Path to save the outputted flip classifier.
        tail_filter_iters (int): Number of tail filtering iterations
        prefilter_kernel_size (int): Size of the median spatial filter.
        '''

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            super().__init__()

            # User input parameters
            self.input_dir = input_dir
            self.max_frames = max_frames
            self.output_file = output_file
            self.cleaned_data = self.load_sessions(input_dir, max_frames, tail_filter_iters, prefilter_kernel_size)

            # Widget values
            self.frame_num_slider.max = int(max_frames) - 1
            self.frame_ranges = []

            self.start = self.frame_num_slider.value
            self.stop = 0

            # Flip Classifier Model to train
            self.clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1,
                                            max_depth=10, class_weight='balanced', warm_start=False)

            self.selected_ranges.options = self.frame_ranges

            # Callbacks
            self.clear_button.on_click(self.clear_on_click)
            self.start_button.on_click(self.start_stop_frame_range)

    def start_stop_frame_range(self, b):
        '''
        Callback function that triggers the "Add Range" functionality.
         If user clicks the button == 'Start Range', then the function will start including frames
         in the correct flip set. Else, it will end the included range and truncate the slider range
         from the start index.

        Parameters
        ----------
        b (button click)

        Returns
        -------
        '''

        if self.start_button.description == 'Start Range':
            self.start_button.button_style = 'success'
            self.start = self.frame_num_slider.value
            self.start_button.description = 'End Range'
        elif self.start_button.description == 'End Range':
            self.stop = self.frame_num_slider.value
            done = False
            if self.stop > self.start:
                self.frame_ranges.append(range(self.start, self.stop))
                self.selected_ranges.options = self.frame_ranges

            if not done:
                self.start_button.description = 'Start Range'
                self.start_button.button_style = 'info'
            else:
                self.start_button.description = 'Done'
                print('Dataset labeling complete. Clear the output and continue to model training.')

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

    def load_sessions(self, input_dir, max_frames=1e6, tail_filter_iters=1, space_filter_size=3):
        '''
        Recursively searches for completed h5 extraction files, and loads total_frames=max_frames to
         include in the total dataset. Additionally applies some image filtering prior to returning the data.

        Parameters
        ----------
        input_dir (str): Path to input directory
        max_frames (int): Maximum number of frames to included in loaded dataset.
        tail_filter_iters (int): Number of tail filtering iterations to run.
        space_filter_size (int): Size of median filtering kernel.

        Returns
        -------
        clean_merged_data (3D np.ndarray): Loaded and filtered data to get correctly oriented frame ranges from.
        '''

        h5s, dicts, _ = recursive_find_h5s(root_dir=input_dir)

        unique_uuids, unique_h5s = [], []

        for i, d in enumerate(dicts):
            if d['uuid'] not in unique_uuids:
                unique_uuids.append(d['uuid'])
                unique_h5s.append(h5s[i])

        data = []
        for h5 in tqdm(unique_h5s, desc='Loading Data', total=len(unique_h5s)):
            dset = h5py.File(h5, mode='r')['frames'][()]
            # TODO: potentially add crop size readjustment
            data.append(dset)

        merged_data = np.concatenate(data, axis=0)

        if merged_data.shape[0] > max_frames:
            merged_data = merged_data[np.random.choice(merged_data.shape[0], int(max_frames), replace=False)]

        clean_parameters = {
            'iters_tail': tail_filter_iters,
            'strel_tail': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            'prefilter_space': (space_filter_size,)
        }

        clean_merged_data = clean_frames(merged_data, **clean_parameters)

        return clean_merged_data

    def interactive_launch_frame_selector(self, num):
        '''

        Interactive tool that displays the frame to display with the selected data box.
        Users will use the start_range button to add frame ranges to the range box list.

        Parameters
        ----------
        num (int or ipywidgets.IntSlider): Currently displayed frame number.

        Returns
        -------
        '''

        tools = 'pan, box_zoom, wheel_zoom, reset'

        # plot current frame
        bg_fig = figure(title=f"Current Frame: {num}",
                        tools=tools,
                        output_backend="webgl")

        displayed_frame = self.cleaned_data[num]

        data = dict(image=[displayed_frame],
                    x=[0],
                    y=[0],
                    dw=[displayed_frame.shape[1]],
                    dh=[displayed_frame.shape[0]])

        bokeh_plot_helper(bg_fig, data)

        output = widgets.Output(layout=widgets.Layout(align_items='center'))
        with output:
            show(bg_fig)

        output_box = widgets.HBox([output, self.range_box])

        # Display centered grid plot
        display(output_box, self.button_box)

    def get_corrected_data(self):
        '''
        Apply the selected flip orientation ranges to the entire dataset to correct the
         incorrectly oriented frames.

        Returns
        -------
        '''

        # Get corrected frame ranges
        flips = self.frame_ranges
        correct_idx = np.concatenate([list(flip) for flip in flips])

        # Get list of frame indices where the mouse is facing east
        flip_idx = np.setdiff1d(np.arange(self.cleaned_data.shape[0]), correct_idx)

        self.corrected_data = deepcopy(self.cleaned_data)
        self.corrected_data[flip_idx] = np.flip(self.corrected_data[flip_idx], axis=2)

    def augment_dataset(self):
        '''
        Augments the dataset with 90 degree rotated versions of the truth values, and creates the
        X and Y train/test sets.

        Returns
        -------
        '''

        # Get flipped data
        data_xflip = np.flip(self.corrected_data, axis=2)
        data_yflip = np.flip(self.corrected_data, axis=1)
        data_xyflip = np.flip(data_yflip, axis=2)

        # Pack X and y training sets
        npixels = self.corrected_data.shape[1] * self.corrected_data.shape[2]
        ntrials = self.corrected_data.shape[0]

        self.x = np.vstack((data_xflip.reshape((-1, npixels)), data_xyflip.reshape((-1, npixels)),
                       data_yflip.reshape((-1, npixels)), self.corrected_data.reshape((-1, npixels))))

        # class 1 is x facing west
        self.y = np.concatenate((np.ones((ntrials * 2,)), np.zeros((ntrials * 2,))))

    def prepare_datasets(self, test_size):
        '''
        Correct data after the appropriate flip ranges have been selected, augment and create X,y training sets,
         and split the data to training and testing splits.

        Parameters
        ----------
        test_size (int): Test dataset percent split size

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
                                                                                random_state=0)

    def train_and_evaluate_model(self):
        '''

        Trains the flip classifier and prints its performance accuracy on the test set.

        Returns
        -------
        '''

        self.clf.fit(self.x_train, self.y_train)

        y_predict = self.clf.predict(self.x_test)

        percent_correct = np.mean(self.y_test == y_predict) * 1e2
        print('Performance: {0:3f}% correct'.format(percent_correct))

        if percent_correct > 90:
            print('You have achieved acceptable model accuracy.')
            print('Save the model trained on all the data, re-extract the data and continue to the PCA step.')
        else:
            print('Model performance is not high enough. Either try selecting more accurate frame indices, '
                  'or re-extract the data with your latest flip classifier (with highest accuracy).')

        joblib.dump(self.clf, self.output_file)
        print(f'Saved model in {self.output_file}')

    def train_and_save_model(self):
        '''
        Trains the model on the entire dataset and saves the model to the desired output file path.

        Returns
        -------
        '''

        self.clf.fit(self.x, self.y)

        joblib.dump(self.clf, self.output_file)
        print(f'Saved model in {self.output_file}')
