'''

Interactive ROI detection and extraction preview functionalities. This module utilizes the widgets from
the widgets.py file to facilitate the real-time interaction.

'''

import os
import cv2
import math
import warnings
import numpy as np
from math import isclose
from bokeh.io import show
from copy import deepcopy
import ruamel.yaml as yaml
from tqdm.auto import tqdm
from ipywidgets import fixed
import ipywidgets as widgets
from bokeh.models import Div
from os.path import dirname, basename, join
from moseq2_app.gui.progress import get_session_paths
from moseq2_extract.extract.extract import extract_chunk
from moseq2_app.roi.widgets import InteractiveROIWidgets
from IPython.display import display, clear_output, Markdown
from moseq2_app.roi.view import plot_roi_results, show_extraction
from moseq2_extract.extract.proc import apply_roi, threshold_chunk
from moseq2_extract.helpers.extract import process_extract_batches
from moseq2_extract.io.video import load_movie_data, get_video_info
from moseq2_extract.extract.proc import get_roi, get_bground_im_file
from moseq2_extract.util import (get_bucket_center, get_strels, select_strel,
                                 set_bground_to_plane_fit, set_bg_roi_weights,
                                 check_filter_sizes)


class InteractiveFindRoi(InteractiveROIWidgets):

    def __init__(self, data_path, config_file, session_config, compute_bgs=True):
        '''

        Parameters
        ----------
        data_path (str): Path to base directory containing all sessions to test
        config_file (str): Path to main configuration file.
        session_config (str): Path to session-configuration file.
        compute_bgs (bool): Indicates whether to compute all the session backgrounds prior to app launch.
        '''

        super().__init__()

        # Read default config parameters
        with open(config_file, 'r') as f:
            self.config_data = yaml.safe_load(f)

        self.session_config = session_config
        self.session_parameters = {}

        # Read individual session config if it exists
        if session_config is not None:
            if os.path.exists(session_config):
                with open(session_config, 'r') as f:
                    self.session_parameters = yaml.safe_load(f)
            else:
                warnings.warn('Session configuration file was not found. Generating a new one.')

                # Generate session config file if it does not exist
                session_config = join(dirname(config_file), 'session_config.yaml')
                self.session_parameters = {}
                with open(session_config, 'w+') as f:
                    yaml.safe_dump(self.session_parameters, f)

        self.all_results = {}

        self.config_data['session_config_path'] = session_config
        self.config_data['config_file'] = config_file

        # Update DropDown menu items
        self.sessions = get_session_paths(data_path)
        states = [0] * len(self.sessions.keys())

        # Session selection dict key names
        self.keys = list(self.sessions.keys())

        c_base = int("1F534", base=16)
        options = list(self.sessions.keys())
        colored_options = ['{} {}'.format(chr(c_base + s), o) for s, o in zip(states, options)]

        # Set Initial List options
        self.checked_list.options = colored_options
        self.checked_list.value = colored_options[0]

        # Display validation indicator
        self.indicator_layout = widgets.Layout(display='flex',
                                               flex_flow='column',
                                               font_size='150%',
                                               align_items='center',
                                               width='100%')
        self.indicator = widgets.HTML(value="")

        # Set save parameters button callback
        self.save_parameters.on_click(self.save_clicked)

        # Set check all sessions button callback
        self.check_all.on_click(self.check_all_sessions)

        # Set min-max range slider callback
        self.minmax_heights.observe(self.update_minmax_config, names='value')

        # Set extract button callback
        self.extract_button.on_click(self.extract_button_clicked)

        # Set passing button callback
        self.mark_passing.on_click(self.mark_passing_button_clicked)

        # Set extract frame range slider
        self.frame_range.observe(self.update_config_fr, names='value')

        # Set session select callback
        self.checked_list.observe(self.get_selected_session, names='value')

        self.clear_button.on_click(self.clear_on_click)

        # Update main configuration parameters
        self.config_data = set_bg_roi_weights(self.config_data)
        self.config_data = check_filter_sizes(self.config_data)
        self.config_data['autodetect'] = True

        if compute_bgs:
            self.compute_all_bgs()

    def clear_on_click(self, b):
        '''
        Clears the cell output

        Parameters
        ----------
        b (button click): User clicks Clear Button.

        Returns
        -------
        '''

        clear_output()

    def get_selected_session(self, event):
        '''
        Updates the selected value in checked_list, triggering the view to update with the currently
         selected session's data.

        Parameters
        ----------
        event (ipywidgets event): Current value of checked_list has changed

        Returns
        -------
        '''

        self.checked_list.value = event.new

    def compute_all_bgs(self):
        '''
        Computes all the background images before displaying the app to speed up user interaction.

        Returns
        -------
        '''

        for s, p in tqdm(self.sessions.items(), total=len(self.sessions.keys()), desc='Computing backgrounds'):
            try:
                # Compute background image; saving the image to a file
                get_bground_im_file(p)
            except:
                # Print error if an issue arises
                display(f'Error, could not compute background for session: {s}.')
                pass

    def extract_button_clicked(self, b):
        '''
        Updates the true depth autodetection parameter
         such that the true depth is autodetected for each found session

        Parameters
        ----------
        b (ipywidgets Button): Button click event.

        Returns
        -------
        '''

        clear_output()
        display(self.ui_tools, self.main_out)
        self.get_extraction(self.curr_session, self.curr_bground_im, self.curr_results['roi'])

    def mark_passing_button_clicked(self, b):
        '''
        Callback function that sets the current session as Passing. The indicator will still remain Flagged if the
        segmented frame or the crop-rotated frame are not appearing.

        Parameters
        ----------
        b (ipywidgets event): Button click

        Returns
        -------

        '''

        self.curr_results['flagged'] = False
        self.curr_results['ret_code'] = "0x1f7e2"

        # Update checked list
        self.config_data['pixel_area'] = self.curr_results['counted_pixels']
        self.session_parameters[self.keys[self.checked_list.index]] = deepcopy(self.config_data)
        self.indicator.value = '<center><h2><font color="green";>Passing</h2></center>'

        self.update_checked_list(self.curr_results)

    def check_all_sessions(self, b):
        '''
        Callback function to run the ROI area comparison test on all the existing sessions.
        Saving their individual session parameter sets in the session_parameters dict in the process.
        Additionally updates the button styles to reflect the outcome.

        Parameters
        ----------
        b (button event): User click

        Returns
        -------
        '''

        self.check_all.description = 'Checking...'
        self.save_parameters.button_style = 'info'
        self.save_parameters.icon = ''

        self.test_all_sessions(self.sessions)

        # Handle button styles
        if all(list(self.all_results.values())) == False:
            self.check_all.button_style = 'success'
            self.check_all.icon = 'check'
        else:
            self.check_all.button_style = 'danger'
        self.check_all.description = 'Check All Sessions'

        self.save_parameters.layout = self.layout_visible

    def save_clicked(self, b):
        '''
        Callback function to save the current session_parameters dict into
        the file of their choice (given in the top-most wrapper function).

        Parameters
        ----------
        b (button event): User click

        Returns
        -------
        '''

        # Update current session with current configuration parameters
        self.session_parameters[self.keys[self.checked_list.index]] = deepcopy(self.config_data)

        # Update session parameters
        with open(self.config_data['session_config_path'], 'w+') as f:
            yaml.safe_dump(self.session_parameters, f)

        with open(self.config_data['config_file'], 'w+') as f:
            yaml.safe_dump(self.config_data, f)

        self.save_parameters.button_style = 'success'
        self.save_parameters.icon = 'check'

    def update_minmax_config(self, event):
        '''
        Callback function to update config dict with current UI min/max height range values

        Parameters
        ----------
        event (ipywidget callback): Any user interaction.

        Returns
        -------
        '''

        self.config_data['min_height'] = self.minmax_heights.value[0]
        self.config_data['max_height'] = self.minmax_heights.value[1]

    def update_config_fr(self, event):
        '''
        Callback function to update config dict with current UI depth range values

        Parameters
        ----------
        event (ipywidget callback): Any user interaction.

        Returns
        -------
        '''

        self.config_data['frame_range'] = self.frame_range.value

    def test_all_sessions(self, session_dict):
        '''
        Helper function to test the current configurable UI values on all the
        sessions that were found.

        Parameters
        ----------
        session_dict (dict): dict of session directory names paired with their absolute paths.
        config_data (dict): ROI/Extraction configuration parameters.

        Returns
        -------
        all_results (dict): dict of session names and values used to indicate if a session was flagged,
        with their computed ROI for convenience.
        '''
        checked_options = list(self.checked_list.options)

        self.npassing = 0

        # test saved config data parameters on all sessions
        for i, (sessionName, sessionPath) in enumerate(session_dict.items()):
            if sessionName != self.curr_session:
                # Get background image for each session and test the current parameters on it
                bground_im = get_bground_im_file(sessionPath)
                try:
                    sess_res = self.get_roi_and_depths(bground_im, sessionPath)
                except:
                    sess_res = {'flagged': True, 'ret_code': '0x1f534'}

                # Save session parameters if it is not flagged
                if not sess_res['flagged']:
                    self.npassing += 1
                    self.session_parameters[sessionName] = deepcopy(self.config_data)

                # Update label
                self.checked_lbl.value = f'Passing Sessions: {self.npassing}/{len(self.checked_list.options)}'

                # Set index passing value
                checked_options[i] = f'{chr(int(sess_res["ret_code"], base=16))} {sessionName}'

                # Updating progress
                self.all_results[sessionName] = sess_res['flagged']

        # Updating displayed list
        self.checked_list.options = checked_options
        self.checked_list.value = checked_options[self.checked_list.index]

        if self.npassing == len(self.checked_list.options):
            self.message.value = 'All sessions passed with the current parameter set. \
            Save the parameters and move to the "Extract All" cell.'
        else:
            self.message.value = 'Some sessions were flagged. Save the parameter set for the current passing sessions, \
             then find and save the correct set for the remaining sessions.'

    def interactive_find_roi_session_selector(self, session):
        '''
        First function that is called to find the current selected session's background
        and display the widget interface.

        Parameters
        ----------
        session (str or ipywidget DropDownMenu): path to chosen session.
        config_data (dict): ROI/Extraction configuration parameters

        Returns
        -------
        '''

        self.check_all.button_style = ''
        self.check_all.icon = ''
        self.message.value = ''

        # Get current session name to look up path
        self.formatted_key = session.split(' ')[1]

        if self.formatted_key in self.keys:
            session = self.sessions[self.formatted_key]

        # Get background and display UI plots
        bground_im = get_bground_im_file(session)
        clear_output()

        self.main_out = widgets.interactive_output(self.interactive_depth_finder, {'session': fixed(session),
                                                                                   'bground_im': fixed(bground_im),
                                                                                   'dr': self.bg_roi_depth_range,
                                                                                   'di': self.dilate_iters}
                                                   )

        display(self.main_out)

    def update_checked_list(self, results):
        '''
        Helper function to update the session selector passing indicators when a parameter test is run.

        Parameters
        ----------
        results (dict): ROI detection results dict containing the flag and return code to display.

        Returns
        -------
        '''

        curr_index = self.checked_list.index
        cur_val = list(self.checked_list.options)[curr_index].split(' ')
        checked_options = list(self.checked_list.options)

        # Update Checked List
        checked_options[curr_index] = f'{chr(int(results["ret_code"], base=16))} {cur_val[1]}'
        self.checked_list._initializing_traits_ = True
        self.checked_list.options = checked_options

        self.checked_list._initializing_traits_ = False
        self.checked_list.value = checked_options[curr_index]

    def interactive_depth_finder(self, session, bground_im, dr, di):
        '''
        Interactive helper function that updates that views whenever the depth range or
        dilation iterations sliders are changed.
        At initial launch, it will auto-detect the depth estimation, then it will preserve the parameters
        across session changes.

        Parameters
        ----------
        session (str or ipywidget DropDownMenu): path to input file
        bground_im (2D np.array): Computed session background
        config_data (dict): Extraction configuration parameters
        dr (tuple or ipywidget IntRangeSlider): Depth range to capture
        di (int or ipywidget IntSlider): Dilation iterations

        Returns
        -------
        '''

        if '.tar' in session:
            self.config_data['tar'] = True
        else:
            self.config_data['tar'] = False

        self.save_parameters.button_style = 'primary'
        self.save_parameters.icon = 'none'

        # Setting current context parameters
        self.curr_session = session
        self.curr_bground_im = bground_im

        # Autodetect reference depth range and min-max height values at launch
        if self.config_data['autodetect']:
            self.curr_results = self.get_roi_and_depths(bground_im, session)
            if not self.curr_results['flagged']:
                self.config_data['autodetect'] = False

            # Update the session flag result
            self.all_results[self.keys[self.checked_list.index]] = self.curr_results['flagged']

            # Set initial frame range tuple value
            self.config_data['frame_range'] = self.frame_range.value

            # Update sliders with autodetected values
            self.bg_roi_depth_range.value = self.config_data['bg_roi_depth_range']
            self.minmax_heights.value = [self.config_data['min_height'], self.config_data['max_height']]
        else:
            # Test updated parameters
            self.config_data['bg_roi_depth_range'] = (int(dr[0]), int(dr[1]))
            self.config_data['dilate_iterations'] = di

            # Update the session flag result
            self.get_roi_and_depths(bground_im, session)
            self.all_results[self.keys[self.checked_list.index]] = self.curr_results['flagged']

        # set indicator
        if self.curr_results['flagged']:
            self.indicator.value = '<center><h2><font color="red";>Flagged: Current ROI pixel area may be incorrect. If ROI is acceptable,' \
                                   ' Mark it as passing. Otherwise, change the depth range values.</h2></center>'
        else:
            self.indicator.value = '<center><h2><font color="green";>Passing</h2></center>'
            # Save passing session parameters
            self.session_parameters[self.keys[self.checked_list.index]] = deepcopy(self.config_data)

        # Clear output to update view
        clear_output()

        # Display extraction validation indicator
        display(self.indicator)

        out = widgets.interactive_output(self.prepare_data_to_plot, {'input_file': fixed(session),
                                                                     'bground_im': fixed(bground_im),
                                                                     'roi': fixed(self.curr_results['roi']),
                                                                     'minmax_heights': self.minmax_heights,
                                                                     'fn': self.frame_num})
        # display graphs
        display(out)

    def get_pixels_per_metric(self, pixel_width):
        '''
        Helper function that computes a pixels_per_metric value, and handles
         cases without user input.

        Parameters
        ----------
        pixel_width (int): width of the ROI bounding box in pixels

        Returns
        -------
        pixels_per_inch (float): Computed ratio of real life
        '''

        if self.config_data.get('arena_width') is not None:
            pixels_per_inch = pixel_width / self.config_data['arena_width']
        elif self.config_data.get('true_height') is not None:
            pixels_per_inch = pixel_width / self.config_data['true_height']
        else:
            warnings.warn('Warning: arena_width and true_height were not provided. '
                          'Using arbitrary comparison distance value.')
            self.config_data['true_height'] = float(self.true_depth)
            pixels_per_inch = pixel_width / self.config_data['true_height']

        return pixels_per_inch

    def get_roi_and_depths(self, bground_im, session):
        '''
        Performs bucket centroid estimation to find the coordinates to use as the true depth value.
        The true depth will be used to estimate the background depth_range, then it will update the
        widget values in real time.

        Parameters
        ----------
        bground_im (2D np.array): Computed session background
        session (str): path to currently processed session
        config_data (dict): Extraction configuration parameters

        Returns
        -------
        results (dict): dict that contains computed information. E.g. its ROI, and if it was flagged.
        '''

        # initialize results dict
        self.curr_results = {'flagged': False,
                             'ret_code': "0x1f7e2"}

        if self.config_data['autodetect']:
            # Get max depth as a thresholding limit (this would be the DTD if it already was computed)
            limit = np.max(bground_im)

            # Compute bucket distance thresholding value
            threshold_value = np.mean(bground_im) - np.std(bground_im)

            # Threshold image to find depth at bucket center: the true depth
            cX, cY = get_bucket_center(bground_im, limit, threshold=threshold_value)

            # True depth is at the center of the bucket
            self.true_depth = bground_im[cY][cX]

            # Get true depth range difference
            range_diff = 10 ** (len(str(int(self.true_depth))) - 1)

            # Center the depth ranges around the true depth
            bg_roi_range_min = self.true_depth - range_diff
            bg_roi_range_max = self.true_depth + range_diff

            self.config_data['bg_roi_depth_range'] = (bg_roi_range_min, bg_roi_range_max)

            if bg_roi_range_max > self.bg_roi_depth_range.max:
                self.bg_roi_depth_range.max = bg_roi_range_max + range_diff

        # Get relevant structuring elements
        strel_dilate = select_strel(self.config_data['bg_roi_shape'], tuple(self.config_data['bg_roi_dilate']))
        strel_erode = select_strel(self.config_data['bg_roi_shape'], tuple(self.config_data['bg_roi_erode']))

        try:
            # Get ROI
            rois, plane, bboxes, _, _, _ = get_roi(bground_im,
                                                   **self.config_data,
                                                   strel_dilate=strel_dilate,
                                                   strel_erode=strel_erode,
                                                   get_all_data=True
                                                   )
        except:
            self.curr_results['flagged'] = True
            self.curr_results['ret_code'] = "0x1f534"
            self.curr_results['roi'] = np.zeros_like(self.curr_bground_im)
            self.update_checked_list(results=self.curr_results)
            return self.curr_results

        if self.config_data['use_plane_bground']:
            print('Using plane fit for background...')
            self.curr_bground_im = set_bground_to_plane_fit(bground_im, plane, join(dirname(session), 'proc'))

        if self.config_data['autodetect']:
            # Get pixel dims from bounding box
            xmin = bboxes[0][0][1]
            xmax = bboxes[0][1][1]

            ymin = bboxes[0][0][0]
            ymax = bboxes[0][1][0]

            # bucket width in pixels
            pixel_width = xmax - xmin
            pixel_height = ymax - ymin

            pixels_per_inch = self.get_pixels_per_metric(pixel_width)

            self.config_data['pixels_per_inch'] = float(pixels_per_inch)
            # Corresponds to a rough pixel area estimate
            r = float(cv2.countNonZero(rois[0].astype('uint8')))
            self.config_data['pixel_area'] = r
        else:
            pixels_per_inch = self.config_data['pixels_per_inch']
            # Corresponds to a rough pixel area estimate
            r = float(cv2.countNonZero(rois[0].astype('uint8')))

        # Compute pixel area per metric
        if self.config_data.get('arena_width') is not None:
            # Compute arena area
            if self.config_data['arena_shape'] == 'ellipse':
                area = math.pi * (self.config_data['arena_width'] / 2) ** 2
            elif 'rect' in self.config_data['arena_shape']:
                estimated_height = pixel_height / pixels_per_inch
                area = self.config_data['arena_width'] * estimated_height

            self.config_data['area_px_per_inch'] = r / area / pixels_per_inch

        try:
            assert isclose(self.config_data['pixel_area'], r, abs_tol=50e2)
        except AssertionError:
            if self.config_data.get('pixel_area', 0) > r:
                self.curr_results['flagged'] = True
                self.curr_results['ret_code'] = "0x1f534"

        # Save ROI
        self.curr_results['roi'] = rois[0]
        self.curr_results['counted_pixels'] = r
        self.update_checked_list(results=self.curr_results)

        return self.curr_results

    def get_extraction(self, input_file, bground_im, roi):
        '''
        Extracts selected frame range (with the currently set session parameters)
        and displays the extraction as a Bokeh HTML-embedded div.

        Parameters
        ----------
        input_file (str): Path to session to extract
        config_data (dict): Extraction configuration parameters.
        bground_im (2D np.array): Computed session background.
        roi (2D np.array): Computed Region of interest array to mask bground_im with.

        Returns
        -------
        '''

        # Get structuring elements
        str_els = get_strels(self.config_data)

        # Get output path for extraction video
        output_dir = dirname(input_file)
        outpath = 'extraction_preview'
        view_path = join(output_dir, outpath + '.mp4')

        # Get frames to extract
        frame_batches = [range(self.config_data['frame_range'][0], self.config_data['frame_range'][1])]

        # Remove previous preview
        if os.path.exists(view_path):
            os.remove(view_path)

        # load chunk to display
        process_extract_batches(input_file, self.config_data,
                                bground_im, roi, frame_batches,
                                0, str_els, view_path)

        # display extracted video as HTML Div using Bokeh
        show_extraction(basename(dirname(input_file)), view_path)

    def prepare_data_to_plot(self, input_file, bground_im, roi, minmax_heights, fn):
        '''
        Helper function that generates the display plots with the currently selected parameters,
         and checks if the min-max height parameters are acceptable, updating the success indicator if any
         issues arise.

        Parameters
        ----------
        input_file (str): Path to currently processed depth file
        bground_im (2D np.ndarray): Median depth image of the read video.
        roi (2D Boolean np.ndarray): Computed ROI representing the bucket floor
        minmax_heights (2-tuple): Min and max mouse heights to threshold from background subtracted image.
        fn (int): Frame index to display.

        Returns
        -------
        '''

        # update adjusted min and max heights
        self.config_data['min_height'] = minmax_heights[0]
        self.config_data['max_height'] = minmax_heights[1]

        # update current session parameters
        self.session_parameters[self.keys[self.checked_list.index]] = deepcopy(self.config_data)

        # get segmented frame
        raw_frames = load_movie_data(input_file, range(fn, fn + 30), frame_dims=bground_im.shape[::-1])
        curr_frame = (bground_im - raw_frames)

        # filter out regions outside of ROI
        try:
            filtered_frames = apply_roi(curr_frame, roi)[0].astype(self.config_data['frame_dtype'])
        except:
            # Display ROI error and flag
            filtered_frames = curr_frame.copy()[0]
            if not self.curr_results['flagged']:
                self.indicator.value = '<center><h2><font color="red";>Flagged: Could not apply ROI to loaded frames.</h2></center>'
                self.curr_results['flagged'] = True

        # filter for included mouse height range
        try:
            filtered_frames = threshold_chunk(filtered_frames, minmax_heights[0], minmax_heights[1])
        except:
            # Display min-max heights error and flag
            filtered_frames = curr_frame.copy()[0]
            if not self.curr_results['flagged']:
                self.indicator.value = '<center><h2><font color="red";>Flagged: Mouse Height threshold range is incorrect.</h2></center>'
                self.curr_results['flagged'] = True

        # Get overlayed ROI
        overlay = bground_im.copy()
        overlay[roi != True] = 0

        # prepare extraction metadatas
        str_els = get_strels(self.config_data)
        self.config_data['tracking_init_mean'] = None
        self.config_data['tracking_init_cov'] = None

        # extract crop-rotated selected frame
        try:
            result = extract_chunk(**self.config_data,
                                   **str_els,
                                   chunk=raw_frames.copy(),
                                   roi=roi,
                                   bground=bground_im,
                                   )
        except:
            # Display error and flag
            result = {'depth_frames': np.zeros((1, self.config_data['crop_size'][0], self.config_data['crop_size'][1]))}

        if (result['depth_frames'] == np.zeros((1, self.config_data['crop_size'][0], self.config_data['crop_size'][1]))).all():
            if not self.curr_results['flagged']:
                self.indicator.value = '<center><h2><font color="red";>Flagged: Mouse Height threshold range is incorrect.</h2></center>'
                self.curr_results['flagged'] = True
        else:
            self.indicator.value = "<center><h2><font color='green';>Passing</h2></center>"
            self.curr_results['flagged'] = False

        # Make and display plots
        plot_roi_results(self.formatted_key, bground_im, roi, overlay, filtered_frames, result['depth_frames'][0], fn)

class InteractiveExtractionViewer:

    def __init__(self, data_path):
        '''

        Parameters
        ----------
        data_path (str): Path to base directory containing all sessions to test
        '''

        self.sess_select = widgets.Dropdown(options=get_session_paths(data_path, extracted=True),
                                            description='Session:', disabled=False, continuous_update=True)

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.clear_button.on_click(self.clear_on_click)

    def clear_on_click(self, b):
        '''
        Clears the cell output

        Parameters
        ----------
        b (button click)

        Returns
        -------
        '''

        clear_output()

    def get_extraction(self, input_file):
        '''
        Returns a div containing a video object to display.

        Parameters
        ----------
        input_file (str): Path to session extraction video to view.

        Returns
        -------
        '''

        video_dims = get_video_info(input_file)['dims']

        # display extracted video as HTML Div using Bokeh
        video_div = f'''
                        <h2>{input_file}</h2>
                        <video
                            src="{input_file}"; alt="{input_file}"; 
                            height="{video_dims[1]}"; width="{video_dims[0]}"; preload="auto";
                            style="float: center; type: "video/mp4"; margin: 0px 10px 10px 0px;
                            border="2"; autoplay controls loop>
                        </video>
                     '''

        div = Div(text=video_div, style={'width': '100%', 'align-items': 'center', 'display': 'contents'})
        show(div)