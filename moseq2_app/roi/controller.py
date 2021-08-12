'''

Interactive ROI detection and extraction preview functionalities. This module utilizes the widgets from
the widgets.py file to facilitate the real-time interaction.

'''

import gc
import os
import cv2
import bokeh
import warnings
import numpy as np
from math import isclose
from bokeh.io import show
from copy import deepcopy
import ruamel.yaml as yaml
from tqdm.auto import tqdm
import ipywidgets as widgets
from bokeh.models import Div, CustomJS, Slider
from moseq2_extract.io.image import write_image
from IPython.display import display, clear_output
from moseq2_app.gui.progress import get_session_paths
from moseq2_extract.extract.extract import extract_chunk
from moseq2_app.roi.widgets import InteractiveROIWidgets
from os.path import dirname, basename, join, relpath, abspath
from moseq2_app.roi.view import plot_roi_results, show_extraction
from moseq2_extract.extract.proc import apply_roi, threshold_chunk
from moseq2_extract.helpers.extract import process_extract_batches
from moseq2_extract.extract.proc import get_roi, get_bground_im_file
from moseq2_extract.io.video import (load_movie_data, get_video_info,
                                     get_movie_info, load_timestamps_from_movie)
from moseq2_extract.util import (get_bucket_center, get_strels, select_strel, read_yaml,
                                 set_bground_to_plane_fit, detect_and_set_camera_parameters,
                                 check_filter_sizes)


class InteractiveFindRoi(InteractiveROIWidgets):

    def __init__(self, data_path, config_file, session_config, compute_bgs=True, autodetect_depths=False, overwrite=False):
        '''

        Parameters
        ----------
        data_path (str): Path to base directory containing all sessions to test
        config_file (str): Path to main configuration file.
        session_config (str): Path to session-configuration file.
        compute_bgs (bool): Indicates whether to compute all the session backgrounds prior to app launch.
        overwrite (bool): if True, will overwrite the previously saved session_config.yaml file
        '''

        super().__init__()

        # main output gui to be reused
        self.main_out = None
        self.output = None

        # setting input variables
        self.autodetect_depths = autodetect_depths

        # initial number of sessions that are considering Passing
        self.npassing = 0

        # initialize reusable results dicts
        self.curr_results = {'flagged': False,
                             'ret_code': "0x1f7e2",
                             'err_code': -1}
        self.all_results = {}

        # initialize reusable background image variable
        self.curr_bground_im = None

        # Read default config parameters
        self.config_data = read_yaml(config_file)

        # Update DropDown menu items
        self.sessions = get_session_paths(data_path)

        assert len(self.sessions) > 0, "No sessions were found in the provided base_dir"

        # Session selection dict key names
        self.keys = list(self.sessions.keys())

        # Ensure config file is reading frames with >1 thread for fast reloading
        if self.config_data.get('threads', 8) < 1:
            self.config_data['threads'] = 8

        # Ensure config file contains all required parameters prior to creating session_config
        self.config_data = check_filter_sizes(self.config_data)
        self.config_data['pixel_areas'] = []
        self.config_data['autodetect'] = True
        self.config_data['detect'] = True
        if 'bg_roi_erode' not in self.config_data:
            self.config_data['bg_roi_erode'] = (1, 1)
        if 'bg_roi_dilate' not in self.config_data:
            self.config_data['bg_roi_dilate'] = (1, 1)





    def test_all_sessions(self, session_dict):
        '''
        Helper function to test the current configurable UI values on all the
        sessions that were found.

        Parameters
        ----------
        session_dict (dict): dict of session directory names paired with their absolute paths.
        config_data (dict): ROI/Extraction configuration parameters.
        # Create initial shared session_parameters dict
        self.session_parameters = {k: deepcopy(self.config_data) for k in self.keys}

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
                # finfo is a key that points to a dict that contains the following keys:
                # ['file', 'dims', 'fps', 'nframes']. These are determined from moseq2-extract.io.video.get_video_info()
                if 'finfo' not in self.session_parameters[sessionName]:
                    self.session_parameters[sessionName]['finfo'] = get_movie_info(sessionPath)

                # Get background image for each session and test the current parameters on it
                self.session_parameters[sessionName].pop('output_dir', None)
                bground_im = get_bground_im_file(sessionPath, **self.session_parameters[sessionName])
                try:
                    sess_res = self.get_roi_and_depths(bground_im, sessionPath)
                except:
                    sess_res = {'flagged': True, 'ret_code': '0x1f534'}
        # Update global session config: self.session_paramters
        self.get_session_config(session_config=session_config, overwrite=overwrite)
        self.config_data['session_config_path'] = session_config
        self.config_data['config_file'] = config_file

                # Save session parameters if it is not flagged
                if not sess_res['flagged']:
                    self.npassing += 1
        for k in self.keys:
            self.session_parameters[k] = detect_and_set_camera_parameters(self.session_parameters[k], self.sessions[k])

        # Create colored dots for each session item in the checked_list widget
        states = [0] * len(self.sessions.keys())
        c_base = int("1F534", base=16)
        options = list(self.sessions.keys())
        colored_options = ['{} {}'.format(chr(c_base + s), o) for s, o in zip(states, options)]


                self.checked_list._initializing_traits_ = False

                # Updating progress
                self.all_results[sessionName] = sess_res['flagged']
                gc.collect()

        if compute_bgs:
            self.compute_all_bgs()

        if self.npassing == len(self.checked_list.options):
            self.save_clicked()
            self.message.value = 'All sessions passed and the config files have been saved.\n' \
                                 'You can now safely clear the output, and move to the "Extract All" cell.\n'
        else:
            tmp_message = 'Some sessions were flagged. Save the parameter set for the current passing sessions, \
             then find and save the correct set for the remaining sessions.\n'
            if self.autodetect_depths == False:
                tmp_message += ' Try Clearing the output, and rerunning the cell with autodetect_depths = True'
            self.message.value = tmp_message
        # start tool execution
        self.interactive_find_roi_session_selector(self.checked_list.value)

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
        curr_session_key = self.keys[self.checked_list.index]

        if self.formatted_key in self.keys:
            self.curr_session = self.sessions[self.formatted_key]

        # finfo is a key that points to a dict that contains the following keys:
        # ['file', 'dims', 'fps', 'nframes']. These are determined from moseq2-extract.io.video.get_video_info()
        if 'finfo' not in self.session_parameters[curr_session_key]:
            self.session_parameters[curr_session_key]['finfo'] = get_movie_info(self.curr_session)

        if self.curr_session.endswith('.mkv'):
            self.handle_mkv_files(self, curr_session_key, self.curr_session)

        # Update sliders with corresponding session's previously set values
        if not isinstance(self.session_parameters[curr_session_key]['bg_roi_depth_range'], str):
            self.safe_widget_value_update(wid_obj=self.bg_roi_depth_range,
                                          func=self.update_config_dr,
                                          new_val=self.session_parameters[curr_session_key]['bg_roi_depth_range'])

        self.safe_widget_value_update(wid_obj=self.minmax_heights,
                                      func=self.update_minmax_config,
                                      new_val=[self.session_parameters[curr_session_key]['min_height'],
                                               self.session_parameters[curr_session_key]['max_height']])

        self.safe_widget_value_update(wid_obj=self.dilate_iters,
                                      func=self.update_config_di,
                                      new_val=self.session_parameters[curr_session_key]['dilate_iterations'])

        # Get background and display UI plots
        self.session_parameters[curr_session_key].pop('output_dir', None)
        self.curr_bground_im = get_bground_im_file(self.curr_session, **self.session_parameters[curr_session_key])

        self.interactive_depth_finder()

    def interactive_depth_finder(self):
        '''
        Interactive helper function that updates that views whenever the depth range or
        dilation iterations sliders are changed.
        At initial launch, it will auto-detect the depth estimation, then it will preserve the parameters
        across session changes.

        Parameters
        ----------
        Returns
        -------
        '''

        # Get widget values
        minmax_heights = self.minmax_heights.value
        fn = self.frame_num.value
        dr = self.bg_roi_depth_range.value
        di = self.dilate_iters.value

        curr_session_key = self.keys[self.checked_list.index]

        self.save_parameters.button_style = 'primary'
        self.save_parameters.icon = 'none'

        # Autodetect reference depth range and min-max height values at launch
        if self.config_data['autodetect']:
            self.curr_results = self.get_roi_and_depths(self.curr_bground_im, self.curr_session)
            if not self.curr_results['flagged']:
                self.config_data['autodetect'] = False

            # Update the session flag result
            self.all_results[curr_session_key] = self.curr_results['flagged']

            # Set initial frame range tuple value
            self.session_parameters[curr_session_key]['frame_range'] = self.frame_range.value

            # Update sliders with corresponding session's autodetected values
            self.safe_widget_value_update(wid_obj=self.minmax_heights,
                                          func=self.update_minmax_config,
                                          new_val=[self.session_parameters[curr_session_key]['min_height'],
                                                   self.session_parameters[curr_session_key]['max_height']])

            self.safe_widget_value_update(wid_obj=self.bg_roi_depth_range,
                                          func=self.update_config_dr,
                                          new_val=self.session_parameters[curr_session_key]['bg_roi_depth_range'])
        else:
            # Test updated parameters
            self.session_parameters[curr_session_key]['bg_roi_depth_range'] = (int(dr[0]), int(dr[1]))
            self.session_parameters[curr_session_key]['dilate_iterations'] = di

            if self.config_data['detect']:
                # Update the session flag result
                self.curr_results = self.get_roi_and_depths(self.curr_bground_im, self.curr_session)
                self.all_results[curr_session_key] = self.curr_results['flagged']

        # Clear output to update view
        clear_output()

        # Display extraction validation indicator
        display(self.indicator)

        # display graphs
        self.prepare_data_to_plot(self.curr_results['roi'], minmax_heights, fn)

        gc.collect()

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



        # Get relevant structuring elements
        strel_dilate = select_strel(self.config_data['bg_roi_shape'], tuple(self.config_data['bg_roi_dilate']))
        strel_erode = select_strel(self.config_data['bg_roi_shape'], tuple(self.config_data['bg_roi_erode']))

        # get the current background image
        self.session_parameters[curr_session_key].pop('output_dir', None)
        self.curr_bground_im = get_bground_im_file(self.curr_session, **self.session_parameters[curr_session_key])

        try:
            # Get ROI
            rois, plane, bboxes, _, _, _ = get_roi(bground_im,
                                                   **self.session_parameters[curr_session_key],
                                                   strel_dilate=strel_dilate,
                                                   strel_erode=strel_erode,
                                                   get_all_data=True
                                                   )
        except ValueError:
            # bg depth range did not capture any area
            # flagged + ret_code are used to display a red circle in the session selector to indicate a failed
            # roi detection.
            return curr_results
        except Exception as e:
            # catching any remaining possible exceptions to preserve the integrity of the interactive GUI.
            print(e)
            return curr_results

        if self.config_data['use_plane_bground']:
            print('Using plane fit for background...')
            self.curr_bground_im = set_bground_to_plane_fit(bground_im, plane, join(dirname(session), 'proc'))

        if self.config_data['autodetect']:
            # Corresponds to a rough pixel area estimate
            r = float(cv2.countNonZero(rois[0].astype('uint8')))
            self.config_data['pixel_areas'].append(r)
            self.session_parameters[curr_session_key]['pixel_area'] = r
        else:
            # Corresponds to a rough pixel area estimate
            r = float(cv2.countNonZero(rois[0].astype('uint8')))
            self.session_parameters[curr_session_key]['pixel_area'] = r

        # initialize flag to check whether this session's ROI has a comparable number of pixels
        # to the previously viewed sessions.
        res = False

        # check if the current measured area is within is the current list of ROI areas
        for area in self.config_data.get('pixel_areas', []):
            # if the current session's ROI is smaller than all the previously checked sessions
            # by at least 500 square pixels. Then this session's ROI will be flagged
            if isclose(area, r, abs_tol=50e2):
                res = True
                break

        if not res and (len(self.config_data.get('pixel_areas', [])) > 0):
            curr_results['flagged'] = True
            curr_results['ret_code'] = "0x1f534"
        else:
            # add accepted area size to
            self.config_data['pixel_areas'].append(r)


        return curr_results

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

        curr_session_key = self.keys[self.checked_list.index]

        # Get structuring elements
        str_els = get_strels(self.config_data)

        # Get output path for extraction video
        output_dir = dirname(input_file)
        outpath = 'extraction_preview'
        view_path = join(output_dir, outpath + '.mp4')

        # Get frames to extract
        frame_batches = [range(self.frame_range.value[0], self.frame_range.value[1])]

        # Remove previous preview
        if os.path.exists(view_path):
            os.remove(view_path)

        # load chunk to display
        process_extract_batches(input_file, self.session_parameters[curr_session_key],
                                bground_im, roi, frame_batches,
                                str_els, view_path)

        # display extracted video as HTML Div using Bokeh
        if self.output is not None:
            self.output = None
        self.output = show_extraction(basename(dirname(input_file)), view_path)
        gc.collect()

    def prepare_data_to_plot(self, roi, minmax_heights, fn):
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

        curr_session_key = self.keys[self.checked_list.index]

        # update adjusted min and max heights
        self.session_parameters[curr_session_key]['min_height'] = int(minmax_heights[0])
        self.session_parameters[curr_session_key]['max_height'] = int(minmax_heights[1])

        # prepare extraction metadatas
        str_els = get_strels(self.config_data)
        self.session_parameters[curr_session_key]['tracking_init_mean'] = None
        self.session_parameters[curr_session_key]['tracking_init_cov'] = None
        self.session_parameters[curr_session_key]['true_depth'] = int(self.true_depth)

        # get segmented frame
        raw_frames = load_movie_data(self.curr_session, 
                                    range(fn, fn + 30),
                                    **self.session_parameters[curr_session_key],
                                    frame_size=self.curr_bground_im.shape[::-1])

        # subtract background
        curr_frame = (self.curr_bground_im - raw_frames)

        # filter out regions outside of ROI
        try:
            filtered_frames = apply_roi(curr_frame, roi)[0].astype(self.config_data['frame_dtype'])
            
        except:
            # Display ROI error and flag
            filtered_frames = curr_frame.copy()[0]
            if not self.curr_results['flagged']:
                self.curr_results['flagged'] = True

        # filter for included mouse height range
        try:
            filtered_frames = threshold_chunk(filtered_frames, minmax_heights[0], minmax_heights[1])
        except:
            # Display min-max heights error and flag
            filtered_frames = curr_frame.copy()[0]
            if not self.curr_results['flagged']:
                self.curr_results['flagged'] = True

        # Get overlayed ROI
        overlay = self.curr_bground_im.copy()
        overlay[roi != True] = 0

        # extract crop-rotated selected frame
        try:
            result = extract_chunk(**self.session_parameters[curr_session_key],
                                   **str_els,
                                   chunk=raw_frames.copy(),
                                   roi=roi,
                                   bground=self.curr_bground_im,
                                   )
        except:
            # Display error and flag
            result = {'depth_frames': np.zeros((1, self.config_data['crop_size'][0], self.config_data['crop_size'][1]))}
            self.curr_results['flagged'] = False

        if self.config_data.get('camera_type', 'kinect') == 'azure':
            # orienting preview images to match sample extraction
            display_bg = np.flip(self.curr_bground_im.copy(), 0)
            overlay = np.flip(overlay, 0) # overlayed roi
            filtered_frames = np.flip(filtered_frames, 0) # segmented
        else:
            display_bg = self.curr_bground_im

        # Make and display plots
        plot_roi_results(self.formatted_key, display_bg, roi, overlay, filtered_frames, result['depth_frames'][0], fn)
        gc.collect()

class InteractiveExtractionViewer:

    def __init__(self, data_path, flipped=False):
        '''

        Parameters
        ----------
        data_path (str): Path to base directory containing all sessions to test
        flipped (bool): indicates whether to show corrected flip videos
        '''

        self.sess_select = widgets.Dropdown(options=get_session_paths(data_path, extracted=True, flipped=flipped),
                                            description='Session:', disabled=False, continuous_update=True)

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.clear_button.on_click(self.clear_on_click)

    def clear_on_click(self, b=None):
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

        video_div = f'''
                        <h2>{input_file}</h2>
                        <video
                            src="{relpath(input_file)}"; alt="{abspath(input_file)}"; id="preview";
                            height="{video_dims[1]}"; width="{video_dims[0]}"; preload="auto";
                            style="float: center; type: "video/mp4"; margin: 0px 10px 10px 0px;
                            border="2"; autoplay controls loop>
                        </video>
                        <script>
                            document.querySelector('video').playbackRate = 0.1;
                        </script>
                     '''

        div = Div(text=video_div, style={'width': '100%', 'align-items': 'center', 'display': 'contents'})

        slider = Slider(start=0, end=4, value=1, step=0.1,
                             format="0[.]00", title=f"Playback Speed")

        callback = CustomJS(
            args=dict(slider=slider),
            code="""
                    document.querySelector('video').playbackRate = slider.value;
                 """
        )

        slider.js_on_change('value', callback)
        show(slider)
        show(div)