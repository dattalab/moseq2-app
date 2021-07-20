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
                                 check_filter_sizes, graduate_dilated_wall_area)
try:
    from kora.drive import upload_public
except (ImportError, ModuleNotFoundError) as error:
    print(error)

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

        self.autodetect_depths = autodetect_depths

        # Read default config parameters
        self.config_data = read_yaml(config_file)

        self.session_config = session_config

        # Update DropDown menu items
        self.sessions = get_session_paths(data_path)

        assert len(self.sessions) > 0, "No sessions were found in the provided base_dir"

        # Session selection dict key names
        self.keys = list(self.sessions.keys())

        self.session_parameters = {k: deepcopy(self.config_data) for k in self.keys}

        # Read individual session config if it exists
        if session_config is None:
            self.generate_session_config(session_config)
        elif not os.path.exists(session_config):
            self.generate_session_config(session_config)
        else:
            if os.path.exists(session_config) and not overwrite:
                if os.stat(session_config).st_size <= 0:
                    self.generate_session_config(session_config)

        self.session_parameters = read_yaml(session_config)

        # Extra check: Handle broken session config files and generate default session params from config_data
        if self.session_parameters is None:
            self.session_parameters = {k: deepcopy(self.config_data) for k in self.keys}
        elif self.session_parameters == {}:
            self.session_parameters = {k: deepcopy(self.config_data) for k in self.keys}

        # add missing keys for newly found sessions
        if len(list(self.session_parameters.keys())) < len(self.keys):
            for key in self.keys:
                if key not in self.session_parameters:
                    self.session_parameters[key] = deepcopy(self.config_data)

        self.all_results = {}

        self.config_data['session_config_path'] = session_config
        self.config_data['config_file'] = config_file

        states = [0] * len(self.sessions.keys())

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

        # Set session select callback
        self.checked_list.observe(self.get_selected_session, names='value')

        # Update main configuration parameters
        self.minmax_heights.value = (self.config_data.get('min_height', 10), self.config_data.get('max_height', 100))
        self.dilate_iters.value = self.config_data.get('dilate_iterations', 0)

        if self.config_data.get('threads', 8) < 1:
            self.config_data['threads'] = 8

        for k in self.keys:
            self.session_parameters[k] = detect_and_set_camera_parameters(self.session_parameters[k], self.sessions[k])

        self.config_data = check_filter_sizes(self.config_data)
        self.config_data['pixel_areas'] = []
        self.config_data['autodetect'] = True
        self.config_data['detect'] = True
        if 'bg_roi_erode' not in self.config_data:
            self.config_data['bg_roi_erode'] = (1, 1)
        if 'bg_roi_dilate' not in self.config_data:
            self.config_data['bg_roi_dilate'] = (1, 1)

        if compute_bgs:
            self.compute_all_bgs()

    def generate_session_config(self, path):
        '''
        Generates the default/initial session configuration file.

        Returns
        -------
        '''
        warnings.warn('Session configuration file was not found. Generating a new one.')

        # Generate session config file if it does not exist
        with open(path, 'w+') as f:
            yaml.safe_dump(self.session_parameters, f)

    def compute_all_bgs(self):
        '''
        Computes all the background images before displaying the app to speed up user interaction.

        Returns
        -------
        '''

        for s, p in tqdm(self.sessions.items(), total=len(self.sessions.keys()), desc='Computing backgrounds'):
            try:
                # finfo is a key that points to a dict that contains the following keys:
                # ['file', 'dims', 'fps', 'nframes']. These are determined from moseq2-extract.io.video.get_video_info()
                if 'finfo' not in self.session_parameters[s]:
                    self.session_parameters[s]['finfo'] = get_movie_info(p)
                    if p.endswith('.mkv'):
                        self.session_parameters[s]['timestamps'] = load_timestamps_from_movie(p,
                                                                                              threads=self.config_data['threads'],
                                                                                              mapping=self.config_data.get('mapping', 'DEPTH'))
                        self.session_parameters[s]['finfo']['nframes'] = len(self.session_parameters[s]['timestamps'])

                # Compute background image; saving the image to a file
                self.session_parameters[s].pop('output_dir', None)
                get_bground_im_file(p, **self.session_parameters[s], output_dir=None)
            except:
                # Print error if an issue arises
                display(f'Error, could not compute background for session: {s}.')
                pass

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
                # finfo is a key that points to a dict that contains the following keys:
                # ['file', 'dims', 'fps', 'nframes']. These are determined from moseq2-extract.io.video.get_video_info()
                if 'finfo' not in self.session_parameters[sessionName]:
                    self.session_parameters[sessionName]['finfo'] = get_movie_info(sessionPath)

                if sessionPath.endswith('.mkv'):
                    self.session_parameters[sessionName]['timestamps'] = \
                        load_timestamps_from_movie(sessionPath, self.config_data['threads'],
                                                   self.config_data.get('mapping', 'DEPTH'))
                    self.session_parameters[sessionName]['finfo']['nframes'] = \
                        len(self.session_parameters[sessionName]['timestamps'])

                # Get background image for each session and test the current parameters on it
                self.session_parameters[sessionName].pop('output_dir', None)
                bground_im = get_bground_im_file(sessionPath, **self.session_parameters[sessionName])
                try:
                    sess_res = self.get_roi_and_depths(bground_im, sessionPath)
                except:
                    sess_res = {'flagged': True, 'ret_code': '0x1f534'}

                # Save session parameters if it is not flagged
                if not sess_res['flagged']:
                    self.npassing += 1

                # Update label
                self.checked_lbl.value = f'Sessions with Passing ROI Sizes: {self.npassing}/{len(self.checked_list.options)}'

                # Set index passing value
                checked_options[i] = f'{chr(int(sess_res["ret_code"], base=16))} {sessionName}'
                # Safely updating displayed list
                self.checked_list._initializing_traits_ = True
                self.checked_list.options = checked_options

                self.checked_list._initializing_traits_ = False

                # Updating progress
                self.all_results[sessionName] = sess_res['flagged']
                gc.collect()

        self.checked_list.value = checked_options[self.checked_list.index]

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
            self.session_parameters[curr_session_key]['timestamps'] = \
                load_timestamps_from_movie(self.curr_session, threads=self.config_data['threads'],
                                           mapping=self.config_data.get('mapping', 'DEPTH'))
            self.session_parameters[curr_session_key]['finfo']['nframes'] = \
                len(self.session_parameters[curr_session_key]['timestamps'])

        # Update sliders with corresponding session's previously set values
        if not isinstance(self.session_parameters[curr_session_key]['bg_roi_depth_range'], str):
            self.bg_roi_depth_range.value = self.session_parameters[curr_session_key]['bg_roi_depth_range']
        self.minmax_heights.value = [self.session_parameters[curr_session_key]['min_height'],
                                     self.session_parameters[curr_session_key]['max_height']]
        self.dilate_iters.value = self.session_parameters[curr_session_key]['dilate_iterations']

        # Get background and display UI plots
        self.session_parameters[curr_session_key].pop('output_dir', None)
        self.curr_bground_im = get_bground_im_file(self.curr_session, **self.session_parameters[curr_session_key])

        if self.main_out is None:
            self.main_out = widgets.interactive_output(self.interactive_depth_finder, {
                                                                                   'minmax_heights': self.minmax_heights,
                                                                                   'fn': self.frame_num,
                                                                                   'dr': self.bg_roi_depth_range,
                                                                                   'di': self.dilate_iters
                                                                                  }
                                                   )

        display(self.clear_button, self.ui_tools)
        # display(self.main_out)
        gc.collect()

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

    def interactive_depth_finder(self, minmax_heights, fn, dr, di):
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
            self.bg_roi_depth_range.value = self.session_parameters[curr_session_key]['bg_roi_depth_range']
            self.minmax_heights.value = [self.session_parameters[curr_session_key]['min_height'],
                                         self.session_parameters[curr_session_key]['max_height']]
        else:
            # Test updated parameters
            self.session_parameters[curr_session_key]['bg_roi_depth_range'] = (int(dr[0]), int(dr[1]))
            self.session_parameters[curr_session_key]['dilate_iterations'] = di

            if self.config_data['detect']:
                # Update the session flag result
                self.curr_results = self.get_roi_and_depths(self.curr_bground_im, self.curr_session)
                self.all_results[curr_session_key] = self.curr_results['flagged']

        # set indicator
        if self.curr_results['flagged']:
            self.indicator.value = '<center><h2><font color="red";>Flagged: Current ROI pixel area may be incorrect. If ROI is acceptable,' \
                                   ' Mark it as passing. Otherwise, change the depth range values.</h2></center>'
        else:
            self.indicator.value = '<center><h2><font color="green";>Passing</h2></center>'

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

        # initialize results dict
        curr_results = {'flagged': False,
                        'ret_code': "0x1f7e2"}

        curr_session_key = self.keys[self.checked_list.index]

        if self.config_data['autodetect']:
            # Get max depth as a thresholding limit (this would be the DTD if it already was computed)
            limit = np.max(bground_im)

            # Compute bucket distance thresholding value
            threshold_value = np.median(bground_im)
            self.session_parameters[curr_session_key]['bg_threshold'] = int(threshold_value)

            # Threshold image to find depth at bucket center: the true depth
            cX, cY = get_bucket_center(bground_im, limit, threshold=threshold_value)

            # True depth is at the center of the bucket
            self.true_depth = bground_im[cY][cX]
            self.session_parameters[curr_session_key]['true_depth'] = int(self.true_depth)

            # Get true depth range difference
            range_diff = 10 ** (len(str(int(self.true_depth))) - 1)

            # Center the depth ranges around the true depth
            bg_roi_range_min = int(self.true_depth - range_diff)
            bg_roi_range_max = int(self.true_depth + range_diff)

            self.config_data['bg_roi_depth_range'] = (bg_roi_range_min, bg_roi_range_max)

            if bg_roi_range_max > self.bg_roi_depth_range.max:
                self.bg_roi_depth_range.max = bg_roi_range_max + range_diff

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
            curr_results['flagged'] = True
            curr_results['ret_code'] = "0x1f534"

            # setting the roi variable to 1's array to match the background image. This way,
            # bokeh will still have an image to display.
            curr_results['roi'] = np.ones_like(self.curr_bground_im)
            self.update_checked_list(results=curr_results)
            return curr_results
        except ValueError:
            # bg depth range did not capture any area
            curr_results['flagged'] = True
            curr_results['ret_code'] = "0x1f534"
            curr_results['roi'] = np.ones_like(self.curr_bground_im)
            self.update_checked_list(results=curr_results)
            return curr_results
        except Exception as e:
            print(e)
            curr_results['flagged'] = True
            curr_results['ret_code'] = "0x1f534"
            curr_results['roi'] = np.ones_like(self.curr_bground_im)
            self.update_checked_list(results=curr_results)
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

        res = False
        # check if the current measured area is within is the current list of ROI areas
        for area in self.config_data.get('pixel_areas', []):
            if isclose(area, r, abs_tol=50e2) or r > area:
                res = True
                break
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
        if not res and (len(self.config_data.get('pixel_areas', [])) > 0):
            curr_results['flagged'] = True
            curr_results['ret_code'] = "0x1f534"
        else:
            # add accepted area size to
            self.config_data['pixel_areas'].append(r)

        # Save ROI
        curr_results['roi'] = rois[0]
        curr_results['counted_pixels'] = r
        self.update_checked_list(results=curr_results)
        gc.collect()

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
        raw_frames = load_movie_data(self.curr_session,
                                    range(fn, fn + 30),
                                    **self.session_parameters[curr_session_key],
                                    frame_size=self.curr_bground_im.shape[::-1])
        if not self.config_data.get('graduate_walls', False):
            curr_frame = (self.curr_bground_im - raw_frames)
        else:
            mouse_on_edge = (self.curr_bground_im < self.true_depth) & (raw_frames < self.curr_bground_im)
            curr_frame = (self.curr_bground_im - raw_frames) * np.logical_not(mouse_on_edge) + \
                         (self.true_depth - raw_frames) * mouse_on_edge

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

        if (result['depth_frames'] == np.zeros((1, self.config_data['crop_size'][0], self.config_data['crop_size'][1]))).all():
            if not self.curr_results['flagged']:
                self.indicator.value = '<center><h2><font color="red";>Flagged: Mouse Height threshold range is incorrect.</h2></center>'
                self.curr_results['flagged'] = True
        else:
            self.indicator.value = "<center><h2><font color='green';>Passing</h2></center>"
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
        url = upload_public(input_file)
        input_file = '/'.join(input_file.split('/')[-4:]) # truncate the base dir part of the file path
        video_div = f'''
                        <h2>{input_file}</h2>
                        <link rel="stylesheet" href="/nbextensions/google.colab/tabbar.css">
                        <video
                            src="{url}"; alt="{url}"; id="preview";
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