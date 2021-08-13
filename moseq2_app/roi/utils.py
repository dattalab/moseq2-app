'''

'''

import os
import gc
import cv2
import sys
import warnings
import numpy as np
from math import isclose
from copy import deepcopy
from os.path import exists
import ruamel.yaml as yaml
from tqdm.auto import tqdm
from os.path import join, dirname
from IPython.display import display
from moseq2_extract.util import detect_and_set_camera_parameters
from moseq2_extract.extract.proc import get_roi, get_bground_im_file
from moseq2_extract.io.video import get_movie_info, load_timestamps_from_movie
from moseq2_extract.util import (set_bground_to_plane_fit, select_strel,
                                 read_yaml, get_bucket_center, check_filter_sizes)
class InteractiveFindRoiUtilites:
    '''

    '''

    def __init__(self):
        '''
        '''
        pass

    def get_session_config(self, session_config, overwrite):
        '''

        Parameters
        ----------
        session_config
        overwrite

        Returns
        -------
        '''

        # Ensure config file is reading frames with >1 thread for fast reloading
        if self.config_data.get('threads', 8) < 1:
            self.config_data['threads'] = 8

        # Ensure config file contains all required parameters prior to creating session_config
        self.config_data = check_filter_sizes(self.config_data)

        # Set camera type if not supplied
        if 'camera_type' not in self.config_data:
            self.config_data['camera_type'] = 'auto'

        self.config_data['pixel_areas'] = []
        self.config_data['autodetect'] = True
        self.config_data['detect'] = True
        if 'bg_roi_erode' not in self.config_data:
            self.config_data['bg_roi_erode'] = (1, 1)
        if 'bg_roi_dilate' not in self.config_data:
            self.config_data['bg_roi_dilate'] = (1, 1)

        # Create initial shared session_parameters dict
        self.session_parameters = {k: deepcopy(self.config_data) for k in self.keys}

        # Update global session config: self.session_paramters
        self.config_data['session_config_path'] = session_config

        # Read individual session config if it exists
        if session_config is None:
            self.generate_session_config(session_config)
        elif not exists(session_config):
            self.generate_session_config(session_config)
        else:
            if exists(session_config) and not overwrite:
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

        # ensure that all the session's video reading parameters have been detected.
        for k in self.keys:
            self.session_parameters[k] = detect_and_set_camera_parameters(self.session_parameters[k], self.sessions[k])

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

    def check_roi_validity(self, n_pixels):
        '''

        Parameters
        ----------
        n_pixels

        Returns
        -------
        '''

        res = False
        pixel_areas = self.config_data.get('pixel_areas', [])

        # check if the current measured area is within is the current list of ROI areas
        if len(pixel_areas) < 3:
            abs_tol = 50e2
        else:
            abs_tol = np.std(pixel_areas)

        for area in pixel_areas:
            # if the current session's ROI is smaller than all the previously checked sessions
            # by at least 500 square pixels. Then this session's ROI will be flagged
            if isclose(area, n_pixels, abs_tol=abs_tol):
                res = True
                break

        # set flag to check whether this session's ROI has a comparable number of pixels
        # to the previously viewed sessions.
        if not res and (len(pixel_areas) > 0):
            self.curr_results['flagged'] = True
            if n_pixels < np.mean(pixel_areas):
                self.curr_results['err_code'] = 4
            else:
                self.curr_results['err_code'] = 3
            self.curr_results['ret_code'] = "0x1f534"
        else:
            # add accepted area size to
            self.curr_results['flagged'] = False
            self.curr_results['err_code'] = -1
            self.curr_results['ret_code'] = "0x1f7e2"
            self.config_data['pixel_areas'].append(n_pixels)

    def autodetect_depth_range(self, curr_session_key):
        '''

        Parameters
        ----------
        curr_session_key

        Returns
        -------
        '''

        if self.config_data['autodetect']:
            # Get max depth as a thresholding limit (this would be the DTD if it already was computed)
            limit = np.max(self.curr_bground_im)

            # Compute bucket distance thresholding value
            threshold_value = np.median(self.curr_bground_im)
            self.session_parameters[curr_session_key]['bg_threshold'] = int(threshold_value)

            # Threshold image to find depth at bucket center: the true depth
            try:
                cX, cY = get_bucket_center(self.curr_bground_im, limit, threshold=threshold_value)
            except ZeroDivisionError as e:
               print(e)
               self.curr_results['err_code'] = 0
               cY, cX = (int(x) for x in self.curr_bground_im.shape)

            # True depth is at the center of the bucket
            self.true_depth = self.curr_bground_im[cY][cX]
            self.session_parameters[curr_session_key]['true_depth'] = int(self.true_depth)

            # Get true depth range difference
            range_diff = 10 ** (len(str(int(self.true_depth))) - 1)

            # Center the depth ranges around the true depth
            bg_roi_range_min = int(self.true_depth - range_diff)
            bg_roi_range_max = int(self.true_depth + range_diff)

            self.session_parameters[curr_session_key]['bg_roi_depth_range'] = (bg_roi_range_min, bg_roi_range_max)

            if bg_roi_range_max > self.bg_roi_depth_range.max:
                self.bg_roi_depth_range.max = bg_roi_range_max + range_diff

    def handle_mkv_files(self, session_key, session_path):
        '''

        Parameters
        ----------
        session_key
        session_path
        Returns
        -------
        '''

        self.session_parameters[session_key]['timestamps'] = load_timestamps_from_movie(session_path,
                                                                                        threads=self.config_data['threads'],
                                                                                        mapping=self.config_data.get('mapping', 'DEPTH'))
        self.session_parameters[session_key]['finfo']['nframes'] = len(self.session_parameters[session_key]['timestamps'])

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
                        self.handle_mkv_files(s, p)

                # Compute background image; saving the image to a file
                self.session_parameters[s].pop('output_dir', None)
                get_bground_im_file(p, **self.session_parameters[s], output_dir=None)
            except Exception as e:
                # Print error if an issue arises
                tb = sys.exc_info()[2]
                display(f'{e.with_traceback(tb=tb)}')
                self.curr_results['err_code'] = 0
                pass

    def get_roi_and_depths(self):
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

        curr_session_key = self.keys[self.checked_list.index]

        # Compute suitable depth range based on depth value at bucket centroid
        self.autodetect_depth_range(curr_session_key)

        # Get relevant structuring elements
        strel_dilate = select_strel(self.config_data['bg_roi_shape'], tuple(self.config_data['bg_roi_dilate']))
        strel_erode = select_strel(self.config_data['bg_roi_shape'], tuple(self.config_data['bg_roi_erode']))

        # get the current background image
        self.session_parameters[curr_session_key].pop('output_dir', None)
        self.curr_bground_im = get_bground_im_file(self.curr_session, **self.session_parameters[curr_session_key])

        try:
            # Get ROI
            rois, plane, bboxes, _, _, _ = get_roi(self.curr_bground_im,
                                                   **self.session_parameters[curr_session_key],
                                                   strel_dilate=strel_dilate,
                                                   strel_erode=strel_erode,
                                                   get_all_data=True
                                                   )
        except ValueError:
            # bg depth range did not capture any area
            # flagged + ret_code are used to display a red circle in the session selector to indicate a failed
            # roi detection.
            self.curr_results['flagged'] = True
            self.curr_results['err_code'] = 2
            self.curr_results['ret_code'] = "0x1f534"

            # setting the roi variable to 1's array to match the background image. This way,
            # bokeh will still have an image to display.
            self.curr_results['roi'] = np.ones_like(self.curr_bground_im)
            self.update_checked_list()
            return
        except Exception as e:
            # catching any remaining possible exceptions to preserve the integrity of the interactive GUI.
            self.curr_results['flagged'] = True
            self.curr_results['err_code'] = 2
            self.curr_results['ret_code'] = "0x1f534"
            self.curr_results['roi'] = np.ones_like(self.curr_bground_im)
            self.update_checked_list()
            return

        if self.config_data['use_plane_bground']:
            print('Using plane fit for background...')
            self.curr_bground_im = set_bground_to_plane_fit(self.curr_bground_im, plane, join(dirname(self.curr_session), 'proc'))

        if self.config_data['autodetect']:
            # Corresponds to a rough pixel area estimate
            r = float(cv2.countNonZero(rois[0].astype('uint16')))
            self.config_data['pixel_areas'].append(r)
            self.session_parameters[curr_session_key]['pixel_area'] = r
        else:
            # Corresponds to a rough pixel area estimate
            r = float(cv2.countNonZero(rois[0].astype('uint16')))
            self.session_parameters[curr_session_key]['pixel_area'] = r

        self.check_roi_validity(n_pixels=r)

        # Save ROI
        self.curr_results['roi'] = rois[0]
        self.curr_results['counted_pixels'] = r

        # Update results
        self.update_checked_list()
        gc.collect()

        return

    def get_all_session_roi_results(self, session_dict):
        '''

        Parameters
        ----------
        session_dict

        Returns
        -------

        '''

        checked_options = list(self.checked_list.options)

        # test saved config data parameters on all sessions
        for i, (sessionName, sessionPath) in enumerate(session_dict.items()):
            self.curr_session = sessionPath
            # finfo is a key that points to a dict that contains the following keys:
            # ['file', 'dims', 'fps', 'nframes']. These are determined from moseq2-extract.io.video.get_video_info()
            if 'finfo' not in self.session_parameters[sessionName]:
                self.session_parameters[sessionName]['finfo'] = get_movie_info(self.curr_session)

            if self.curr_session.endswith('.mkv'):
                self.handle_mkv_files(sessionName, self.curr_session)

            # Get background image for each session and test the current parameters on it
            self.session_parameters[sessionName].pop('output_dir', None)
            self.curr_bground_im = get_bground_im_file(self.curr_session, **self.session_parameters[sessionName])
            self.get_roi_and_depths()

            # Save session parameters if it is not flagged
            if not self.curr_results['flagged']:
                self.npassing += 1

            # Update nPassing Sessions label in display
            self.checked_lbl.value = f'Sessions with Passing ROI Sizes: {self.npassing}/{len(self.checked_list.options)}'

            # Set index passing value
            checked_options[i] = f'{chr(int(self.curr_results["ret_code"], base=16))} {sessionName}'

            # Safely updating displayed list
            self.checked_list._initializing_traits_ = True
            self.checked_list.options = checked_options
            self.checked_list._initializing_traits_ = False

            # Updating progress
            self.all_results[sessionName] = self.curr_results['flagged']
            gc.collect()

        self.checked_list.value = checked_options[self.checked_list.index]