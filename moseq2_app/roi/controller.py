'''

Interactive ROI detection and extraction preview functionalities. This module utilizes the widgets from
the widgets.py file to facilitate the real-time interaction.

'''

import gc
import os
import numpy as np
from bokeh.io import show
import ipywidgets as widgets
from bokeh.models import Div, CustomJS, Slider
from IPython.display import display, clear_output
from moseq2_app.gui.progress import get_session_paths
from moseq2_extract.util import get_strels, read_yaml
from moseq2_extract.extract.extract import extract_chunk
from moseq2_app.roi.widgets import InteractiveROIWidgets
from moseq2_extract.extract.proc import get_bground_im_file
from moseq2_app.roi.utils import InteractiveFindRoiUtilites
from os.path import dirname, basename, join, relpath, abspath
from moseq2_app.roi.view import plot_roi_results, show_extraction
from moseq2_extract.extract.proc import apply_roi, threshold_chunk
from moseq2_extract.helpers.extract import process_extract_batches
from moseq2_extract.io.video import load_movie_data, get_video_info, get_movie_info


class InteractiveFindRoi(InteractiveROIWidgets, InteractiveFindRoiUtilites):

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

        # initialize reusable grid and extraction output widgets
        self.main_output = widgets.Output(layout=widgets.Layout(align_items='center'))
        self.extraction_output = widgets.Output(layout=widgets.Layout(align_items='center', display='inline-block', height='100%', width='100%'))

        # Read default config parameters
        self.config_data = read_yaml(config_file)
        self.config_data['config_file'] = config_file

        # Update DropDown menu items
        self.sessions = get_session_paths(data_path)

        assert len(self.sessions) > 0, "No sessions were found in the provided base_dir"

        # Session selection dict key names
        self.keys = list(self.sessions.keys())

        self.get_session_config(session_config=session_config, overwrite=overwrite)

        # Create colored dots for each session item in the checked_list widget
        states = [0] * len(self.sessions.keys())
        c_base = int("1F534", base=16)
        options = list(self.sessions.keys())
        colored_options = ['{} {}'.format(chr(c_base + s), o) for s, o in zip(states, options)]

        # Set Initial List options and session to display without triggering the view to reload
        self.safe_widget_value_update(wid_obj=self.checked_list,
                                      func=self.get_selected_session,
                                      new_val=colored_options[0],
                                      new_options=colored_options)

        # Update display with loaded configuration parameters: minmax and dilate_iters
        self.safe_widget_value_update(wid_obj=self.minmax_heights,
                                      func=self.update_minmax_config,
                                      new_val=(self.config_data.get('min_height', 10), self.config_data.get('max_height', 100)))

        self.safe_widget_value_update(wid_obj=self.dilate_iters,
                                      func=self.update_config_di,
                                      new_val=self.config_data.get('dilate_iterations', 0))

        if compute_bgs:
            self.compute_all_bgs()

        # start tool execution
        self.interactive_find_roi_session_selector(self.checked_list.value)
        display(self.clear_button, self.ui_tools, self.indicator, self.main_output, self.extraction_output)

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
            self.handle_mkv_files(curr_session_key, self.curr_session)

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

        curr_session_key = self.keys[self.checked_list.index]

        self.save_parameters.button_style = 'primary'
        self.save_parameters.icon = 'none'

        # Autodetect reference depth range and min-max height values at launch
        if self.config_data['autodetect']:
            self.get_roi_and_depths()
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
            dr = self.bg_roi_depth_range.value
            self.session_parameters[curr_session_key]['bg_roi_depth_range'] = (int(dr[0]), int(dr[1]))
            self.session_parameters[curr_session_key]['dilate_iterations'] = self.dilate_iters.value

            if self.config_data['detect']:
                # Update the session flag result
                self.get_roi_and_depths()
                self.all_results[curr_session_key] = self.curr_results['flagged']

        # display graphs
        self.prepare_data_to_plot(self.curr_results['roi'], self.minmax_heights.value, self.frame_num.value)
        gc.collect()

    def test_all_sessions(self, session_dict):
        '''
        Helper to a callback function to test the current configurable UI values on all the
        sessions that were found. Triggered when user clicks "Check All Sessions".

        The function will call a helper utility get_all_session_roi_results() in order to iteratively check each session,
         and update the displayed "circle indicator" color to green if the test passes, and red if it fails.

        If all the sessions pass, the session_config.yaml file will be saved to file and a message is emited indicating that
         the user can close the gui and move to the "Extract All Sessions" cell. If some sessions, fail the emitted message
         will indicate to check these sessions and update their parameters prior to continuing to the extract step.

        Parameters
        ----------
        session_dict (dict): dict of session directory names paired with their absolute paths.

        Returns
        -------
        '''

        self.get_all_session_roi_results(session_dict)

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
        self.extraction_output = show_extraction(basename(dirname(input_file)), view_path, main_output=self.extraction_output)
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
                if self.curr_results['err_code'] < 0:
                    self.curr_results['err_code'] = 5
                self.curr_results['flagged'] = True

        # filter for included mouse height range
        try:
            filtered_frames = threshold_chunk(filtered_frames, minmax_heights[0], minmax_heights[1])
        except:
            # Display min-max heights error and flag
            filtered_frames = curr_frame.copy()[0]
            if not self.curr_results['flagged']:
                if self.curr_results['err_code'] < 0:
                    self.curr_results['err_code'] = 6
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
            self.curr_results['flagged'] = True
            if self.curr_results['err_code'] < 0:
                self.curr_results['err_code'] = 7
            self.curr_results['ret_code'] = "0x1f534"

        if self.config_data.get('camera_type', 'kinect') == 'azure':
            # orienting preview images to match sample extraction
            display_bg = np.flip(self.curr_bground_im.copy(), 0)
            overlay = np.flip(overlay, 0) # overlayed roi
            filtered_frames = np.flip(filtered_frames, 0) # segmented
        else:
            display_bg = self.curr_bground_im

        # Update indicator value with final error
        error_code = self.curr_results['err_code']
        err_formatting_s, formatting_e = '<center><h2><font color="red";>', '</h2></center>'
        if error_code < 0:
            final_indicator_value = f'<center><h2><font color="green";>{self.error_codes[error_code]}</h2></center>'
        else:
            final_indicator_value = err_formatting_s+self.error_codes[error_code]+formatting_e

        # Make and display plots
        self.main_output = plot_roi_results(self.formatted_key, display_bg, roi, overlay, filtered_frames, result['depth_frames'][0], fn, main_out=self.main_output)

        # update text indicator value
        self.indicator.value = final_indicator_value
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