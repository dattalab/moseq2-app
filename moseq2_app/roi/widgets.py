'''
Ipywidgets used to facilitate interactive ROI detection
'''
import gc
import bokeh
from copy import deepcopy
import ruamel.yaml as yaml
import ipywidgets as widgets
from ipywidgets import VBox, HBox
from os.path import dirname, join
from moseq2_extract.io.image import write_image
from IPython.display import display, clear_output


class InteractiveROIWidgets:
    '''
    Class that contains Ipywidget widgets and layouts to facilitate interactive ROI finding functionality.
    This class is extended by the controller class InteractiveFindRoi.
    '''

    def __init__(self):
        '''
        Initializing all the ipywidgets widgets in a new context.
        '''

        style = {'description_width': 'initial'}

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.label_layout = widgets.Layout(display='flex', flex_flow='column', align_items='center', width='100%')
        self.layout_hidden = widgets.Layout(visibility='hidden', display='none')
        self.layout_visible = widgets.Layout(visibility='visible', display='inline-flex')

        self.message_layout = widgets.Layout(display='flex', align_items='center', width='inherit')

        # roi widgets
        self.roi_label = widgets.Label(value="ROI Parameters", layout=self.label_layout)
        self.bg_roi_depth_range = widgets.IntRangeSlider(value=[650, 750], min=0, max=1500, step=1,
                                                         description='Depth Range', continuous_update=False,
                                                         disable=False, style=style)
        self.dilate_iters = widgets.IntSlider(value=0, min=0, max=25, step=1, description='Dilate Iters:',
                                              continuous_update=False, style=style)
        self.frame_num = widgets.IntSlider(value=0, min=0, max=1000, step=1, description='Current Frame:',
                                           tooltip='Processed Frame Index to Display',
                                           disabled=False, continuous_update=False, style=style)

        self.toggle_autodetect = widgets.Checkbox(value=False, description='Autodetect Depth Range',
                                                  tooltip='Auto-detect depths', layout=widgets.Layout(display='none'))

        # extract widgets
        self.ext_label = widgets.Label(value="Extract Parameters", layout=self.label_layout)

        self.minmax_label = widgets.Label(value="Mouse Height Range to Include", layout=self.label_layout)
        self.minmax_heights = widgets.IntRangeSlider(value=[10, 100], min=0, max=400, step=1,
                                                     description='', style=style,
                                                     continuous_update=False)
        self.fr_label = widgets.Label(value="Frame Range to Extract", layout=self.label_layout)
        self.frame_range = widgets.IntRangeSlider(value=[0, 300], min=0, max=3000, step=1,
                                                  tooltip='Frames to Extract Sample',
                                                  description='', style=style, continuous_update=False)

        # check all button label
        self.checked_lbl = widgets.Label(value="Session Select", layout=self.label_layout,
                                         button_style='info', continuous_update=False)

        self.message = widgets.Label(value="", font_size=50, layout=self.message_layout)

        self.button_layout = widgets.Layout(flex_flow='column', align_items='center', width='80%')

        # buttons
        self.save_parameters = widgets.Button(description='Save Parameters', disabled=False, tooltip='Save Parameters')
        self.check_all = widgets.Button(description='Check All Sessions', disabled=False,
                                        tooltip='Extract full session using current parameters')

        self.extract_button = widgets.Button(description='Extract Sample', disabled=False, layout=self.button_layout,
                                             tooltip='Preview extraction output')
        self.mark_passing = widgets.Button(description='Save ROI', disabled=False,
                                           tooltip='If a session is incorrectly flagged, click this button to '
                                                   'mark the current session as passing, adding it to the list'
                                                   ' of acceptable ROI sizes, and saving it to a tiff file '
                                                   'to be reloaded later.',
                                           layout=self.label_layout)

        self.checked_list = widgets.Select(options=[], description='', continuous_update=False, disabled=False)

        self.box_layout = widgets.Layout(display='flex',
                                         justify_content='center',
                                         height='100%',
                                         align_items='center')

        self.column_layout = widgets.Layout(display='inline-flex', justify_content='space-between',
                                            align_items='center', height='100%', width='90%')

        self.row_layout = widgets.Layout(display='inline-flex',
                                         justify_content='space-between',
                                         border='solid',
                                         height='100%', width='100%')

        # groupings
        # ui widgets
        self.roi_tools = VBox([self.roi_label,
                               self.bg_roi_depth_range,
                               self.dilate_iters,
                               self.frame_num],
                              layout=self.column_layout)

        self.extract_tools = VBox([self.ext_label,
                                   VBox([self.minmax_label, self.minmax_heights]),
                                   VBox([self.fr_label, self.frame_range])],
                                   # self.extract_button
                                  layout=self.column_layout)

        self.button_box = VBox([HBox([self.check_all, self.save_parameters]),
                                self.checked_lbl,
                                self.checked_list,
                                self.mark_passing],
                               layout=self.column_layout)

        self.ui_tools = VBox([HBox([self.roi_tools, self.extract_tools, self.button_box], layout=self.row_layout),
                              self.message],
                             layout=self.box_layout)

        ### Event Listeners ###

        self.clear_button.on_click(self.clear_on_click)

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
        self.frame_num.observe(self.update_config_fn, names='value')

        self.bg_roi_depth_range.observe(self.update_config_dr, names='value')

        self.dilate_iters.observe(self.update_config_di, names='value')

    def clear_on_click(self, b=None):
        '''
        Clears the cell output

        Parameters
        ----------
        b (button click): User clicks Clear Button.

        Returns
        -------
        '''
        bokeh.io.curdoc().clear()
        bokeh.io.state.State().reset()
        bokeh.io.reset_output()
        clear_output()
        del self

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

        if event.old.split(' ')[1] != event.new.split(' ')[1]:
            self.checked_list.value = event.new

            gc.collect()
            bokeh.io.curdoc().clear()
            bokeh.io.state.State().reset()
            bokeh.io.reset_output()
            bokeh.io.output_notebook(hide_banner=True)
            clear_output(wait=True)
            self.main_out = None

            self.config_data['detect'] = True
            if self.autodetect_depths:
                self.config_data['autodetect'] = True
            self.interactive_find_roi_session_selector(self.checked_list.value)

    def extract_button_clicked(self, b=None):
        '''
        Updates the true depth autodetection parameter
         such that the true depth is autodetected for each found session

        Parameters
        ----------
        b (ipywidgets Button): Button click event.

        Returns
        -------
        '''
        clear_output(wait=True)
        display(self.clear_button, self.ui_tools)
        display(self.main_out)
        self.get_extraction(self.curr_session, self.curr_bground_im, self.curr_results['roi'])

    def mark_passing_button_clicked(self, b=None):
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
        self.config_data['pixel_areas'].append(self.curr_results['counted_pixels'])
        self.indicator.value = '<center><h2><font color="green";>Passing</h2></center>'

        self.update_checked_list(self.curr_results)

        # write ROI to correct path
        write_image(join(dirname(self.curr_session), 'proc', f'roi_00.tiff'), self.curr_results['roi'])

    def check_all_sessions(self, b=None):
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

    def save_clicked(self, b=None):
        '''
        Callback function to save the current session_parameters dict into
        the file of their choice (given in the top-most wrapper function).

        Parameters
        ----------
        b (button event): User click

        Returns
        -------
        '''

        self.config_data.pop('timestamps', None)
        saved_areas = self.config_data.pop('pixel_areas', None)
        for k in self.session_parameters.keys():
            self.session_parameters[k].pop('timestamps', None)
            self.session_parameters[k].pop('pixel_areas', None)

        # Update main config file
        with open(self.config_data['config_file'], 'w+') as f:
            yaml.safe_dump(self.config_data, f)

        # Update session parameters
        with open(self.config_data['session_config_path'], 'w+') as f:
            yaml.safe_dump(self.session_parameters, f)

        self.save_parameters.button_style = 'success'
        self.save_parameters.icon = 'check'
        self.config_data['pixel_areas'] = saved_areas

    def update_minmax_config(self, event=None):
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
        self.config_data['detect'] = True

    def update_config_dr(self, event=None):
        '''
        Callback function to update config dict with current UI depth range values

        Parameters
        ----------
        event (ipywidget callback): Any user interaction.

        Returns
        -------
        '''

        # setting detect to true in order to update the displayed ROI based on the current DR values
        self.config_data['detect'] = True

        self.config_data['bg_roi_depth_range'] = (int(self.bg_roi_depth_range.value[0]), int(self.bg_roi_depth_range.value[1]))

        self.session_parameters[self.keys[self.checked_list.index]]['bg_roi_depth_range'] = deepcopy(self.config_data['bg_roi_depth_range'])

    def update_config_di(self, event=None):
        '''
        Callback function to update config dict with current UI dilation iterations

        Parameters
        ----------
        event (ipywidget callback): Any user interaction.

        Returns
        -------
        '''

        self.config_data['dilate_iterations'] = int(self.dilate_iters.value)
        self.session_parameters[self.keys[self.checked_list.index]]['dilate_iterations'] = int(self.dilate_iters.value)
        self.config_data['detect'] = True

    def update_config_fr(self, event=None):
        '''
        Callback function to update config dict with current UI depth range values

        Parameters
        ----------
        event (ipywidget callback): Any user interaction.

        Returns
        -------
        '''

        self.config_data['frame_range'] = self.frame_range.value
        self.session_parameters[self.keys[self.checked_list.index]]['frame_range'] = self.frame_range.value
        self.config_data['detect'] = False

    def update_config_fn(self, event=None):
        '''
        Callback function to update config dict with current UI depth range values

        Parameters
        ----------
        event (ipywidget callback): Any user interaction.

        Returns
        -------
        '''

        self.config_data['frame_num'] = self.frame_num.value
        self.session_parameters[self.keys[self.checked_list.index]]['frame_num'] = self.frame_num.value
        self.config_data['detect'] = False