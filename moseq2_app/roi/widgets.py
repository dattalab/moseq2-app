'''

Ipywidgets used to facilitate interactive ROI detection

'''

import ipywidgets as widgets
from ipywidgets import HBox, VBox


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

        # roi widgets
        self.roi_label = widgets.Label(value="ROI Parameters", layout=self.label_layout)
        self.bg_roi_depth_range = widgets.IntRangeSlider(value=[650, 750], min=0, max=1500, step=1,
                                                         description='Depth Range', continuous_update=False,
                                                         style=style)
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
        self.minmax_heights = widgets.IntRangeSlider(value=[13, 100], min=0, max=255, step=1,
                                                     description='', style=style,
                                                     continuous_update=False)
        self.fr_label = widgets.Label(value="Frame Range to Extract", layout=self.label_layout)
        self.frame_range = widgets.IntRangeSlider(value=[0, 300], min=0, max=3000, step=1,
                                                  tooltip='Frames to Extract Sample',
                                                  description='', style=style, continuous_update=False)

        # check all button label
        self.checked_lbl = widgets.Label(value="Session Select", layout=self.label_layout,
                                         button_style='info', continuous_update=False)

        self.message = widgets.Label(value="", font_size=50, layout=self.label_layout)

        self.button_layout = widgets.Layout(flex_flow='column', align_items='center', width='80%')

        # buttons
        self.save_parameters = widgets.Button(description='Save Parameters', disabled=False, tooltip='Save Parameters')
        self.check_all = widgets.Button(description='Check All Sessions', disabled=False,
                                        tooltip='Extract full session using current parameters')

        self.extract_button = widgets.Button(description='Extract Sample', disabled=False, layout=self.button_layout,
                                             tooltip='Preview extraction output')
        self.mark_passing = widgets.Button(description='Mark Passing', disabled=False, layout=self.label_layout,
                                           tooltip='Mark current session as passing')

        self.checked_list = widgets.Select(options=[], description='', continuous_update=False, disabled=False)

        self.box_layout = widgets.Layout(display='inline-flex',
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
                                   VBox([self.fr_label, self.frame_range]),
                                   self.extract_button],
                                  layout=self.column_layout)

        self.button_box = VBox([HBox([self.check_all, self.save_parameters]),
                                self.checked_lbl,
                                self.checked_list,
                                self.mark_passing],
                               layout=self.column_layout)

        self.ui_tools = VBox([HBox([self.roi_tools, self.extract_tools, self.button_box], layout=self.row_layout),
                              self.message],
                             layout=self.box_layout)