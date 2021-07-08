'''

Widgets module containing all the interactive components of the frame selection GUI.

'''

import ipywidgets as widgets
from ipywidgets import VBox, HBox
from IPython.display import clear_output

class FlipClassifierWidgets:

    def __init__(self, continuous_update):
        '''
        Initializes all flip classifier widgets
        '''

        style = {'description_width': 'initial'}
        layout = widgets.Layout(width='90%')
        hidden = widgets.Layout(visibility='hidden')
        dir_button_hidden = widgets.Layout(visibility='hidden')

        self.session_select_dropdown = widgets.Dropdown(options=[], style=style, layout={'width': '50%'},
                                                        description='Session to Select Frames From',
                                                        continuous_update=True,
                                                        disabled=False)

        self.frame_num_slider = widgets.IntSlider(value=0, min=0, max=1000, step=1, description='Current Frame:',
                                                  tooltip='Processed Frame Index to Display', layout=layout,
                                                  disabled=False, continuous_update=continuous_update, style=style)

        self.start_button = widgets.Button(description='Start Range', disabled=False,
                                           button_style='info', tooltip='Frame Range Selector')

        self.face_left_button = widgets.Button(id='left', description='Facing Left', disabled=False,
                                               layout=dir_button_hidden, button_style='info',
                                               tooltip='Indicate mouse is facing left')
        self.face_right_button = widgets.Button(id='right', description='Facing Right', disabled=False,
                                                layout=dir_button_hidden, button_style='info',
                                                tooltip='Indicate mouse is facing right')
        
        self.box_layout = widgets.Layout(align_items='stretch', width='100%')

        self.selector_box = HBox([self.start_button, self.face_left_button, self.face_right_button], layout=self.box_layout)

        self.button_box = VBox([self.frame_num_slider, self.selector_box])

        self.curr_total_label = widgets.HTML(value='<center><h4><font color="black";>Current Total Selected Frames: 0</h4></center>', layout=layout)

        self.selected_ranges_label = widgets.Label('Selected Correct Frame Ranges')
        self.selected_ranges = widgets.Select(options=[], description='', layout=widgets.Layout(height='100%', width='auto'),
                                              continuous_update=False, disabled=False)

        self.delete_selection_button = widgets.Button(id='delete', description='Delete Selection',
                                                      disabled=False, layout=hidden,
                                                      button_style='danger', tooltip='Delete current selection')

        self.range_box = VBox([self.curr_total_label, self.selected_ranges_label,
                               self.selected_ranges, self.delete_selection_button])

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

    def clear_on_click(self, b=None):
        '''
        Clears the output.

        Parameters
        ----------
        b (button click)

        Returns
        -------
        '''

        clear_output()

    def on_selected_range_value(self, event=None):
        '''
        Callback function to make the delete button visible once the user selects one of the frame ranges.

        Returns
        -------
        '''

        self.delete_selection_button.layout.visibility = 'visible'
