'''

Widgets module containing all the interactive components of the frame selection GUI.

'''

from ipywidgets import VBox, HBox
import ipywidgets as widgets

class FlipClassifierWidgets:

    def __init__(self, continuous_update):
        '''
        Initializes all flip classifier widgets
        '''

        style = {'description_width': 'initial'}
        layout = widgets.Layout(width='90%')
        hidden = widgets.Layout(visibility='hidden')
        self.session_select_dropdown = widgets.Dropdown(options=[], style=style, layout={'width': '50%'},
                                                        description='Session to Select Frames From',
                                                        continuous_update=True,
                                                        disabled=False)

        self.frame_num_slider = widgets.IntSlider(value=0, min=0, max=1000, step=1, description='Current Frame:',
                                                  tooltip='Processed Frame Index to Display', layout=layout,
                                                  disabled=False, continuous_update=continuous_update, style=style)

        self.start_button = widgets.Button(description='Start Range', disabled=False,
                                           button_style='info', tooltip='Frame Range Selector')

        self.face_left_button = widgets.Button(id='left', description='Facing Left', disabled=False, layout=hidden,
                                           button_style='info', tooltip='Indicate mouse is facing left')
        self.face_right_button = widgets.Button(id='right', description='Facing Right', disabled=False, layout=hidden,
                                           button_style='info', tooltip='Indicate mouse is facing right')
        
        self.box_layout = widgets.Layout(align_items='stretch', width='100%')

        self.selector_box = HBox([self.start_button, self.face_left_button, self.face_right_button], layout=self.box_layout)

        self.button_box = VBox([self.frame_num_slider, self.selector_box])

        self.curr_total_label = widgets.HTML(value="<center><h4>Current Total Selected Frames: 0</h4></center>", layout=layout)

        self.selected_ranges_label = widgets.Label('Selected Correct Frame Ranges')
        self.selected_ranges = widgets.Select(options=[], description='', layout=widgets.Layout(height='100%', width='auto'),
                                              continuous_update=False, disabled=True)

        self.range_box = VBox([self.curr_total_label, self.selected_ranges_label, self.selected_ranges])

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')