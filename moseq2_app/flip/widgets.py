from ipywidgets import VBox
import ipywidgets as widgets

class FlipClassifierWidgets:

    def __init__(self):
        '''
        Initializes all flip classifier widgets
        '''

        style = {'description_width': 'initial'}
        layout = widgets.Layout(width='90%')
        self.frame_num_slider = widgets.IntSlider(value=0, min=0, max=1000, step=1, description='Current Frame:',
                                                  tooltip='Processed Frame Index to Display', layout=layout,
                                                  disabled=False, continuous_update=False, style=style)

        self.start_button = widgets.Button(description='Start Range', disabled=False,
                                           button_style='info', tooltip='Frame Range Selector')

        self.button_box = VBox([self.frame_num_slider, self.start_button])

        self.selected_ranges_label = widgets.Label('Selected Correct Frame Ranges')
        self.selected_ranges = widgets.Select(options=[], description='', layout=widgets.Layout(height='100%'),
                                              continuous_update=False, disabled=True)

        self.range_box = VBox([self.selected_ranges_label, self.selected_ranges])

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')