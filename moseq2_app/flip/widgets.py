"""
Widgets module containing all the interactive components of the frame selection GUI.
"""

import re
import h5py
import ipywidgets as widgets
from ipywidgets import VBox, HBox
from IPython.display import clear_output

class FlipClassifierWidgets:

    def __init__(self, path_dict, max_frames, continuous_update=True, launch_gui=True):
        """
        Initialize all flip classifier widgets

        Args:
        path_dict (dict): dict of sessionName-keys pointing to each session's results_00.h5 file (extraction output).
        max_frames (int): max number of frames to include in a dataset.
        continuous_update (bool): indicates whether to asynchronously reload the display as the user scrolls the slider.
        launch_gui (bool): indicates to callbacks to display the GUI using a FlipRangeTool function in flip.controller.py
        """

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

        # Widget values edited via callback functions
        # variables used in self.start_stop_frame_range()
        self.start = self.frame_num_slider.value
        self.stop = 0

        # variables used in self.on_delete_selection_clicked()
        self.max_frames = max_frames
        self.frame_ranges = []
        self.display_frame_ranges = []
        self.selected_ranges.options = self.frame_ranges

        # variable used in update_state_on_selected_range()
        self.path_dict = path_dict
        self.curr_total_selected_frames = 0

        # initialize selected frame range dictionary
        self.selected_frame_ranges_dict = {k: [] for k in self.path_dict}

        # variable for launching GUI view
        self.launch_gui = launch_gui

        # Callbacks
        self.clear_button.on_click(self.clear_on_click)
        self.start_button.on_click(self.start_stop_frame_range)
        self.face_left_button.on_click(self.facing_range_callback)
        self.face_right_button.on_click(self.facing_range_callback)
        self.delete_selection_button.on_click(self.on_delete_selection_clicked)
        self.selected_ranges.observe(self.on_selected_range_value, names='value')
        self.frame_num_slider.observe(self.curr_frame_update, names='value')

        # observe dropdown value changes
        self.session_select_dropdown.options = self.path_dict
        self.session_select_dropdown.observe(self.changed_selected_session, names='value')

    def clear_on_click(self, b=None):
        """
        Clear the output.

        Args:
        b (button click): user click the button
        """

        clear_output()

    def on_selected_range_value(self, event=None):
        """
        make the delete button visible once the user selects one of the frame ranges.
        """

        self.delete_selection_button.layout.visibility = 'visible'

    def start_stop_frame_range(self, b=None):
        """
        start and stop the "Add Range" functionality.

        Args:
        b (button click): User clicks on "Start or Stop" Range button.

        """
        # If user clicks the button == 'Start Range', then the function will start including frames in the correct flip set. 
        # After it is clicked, the button will function as a "Cancel Selection" button, hiding the direction selection buttons.

        if self.start_button.description == 'Start Range':
            self.start = self.frame_num_slider.value
            self.start_button.description = 'Cancel Select'
            self.start_button.button_style = 'danger'
            self.face_left_button.layout.visibility = 'visible'
            self.face_right_button.layout.visibility = 'visible'
        else:
            self.start_button.description = 'Start Range'
            self.start_button.button_style = 'info'
            self.face_left_button.layout.visibility = 'hidden'
            self.face_right_button.layout.visibility = 'hidden'

    def facing_range_callback(self, event):
        """
        handle when a user clicks either of the left or right facing buttons after selecting a frame range.
        """
        
        self.stop = self.frame_num_slider.value

        left = False
        if 'left' in event.description.lower():
            left = True

        if self.stop > self.start:
            self.update_state_on_selected_range(left)

            # Update left and right button visibility
            self.face_left_button.layout.visibility = 'hidden'
            self.face_right_button.layout.visibility = 'hidden'

            # Update range selection button
            self.start_button.description = 'Start Range'
            self.start_button.button_style = 'info'

    def update_state_on_selected_range(self, left):
        """
        Update the view upon a correct frame range addition (stop > start).

        Args:
        left (bool): Indicates which direction the mouse is facing the selected range. if True, facing left, else right.
        """

        # Updating list of displayed session + selected frame ranges
        selected_range = range(self.start, self.stop)
        display_selected_range = f'{self.session_select_dropdown.label} - {selected_range}'

        # Set the directional indicators in the displayed range list
        if left:
            display_selected_range = f'L - {display_selected_range}'
        else:
            display_selected_range = f'R - {display_selected_range}'

        self.curr_total_selected_frames += len(selected_range)

        # Update the current frame selector indicator
        old_lbl = self.curr_total_label.value
        old_val = re.findall(r': \d+', old_lbl)[0]
        new_val = old_lbl.replace(old_val, f': {str(self.curr_total_selected_frames)}')
        new_lbl = new_val.split('>')[3].replace('</h4', '')

        # Change indicator color to green if number of total selected
        # frames exceeds selected max number of frames
        if self.curr_total_selected_frames >= self.max_frames:
            new_val = f'<center><h4><font color="green";>{new_lbl}</h4></center>'
        self.curr_total_label.value = new_val

        # appending session list to get frames from for the flip classifier later on
        if selected_range not in self.selected_frame_ranges_dict[self.session_select_dropdown.label]:
            self.selected_frame_ranges_dict[self.session_select_dropdown.label] += [(left, selected_range)]

        # appending to frame ranges to display in table
        self.frame_ranges.append(selected_range)
        self.display_frame_ranges.append(display_selected_range)
        self.selected_ranges.options = self.display_frame_ranges

    def on_delete_selection_clicked(self, b=None):
        """
        delete the currently selected frame range from the list upon clicking the Delete button.

        Args:
        b (ipywidgets.Event): Button click event.
        """

        new_list = list(self.selected_ranges.options)

        if len(new_list) > 0:
            curr_index = new_list.index(self.selected_ranges.value)

            # parse selected frame range value
            vals = new_list[curr_index].split(' - ')
            delete_key = vals[1]
            direction = False if vals[0] == 'R' else True
            range_to_delete = eval(vals[2])

            # delete the selection from the session range dictionary
            to_drop = (direction, range_to_delete)
            self.selected_frame_ranges_dict[delete_key].remove(to_drop)

            # update the current total selected frames indicator
            self.curr_total_selected_frames -= len(list(range_to_delete))
            old_lbl = self.curr_total_label.value
            old_val = re.findall(r': \d+', old_lbl)[0]
            new_val = old_lbl.replace(old_val, f': {str(self.curr_total_selected_frames)}')
            new_lbl = new_val.split('>')[3].replace('</h4', '')

            if self.curr_total_selected_frames >= self.max_frames:
                new_val = f'<center><h4><font color="green";>{new_lbl}</h4></center>'
            else:
                new_val = f'<center><h4><font color="black";>{new_lbl}</h4></center>'
            self.curr_total_label.value = new_val

            # update the remainder of the helper lists
            new_list.pop(curr_index)
            self.frame_ranges.pop(curr_index)
            self.display_frame_ranges.pop(curr_index)
            self.selected_ranges.options = new_list

    def changed_selected_session(self, event=None):
        """
        load newly selected session.

        Args:
        event (ipywidgets Event): self.session_select_dropdown.value is changed
        """

        # check if button is in middle range selection
        if self.start_button.description == 'End Range':
            self.start_button.description = 'Start Range'
            self.start_button.button_style = 'info'

            self.start, self.stop = 0, 0

        # if so reset the button and start stop values
        with h5py.File(self.path_dict[self.session_select_dropdown.label], mode='r') as f:
            self.frame_num_slider.max = f['frames'].shape[0] - 1
        self.frame_num_slider.value = 0
        clear_output(wait=True)

        if self.launch_gui:
            self.interactive_launch_frame_selector()

    def curr_frame_update(self, event):
        """
        Update the currently displayed frame when the slider is moved.

        Args:
        event (ipywidgets Event): self.frame_num_slider.value is changed.
        """
        self.frame_num_slider.value = event['new']
        clear_output(wait=True)
        self.interactive_launch_frame_selector()
