"""
Widgets module containing classes with components for each of the interactive syllable visualization tools
"""

import ipywidgets as widgets
from ipywidgets import HBox, VBox
from bokeh.models.widgets import PreText
from IPython.display import clear_output
from moseq2_app.viz.view import display_crowd_movies

class SyllableLabelerWidgets:

    def __init__(self):
        """
        launch the widget for labelling syllables with name and descriptions using the crowd movies.
        """
        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.syll_select = widgets.Dropdown(options={}, description='Syllable #:', disabled=False)

        # labels
        self.cm_lbl = PreText(text="Crowd Movie") # current crowd movie number

        self.syll_lbl = widgets.Label(value="Syllable Name") # name user prompt label
        self.desc_lbl = widgets.Label(value="Short Description") # description label

        self.syll_info_lbl = widgets.Label(value="Syllable Info", font_size=24)

        self.syll_usage_value_lbl = widgets.Label(value="")
        self.syll_speed_value_lbl = widgets.Label(value="")
        self.syll_duration_value_lbl = widgets.Label(value="")

        # text input widgets
        self.lbl_name_input = widgets.Text(value='',
                                           placeholder='Syllable Name',
                                           tooltip='2 word name for syllable')

        self.desc_input = widgets.Text(value='',
                                       placeholder='Short description of behavior',
                                       tooltip='Describe the behavior.',
                                       disabled=False)

        # buttons
        self.prev_button = widgets.Button(description='Prev', disabled=False, tooltip='Previous Syllable', layout=widgets.Layout(flex='2 1 0', width='auto', height='40px'))
        self.set_button = widgets.Button(description='Save Setting', disabled=False, tooltip='Save current inputs.', button_style='primary', layout=widgets.Layout(flex='3 1 0', width='auto', height='40px'))
        self.next_button = widgets.Button(description='Next', disabled=False, tooltip='Next Syllable', layout=widgets.Layout(flex='2 1 0', width='auto', height='40px'))

        # Box Layouts
        self.label_layout = widgets.Layout(flex_flow='column', height='100%')

        self.ui_layout = widgets.Layout(flex_flow='row', width='auto')

        self.data_layout = widgets.Layout(flex_flow='row', padding='top',
                                          align_content='center', justify_content='space-around',
                                          width='100%')

        self.data_col_layout = widgets.Layout(flex_flow='column',
                                              align_items='center',
                                              align_content='center',
                                              justify_content='space-around',
                                              width='100%')

        self.center_layout = widgets.Layout(justify_content='space-around',
                                            align_items='center')

        # label box
        self.lbl_box = VBox([self.syll_lbl, self.desc_lbl], layout=self.label_layout)

        # input box
        self.input_box = VBox([self.lbl_name_input, self.desc_input], layout=self.label_layout)

        # syllable info box
        self.info_boxes = VBox([self.syll_info_lbl], layout=self.center_layout)

        self.data_box = VBox([HBox([self.lbl_box, self.input_box], layout=self.data_layout), self.info_boxes],
                             layout=self.data_col_layout)

        # button box
        self.button_box = HBox([self.prev_button, self.set_button, self.next_button], layout=self.ui_layout)

    def clear_on_click(self, b=None):
        """
        Clear the cell output

        Args:
        b (button click)
        """

        clear_output()
        del self

    def on_next(self, event=None):
        """
        trigger an view update when the user clicks the "Next" button.

        Args:
        event (ipywidgets.ButtonClick): User clicks next button.
        """

        # Updating dict
        self.syll_info[self.syll_select.index]['label'] = self.lbl_name_input.value
        self.syll_info[self.syll_select.index]['desc'] = self.desc_input.value

        # Handle cycling through syllable labels
        if self.syll_select.index < len(self.syll_select.options) - 1:
            # Updating selection to trigger update
            self.syll_select.index += 1
        else:
            self.syll_select.index = 0

        # Updating input values with current dict entries
        self.lbl_name_input.value = self.syll_info[self.syll_select.index]['label']
        self.desc_input.value = self.syll_info[self.syll_select.index]['desc']

        self.write_syll_info(curr_syll=self.syll_select.index)

    def on_prev(self, event=None):
        """
        trigger an view update when the user clicks the "Previous" button.

        Args:
        event (ipywidgets.ButtonClick): User clicks 'previous' button.
        """

        # Update syllable information dict
        self.syll_info[self.syll_select.index]['label'] = self.lbl_name_input.value
        self.syll_info[self.syll_select.index]['desc'] = self.desc_input.value

        # Handle cycling through syllable labels
        if self.syll_select.index != 0:
            # Updating selection to trigger update
            self.syll_select.index -= 1
        else:
            self.syll_select.index = len(self.syll_select.options) - 1

        # Reloading previously inputted text area string values
        self.lbl_name_input.value = self.syll_info[self.syll_select.index]['label']
        self.desc_input.value = self.syll_info[self.syll_select.index]['desc']

        self.write_syll_info(curr_syll=self.syll_select.index)

    def on_set(self, event=None):
        """
        save the dict to syllable information file.

        Args:
        event (ipywidgets.ButtonClick): User clicks the 'Save' button.
        """

        # Update dict
        self.syll_info[self.syll_select.index]['label'] = self.lbl_name_input.value
        self.syll_info[self.syll_select.index]['desc'] = self.desc_input.value

        self.write_syll_info(curr_syll=self.syll_select.index)

        # Update button style
        self.set_button.button_style = 'success'

class CrowdMovieCompareWidgets:

    def __init__(self):
        """
        initialize crowd movie compare widget
        """
        style = {'description_width': 'initial'}

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.label_layout = widgets.Layout(flex_flow='column', max_height='100px')
        self.layout_hidden = widgets.Layout(display='none', align_items='stretch')
        self.layout_visible = widgets.Layout(display='flex',  align_items='stretch', justify_items='center')

        self.cm_syll_select = widgets.Dropdown(options=[], description='Syllable #:', disabled=False)
        self.num_examples = widgets.IntSlider(value=20, min=1, max=100, step=1, description='# of Example Mice:',
                                              disabled=False, continuous_update=False, style=style,
                                              layout=widgets.Layout(display='flex', align_items='stretch'))

        self.cm_sources_dropdown = widgets.Dropdown(options=['group', 'SessionName', 'SubjectName'], style=style,
                                                    description='Movie Sources:')

        self.cm_session_sel = widgets.SelectMultiple(options=[], description='Sessions:', rows=10,
                                                     style=style, layout=self.layout_hidden)

        self.cm_trigger_button = widgets.Button(description='Generate Movies',
                                                tooltip='Make Crowd Movies',
                                                layout=widgets.Layout(display='none', width='100%',
                                                                      align_items='stretch'))

        self.syllable_box = VBox([self.cm_syll_select, self.num_examples])

        self.session_box = VBox([self.cm_sources_dropdown, self.cm_session_sel, self.cm_trigger_button])

        self.widget_box = HBox([self.syllable_box, self.session_box],
                               layout=widgets.Layout(flex_flow='row',
                                                     border='solid',
                                                     width='100%',
                                                     justify_content='space-around'))

    def clear_on_click(self, b=None):
        """
        Clear the cell output

        Args:
        b (buttion click)
        """

        clear_output()

    def select_session(self, event=None):
        """
        get the session scalar information.

        Args:
        event (event): User clicks on multiple sessions in the SelectMultiple widget
        """

        # Set currently selected sessions
        self.config_data['session_names'] = list(self.cm_session_sel.value)

        # Update session_syllable info dict
        self.get_selected_session_syllable_info(self.config_data['session_names'])

    def show_session_select(self, change):
        """
        change current view to show session selector.

        Args:
        change (event): User switches their DropDownMenu selection
        """

        # Handle display syllable selection and update config_data crowd movie generation
        # source selector.
        options = [self.sorted_index['files'][s]['metadata'] for s in self.sessions]

        if change.new == 'SessionName':
            # Show session selector
            self.cm_session_sel.options = sorted([o['SessionName'] for o in options])
            self.cm_session_sel.layout = self.layout_visible
            self.cm_trigger_button.layout.display = 'block'
            self.config_data['separate_by'] = 'sessions'
        elif change.new == 'SubjectName':
            self.cm_session_sel.options = sorted([o['SubjectName'] for o in options])
            self.cm_session_sel.layout = self.layout_visible
            self.cm_trigger_button.layout.display = 'block'
            self.config_data['separate_by'] = 'subjects'
        elif change.new == 'group':
            # Hide session selector
            self.cm_session_sel.layout = self.layout_hidden
            self.cm_trigger_button.layout.display = 'none'
            self.config_data['separate_by'] = 'groups'

    def on_click_trigger_button(self, b=None):
        """
        Generate crowd movies and displays them when the user clicks the trigger button

        Args:
        b (ipywidgets.Button click event): User clicks "Generate Movies" button
        """

        syll_number = int(self.cm_syll_select.value.split(' - ')[0])

        # Compute current selected syllable's session dict.
        grouped_syll_dict = self.session_dict[syll_number]['session_info']

        self.config_data['session_names'] = list(grouped_syll_dict.keys())

        # Get Crowd Movie Divs
        divs, self.bk_plots = self.generate_crowd_movie_divs(grouped_syll_dict)

        # Display generated movies
        display_crowd_movies(self.widget_box, self.curr_label, self.curr_desc, divs, self.bk_plots)