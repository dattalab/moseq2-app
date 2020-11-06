import ipywidgets as widgets
from ipywidgets import HBox, VBox
from bokeh.models.widgets import PreText

class SyllableLabelerWidgets:

    def __init__(self):

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

        self.center_layout = widgets.Layout(display='flex', align_items='center')

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

class CrowdMovieCompareWidgets:

    def __init__(self):
        style = {'description_width': 'initial'}

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.label_layout = widgets.Layout(flex_flow='column', max_height='100px')
        self.layout_hidden = widgets.Layout(display='none', align_items='stretch')
        self.layout_visible = widgets.Layout(display='flex',  align_items='stretch', justify_items='center')

        self.cm_syll_select = widgets.Dropdown(options=[], description='Syllable #:', disabled=False)
        self.num_examples = widgets.IntSlider(value=20, min=1, max=40, step=1, description='# of Example Mice:',
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
