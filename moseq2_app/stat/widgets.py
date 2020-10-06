import ipywidgets as widgets
from ipywidgets import HBox, VBox

class SyllableStatWidgets:

    def __init__(self):

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.layout_hidden = widgets.Layout(display='none')
        self.layout_visible = widgets.Layout(display='block')

        self.stat_dropdown = widgets.Dropdown(options=['usage', 'Centroid Speed', '2D Velocity', '3D Velocity', 'Height', 'Distance to Center'], description='Stat to Plot:', disabled=False)

        # add dist to center
        self.sorting_dropdown = widgets.Dropdown(options=['usage', 'Centroid Speed', '2D Velocity', '3D Velocity', 'Height', 'Distance to Center', 'Similarity', 'Difference'], description='Sorting:', disabled=False)
        self.ctrl_dropdown = widgets.Dropdown(options=[], description='Group 1:', disabled=False)
        self.exp_dropdown = widgets.Dropdown(options=[], description='Group 2:', disabled=False)

        self.grouping_dropdown = widgets.Dropdown(options=['group', 'SessionName'], description='Grouping:', disabled=False)
        self.session_sel = widgets.SelectMultiple(options=[], description='Sessions:', rows=10,
                                                  layout=self.layout_hidden, disabled=False)

        self.errorbar_dropdown = widgets.Dropdown(options=['SEM', 'STD'], description='Error Bars:', disabled=False)

        ## boxes
        self.stat_box = VBox([self.stat_dropdown, self.errorbar_dropdown])
        self.mutation_box = VBox([self.ctrl_dropdown, self.exp_dropdown])

        self.sorting_box = VBox([self.sorting_dropdown, self.mutation_box])
        self.session_box = VBox([self.grouping_dropdown, self.session_sel])

        self.stat_widget_box = HBox([self.stat_box, self.sorting_box, self.session_box])