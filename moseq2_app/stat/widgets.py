'''

Widgets module containing classes with components for each of the interactive syllable
 statistics tools: Syllable Stats plot, and Transition Graph plot.

'''

import numpy as np
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import clear_output

class SyllableStatWidgets:

    def __init__(self):
        style = {'description_width': 'initial', 'display': 'flex-grow', 'align_items': 'stretch'}

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.layout_hidden = widgets.Layout(display='none')
        self.layout_visible = widgets.Layout(display='block')

        self.stat_dropdown = widgets.Dropdown(options=['usage', '2D Velocity', '3D Velocity', 'Height', 'Distance to Center'], description='Stat to Plot:', disabled=False)

        self.sorting_dropdown = widgets.Dropdown(options=['usage', '2D Velocity', '3D Velocity', 'Height', 'Distance to Center', 'Similarity', 'Difference'], description='Sorting:', disabled=False)
        self.thresholding_dropdown = widgets.Dropdown(
            options=['usage', '2D Velocity', '3D Velocity', 'Height', 'Distance to Center'],
            description='Threshold By:', disabled=False, style=style)

        self.ctrl_dropdown = widgets.Dropdown(options=[], description='Group 1:', disabled=False)
        self.exp_dropdown = widgets.Dropdown(options=[], description='Group 2:', disabled=False)

        self.grouping_dropdown = widgets.Dropdown(options=['group', 'SessionName', 'SubjectName'], description='Grouping:', disabled=False)
        self.session_sel = widgets.SelectMultiple(options=[], description='Sessions:', rows=10,
                                                  layout=self.layout_hidden, disabled=False)

        self.errorbar_dropdown = widgets.Dropdown(options=['CI 95%', 'SEM', 'STD'], description='Error Bars:', disabled=False)

        ## boxes
        self.data_layout = widgets.Layout(flex_flow='row', padding='top', justify_content='space-around', width='100%')

        self.stat_box = VBox([self.stat_dropdown, self.errorbar_dropdown])
        self.mutation_box = VBox([self.ctrl_dropdown, self.exp_dropdown])

        self.sorting_box = VBox([self.sorting_dropdown, self.mutation_box, self.thresholding_dropdown])
        self.session_box = VBox([self.grouping_dropdown, self.session_sel])

        self.stat_widget_box = VBox([HBox([self.stat_box, self.sorting_box, self.session_box])])

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
        del self

    def on_grouping_update(self, event):
        '''
        Updates the MultipleSelect widget upon selecting groupby == SubjectName or SessionName.
        Hides it if groupby == group.

        Parameters
        ----------
        event (user clicks new grouping)

        Returns
        -------
        '''

        if event.new == 'SessionName':
            self.session_sel.layout.display = "flex"
            self.session_sel.layout.align_items = 'stretch'
            self.session_sel.options = self.session_names
        elif event.new == 'SubjectName':
            self.session_sel.layout.display = "flex"
            self.session_sel.layout.align_items = 'stretch'
            self.session_sel.options = self.subject_names
        else:
            self.session_sel.layout.display = "none"

        self.session_sel.value = [self.session_sel.options[0]]

class TransitionGraphWidgets:


    '''
    edge_thresholder = widgets.FloatRangeSlider(value=[0.0025, 1], min=0, max=1, step=0.001, style=style, readout_format='.4f',
                                                description='Edges weights to display', continuous_update=False)
    usage_thresholder = widgets.FloatRangeSlider(value=[0, 1], min=0, max=1, step=0.001, style=style, readout_format='.4f',
                                                description='Usage nodes to display', continuous_update=False)
    speed_thresholder = widgets.FloatRangeSlider(value=[-25, 200], min=-50, max=200, step=1, style=style, readout_format='.1f',
                                                description='Threshold nodes by speed', continuous_update=False)
    '''

    def __init__(self):
        style = {'description_width': 'initial', 'display': 'flex-grow', 'align_items': 'stretch'}

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

        col1_layout = widgets.Layout(flex_flow='column', width='75%', align_items='center')
        col2_layout = widgets.Layout(flex_flow='column', align_contents='center', width='75%', align_items='center')

        ui_layout = widgets.Layout(flex_flow='row', border='solid', align_items='stretch',
                                   width='100%', justify_content='space-around')

        self.graph_layout_dropdown = widgets.Dropdown(options=['circular',  'spring', 'spectral'],
                                                      description='Graph Layout',
                                                      style=style, value='circular', continuous_update=False,
                                                      layout=widgets.Layout(align_items='stretch', width='80%'))

        self.color_nodes_dropdown = widgets.Dropdown(options=['Default', '2D velocity',
                                                              '3D velocity', 'Height', 'Distance to Center',
                                                              'Entropy-In', 'Entropy-Out'],
                                                     description='Node Coloring',
                                                     style=style, value='Default', continuous_update=False,
                                                     layout=widgets.Layout(align_items='stretch', width='80%'))

        self.edge_thresholder = widgets.SelectionRangeSlider(options=['tmp'], style=style,
                                                             description='Threshold Edge Weights',
                                                             layout=widgets.Layout(align_items='stretch', width='90%'),
                                                             continuous_update=False)
        self.usage_thresholder = widgets.SelectionRangeSlider(options=['tmp'], style=style, readout_format='.4f',
                                                              description='Threshold Nodes by Usage',
                                                              layout=widgets.Layout(align_items='stretch', width='90%'),
                                                              continuous_update=False)
        self.speed_thresholder = widgets.SelectionRangeSlider(options=['tmp'], style=style, readout_format='.1f',
                                                              description='Threshold Nodes by Speed',
                                                              layout=widgets.Layout(align_items='stretch', width='90%'),
                                                              continuous_update=False)

        self.thresholding_box = HBox([
                                      VBox([self.graph_layout_dropdown, self.edge_thresholder, self.usage_thresholder],
                                           layout=col1_layout),
                                      VBox([self.color_nodes_dropdown, self.speed_thresholder], layout=col2_layout)],
                                           layout=ui_layout)

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

    def set_range_widget_values(self):
        '''
        After the dataset is initialized, the threshold range sliders' values will be set
         according to the standard deviations of the dataset.

        Returns
        -------
        '''

        # Update threshold range values
        edge_threshold_stds = int(np.max(self.trans_mats) / np.std(self.trans_mats))
        usage_threshold_stds = int(self.df['usage'].max() / self.df['usage'].std()) + 2
        speed_threshold_stds = int(self.df['velocity_2d_mm'].max() / self.df['velocity_2d_mm'].std()) + 2

        self.edge_thresholder.options = [float('%.3f' % (np.std(self.trans_mats) * i)) for i in
                                         range(edge_threshold_stds)]
        self.edge_thresholder.index = (1, edge_threshold_stds - 1)

        self.usage_thresholder.options = [float('%.3f' % (self.df['usage'].std() * i)) for i in
                                          range(usage_threshold_stds)]
        self.usage_thresholder.index = (0, usage_threshold_stds - 1)

        self.speed_thresholder.options = [float('%.3f' % (self.df['velocity_2d_mm'].std() * i)) for i in
                                          range(speed_threshold_stds)]
        self.speed_thresholder.index = (0, speed_threshold_stds - 1)

    def on_set_scalar(self, event):
        '''
        Updates the scalar threshold slider filter criteria according to the current node coloring.
        Changes the name of the slider as well.

        Parameters
        ----------
        event (dropdown event): User changes selected dropdown value

        Returns
        -------
        '''

        if event.new == 'Default' or event.new == '2D velocity':
            key = 'velocity_2d_mm'
            self.speed_thresholder.description = 'Threshold Nodes by 2D Velocity'
        elif event.new == '2D velocity':
            key = 'velocity_2d_mm'
            self.speed_thresholder.description = 'Threshold Nodes by 2D Velocity'
        elif event.new == '3D velocity':
            key = 'velocity_3d_mm'
            self.speed_thresholder.description = 'Threshold Nodes by 3D Velocity'
        elif event.new == 'Height':
            key = 'height_ave_mm'
            self.speed_thresholder.description = 'Threshold Nodes by Height'
        elif event.new == 'Distance to Center':
            key = 'dist_to_center_px'
            self.speed_thresholder.description = 'Threshold Nodes by Distance to Center'
        else:
            key = 'velocity_2d_mm'
            self.speed_thresholder.description = 'Threshold Nodes by 2D Velocity'

        scalar_threshold_stds = int(self.df[key].max() / self.df[key].std()) + 2
        self.speed_thresholder.options = [float('%.3f' % (self.df[key].std() * i)) for i in
                                          range(scalar_threshold_stds)]
        self.speed_thresholder.index = (0, scalar_threshold_stds - 1)