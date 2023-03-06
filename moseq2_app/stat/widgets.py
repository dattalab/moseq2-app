"""
Widgets module containing classes with components for each of the interactive syllable statistics tools.

"""

import numpy as np
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import clear_output
from moseq2_viz.model.util import normalize_usages

class SyllableStatWidgets:

    def __init__(self):
        """
        initialize interactive syllable satistics widget.
        """
        style = {'description_width': 'initial', 'display': 'flex-grow', 'align_items': 'stretch'}

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.layout_hidden = widgets.Layout(display='none')
        self.layout_visible = widgets.Layout(display='block')

        self.stat_dropdown = widgets.Dropdown(options=['usage', 'duration', '2D Velocity', '3D Velocity', 'Height', 'Distance to Center'], description='Stat to Plot:', disabled=False)

        self.sorting_dropdown = widgets.Dropdown(options=['usage', 'duration', '2D Velocity', '3D Velocity', 'Height', 'Distance to Center', 'Difference'], description='Sorting:', disabled=False)
        self.thresholding_dropdown = widgets.Dropdown(
            options=['usage', 'duration', '2D Velocity', '3D Velocity', 'Height', 'Distance to Center'],
            description='Threshold By:', disabled=False, style=style)

        self.ctrl_dropdown = widgets.Dropdown(options=[], description='Group 1:', disabled=False)
        self.exp_dropdown = widgets.Dropdown(options=[], description='Group 2:', disabled=False)

        self.grouping_dropdown = widgets.Dropdown(options=['group', 'SessionName', 'SubjectName'], description='Grouping:', disabled=False)
        self.session_sel = widgets.SelectMultiple(options=[], description='Sessions:', rows=10,
                                                  layout=self.layout_hidden, disabled=False)

        self.errorbar_dropdown = widgets.Dropdown(options=['CI 95%', 'None', 'SEM', 'STD'], description='Error Bars:', disabled=False)

        self.hyp_test_dropdown = widgets.Dropdown(options=['KW & Dunn\'s', 'Z-Test', 'T-Test', 'Mann-Whitney'], description='Hypothesis Test:',
                                                  style=style, disabled=False)

        ## boxes
        self.data_layout = widgets.Layout(flex_flow='row', padding='top', justify_content='space-around', width='100%')

        self.stat_box = VBox([self.stat_dropdown])
        self.error_box = VBox([self.errorbar_dropdown])
        self.mutation_box = VBox([self.ctrl_dropdown, self.exp_dropdown, self.hyp_test_dropdown])

        self.sorting_box = VBox([self.sorting_dropdown, self.mutation_box, self.thresholding_dropdown])
        self.session_box = VBox([self.grouping_dropdown, self.session_sel])

        self.stat_widget_box = VBox([HBox([VBox([self.stat_box, self.error_box]), self.sorting_box, self.session_box])])
    
    def clear_on_click(self, b=None):
        """
        Clear the cell output

        Args:
        b (button click)
        """

        clear_output()
        del self

    def on_grouping_update(self, event):
        """
        Update the MultipleSelect widget upon selecting groupby == SubjectName or SessionName. Hides it if groupby == group.

        Args:
        event (user clicks new grouping)
        """

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
class SyllableStatBokehCallbacks:

    def __init__(self, condition=''):
        """
        Initialize JS string elements containing different code chunks. 
        
        Args:
        condition (str): string code block that contains a condition to dynamically filter/edit a Bokeh figure using
         bokeh widgets.
        """

        # filter out syllables that do not pass the given condition
        self.js_condition = condition
        # initializing all of the Bokeh ColumnDataSource variables that will be dynamically updated during user interaction.
        self.js_variables = """
                        // All of these arrays will always end up to be the same length
                    
                        // initialize all the variables that appear in the HoverTool
                        // these same variables represent all the attributes that are held by Bokeh Gylph objects.
                        var index = [], number = [], sem = [];
                        var x = [], y = [], usage = [], duration = [], speed_2d = []; 
                        var speed_3d = [], height = [], dist = []; 
                        var label = [], desc = [], movies = [];
                    
                        // initialize the same variables for the plotted error bars.
                        // this is important in order to filter out both the plotted points AND their error bars.
                        var err_x = [], err_y = [];
                        var err_number = [], err_usage = [], err_duration = []; 
                        var err_speed_2d = [], err_speed_3d = [], err_sem = [];
                        var err_height = [], err_dist = [], err_label = [];
                        var err_desc = [], err_movies = [];\n
                       """

        # iterating through the total number of syllables in the set.
        # iterate through all of the indices on the x-axis of the syllable statistics plot
        self.js_for_loop = """for (var i = 0; i < data['x'].length; i++) {\n"""
        
        # runs upon passing the js_condition. Upon passing,
        self.js_condition_pass = """
                            // append accepted syllables to all the js_variables
                            index.push(i);
                            x.push(data['x'][i]);
                            y.push(data['y'][i]);
                            sem.push(data['sem'][i]);
                            number.push(data['number'][i]);
                            usage.push(data['usage'][i]);
                            duration.push(data['duration'][i]);
                            speed_2d.push(data['speed_2d'][i]);
                            speed_3d.push(data['speed_3d'][i]);
                            height.push(data['height'][i]);
                            dist.push(data['dist_to_center'][i]);
                            label.push(data['label'][i]);
                            desc.push(data['desc'][i]);
                            movies.push(data['movies'][i]);
    
                            err_x.push(err_data['x'][i]);
                            err_y.push(err_data['y'][i]);
                            err_number.push(err_data['number'][i]);
    
                            err_sem.push(err_data['sem'][i]);
                            err_usage.push(err_data['usage'][i]);
                            err_duration.push(err_data['duration'][i]);
                            err_speed_2d.push(err_data['speed_2d'][i]);
                            err_speed_3d.push(err_data['speed_3d'][i]);
                            err_height.push(err_data['height'][i]);
                            err_dist.push(err_data['dist_to_center'][i]);
                            err_label.push(err_data['label'][i]);
                            err_desc.push(err_data['desc'][i]);
                            err_movies.push(err_data['movies'][i]);\n
                            """

        self.js_condition_fail = """
                                } else {
                                    // hide the joining line-plot in the syllable statistics if the included
                                    // syllable set is not contiguous.
                                    line.visible = false;
                                }\n
                            }
                            """
        
        # The final step in the CustomJS Callback; there is an inputted variable "source" that has an
        #   attribute named "data" (source.data), which contains attribute references
        #   to all of the variables in js_variables. If source.data.xyz is updated in this section with a corresponding
        #   variable "xyz", and the change is "emitted" using "source.change.emit();", only then will the Bokeh figure be
        #   updated. The "source" variable is what controls the dynamic setting of variables in Bokeh.
        self.js_update = """
                    // reload the line if all the data points are present
                    if (x.length == data['x'].length) {
                        line.visible = true;
                    }
                    
                    // update the data source controlling the interactive figure and emit the changes.
                    source.data.index = index;
                    source.data.number = number;
                    source.data.x = x;
                    source.data.y = y;
                    source.data.sem = sem;
                    source.data.usage = usage;
                    source.data.duration = duration;
                    source.data.speed_2d = speed_2d;
                    source.data.speed_3d = speed_3d;
                    source.data.height = height;
                    source.data.dist_to_center = dist;
                    source.data.label = label;
                    source.data.desc = desc;
                    source.data.movies = movies;
                    
                    // update the plotted points
                    source.change.emit();
    
                    err_source.data.index = index;
                    err_source.data.x = err_x;
                    err_source.data.y = err_y;
    
                    err_source.data.number = err_number;
                    err_source.data.usage = err_usage;
                    err_source.data.duration = err_duration;
                    err_source.data.sem = err_sem;
                    err_source.data.speed_2d = err_speed_2d;
                    err_source.data.speed_3d = err_speed_3d;
                    err_source.data.height = err_height;
                    err_source.data.dist_to_center = err_dist;
                    err_source.data.label = err_label;
                    err_source.data.desc = err_desc;
                    err_source.data.movies = err_movies;
                    
                    // update the plotted error bars
                    err_source.change.emit();\n
                    """

        self.code = self.js_variables + self.js_for_loop + self.js_condition + \
               self.js_condition_pass + self.js_condition_fail + self.js_update

class TransitionGraphWidgets:

    def __init__(self):
        """initialize the transition graph widget.
        """
        style = {'description_width': 'initial', 'display': 'flex-grow', 'align_items': 'stretch'}

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

        col1_layout = widgets.Layout(flex_flow='column', width='75%', align_items='center')
        col2_layout = widgets.Layout(flex_flow='column', align_contents='center', width='75%', align_items='center')

        ui_layout = widgets.Layout(flex_flow='row', border='solid', align_items='stretch',
                                   width='100%', justify_content='space-around')

        self.graph_layout_dropdown = widgets.Dropdown(options=['circular',  'spring'],
                                                      description='Graph Layout',
                                                      style=style, value='circular', continuous_update=False,
                                                      layout=widgets.Layout(align_items='stretch', width='80%'))

        self.edge_thresholder = widgets.FloatRangeSlider(style=style, step=0.001, min=0, readout_format='.4f',
                                                         description='Threshold Edge Weights',
                                                         layout=widgets.Layout(align_items='stretch', width='90%'),
                                                         continuous_update=False)
        self.usage_thresholder = widgets.FloatRangeSlider(style=style, step=0.001, min=0, readout_format='.3f',
                                                          description='Threshold Nodes by Usage',
                                                          layout=widgets.Layout(align_items='stretch', width='90%'),
                                                          continuous_update=False)

        self.thresholding_box = HBox([
                                      VBox([self.graph_layout_dropdown, self.edge_thresholder, self.usage_thresholder],
                                           layout=col1_layout),
                                      # VBox([self.color_nodes_dropdown, self.speed_thresholder], layout=col2_layout)],
                                      # remove color node selector for now since there is no indication what the colors mean 
                                      #VBox([self.speed_thresholder], layout=col2_layout)
                                      ],
                                           layout=ui_layout)

    def clear_on_click(self, b=None):
        """
        Clear the cell output

        Args:
        b (button click)

        Returns:
        """

        clear_output()

    def set_range_widget_values(self):
        """
        set the threshold range sliders' values according to the standard deviations of the dataset.
        """
        from math import ceil
        # Update threshold range values
        self.edge_thresholder.max = np.max(self.trans_mats)
        self.edge_thresholder.value = (0, np.max(self.trans_mats))
        # find max normalized usage threshold and round up
        usages_max = ceil(max([max(normalize_usages(u).values()) for u in self.usages])*1000)/1000.
        self.usage_thresholder.max = usages_max
        self.usage_thresholder.value = (0, usages_max)