'''

Widgets module containing classes with components for each of the interactive syllable
 statistics tools: Syllable Stats plot, and Transition Graph plot.

'''

import ipywidgets as widgets
from ipywidgets import HBox, VBox

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

class SyllableStatBokehCallbacks:
    def __init__(self, condition=''):

        self.js_condition = condition

        self.js_variables = '''
                        var index = [], number = [], sem = [];
                        var x = [], y = [], usage = [], speed_2d = []; 
                        var speed_3d = [], height = [], dist = []; 
                        var label = [], desc = [], movies = [];
    
                        var err_x = [], err_y = [];
                        var err_number = [], err_usage = []; 
                        var err_speed_2d = [], err_speed_3d = [], err_sem = [];
                        var err_height = [], err_dist = [], err_label = [];
                        var err_desc = [], err_movies = [];\n
                       '''

        self.js_for_loop = '''for (var i = 0; i < data['x'].length; i++) {\n'''

        self.js_condition_pass = '''
                            index.push(i);
                            x.push(data['x'][i]);
                            y.push(data['y'][i]);
                            sem.push(data['sem'][i]);
                            number.push(data['number'][i]);
                            usage.push(data['usage'][i]);
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
                            err_speed_2d.push(err_data['speed_2d'][i]);
                            err_speed_3d.push(err_data['speed_3d'][i]);
                            err_height.push(err_data['height'][i]);
                            err_dist.push(err_data['dist_to_center'][i]);
                            err_label.push(err_data['label'][i]);
                            err_desc.push(err_data['desc'][i]);
                            err_movies.push(err_data['movies'][i]);\n
                            '''

        self.js_condition_fail = '''
                            } else {
                                line.visible = false;
                            }\n
                            '''

        self.js_update = '''
                    source.data.index = index;
                    source.data.number = number;
                    source.data.x = x;
                    source.data.y = y;
                    source.data.sem = sem;
                    source.data.usage = usage;
                    source.data.speed_2d = speed_2d;
                    source.data.speed_3d = speed_3d;
                    source.data.height = height;
                    source.data.dist_to_center = dist;
                    source.data.label = label;
                    source.data.desc = desc;
                    source.data.movies = movies;
    
                    source.change.emit();
    
                    err_source.data.index = index;
                    err_source.data.x = err_x;
                    err_source.data.y = err_y;
    
                    err_source.data.number = err_number;
                    err_source.data.usage = err_usage;
                    err_source.data.sem = err_sem;
                    err_source.data.speed_2d = err_speed_2d;
                    err_source.data.speed_3d = err_speed_3d;
                    err_source.data.height = err_height;
                    err_source.data.dist_to_center = err_dist;
                    err_source.data.label = err_label;
                    err_source.data.desc = err_desc;
                    err_source.data.movies = err_movies;
    
                    err_source.change.emit();\n
                    '''

        self.code = self.js_variables + self.js_for_loop + self.js_condition + \
               self.js_condition_pass + self.js_condition_fail + '}\n' + self.js_update

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
