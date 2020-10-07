'''

Main interactive model syllable statistics results app functionality.

'''

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import ruamel.yaml as yaml
from moseq2_viz.util import parse_index
from IPython.display import clear_output
from moseq2_viz.info.util import entropy, entropy_rate
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from moseq2_app.stat.widgets import SyllableStatWidgets, TransitionGraphWidgets
from moseq2_viz.model.util import (parse_model_results, relabel_by_usage, results_to_dataframe)
from moseq2_app.stat.view import graph_dendrogram, bokeh_plotting, plot_interactive_transition_graph
from moseq2_viz.model.trans_graph import (get_trans_graph_groups, get_group_trans_mats, get_usage_dict,
                                         handle_graph_layout, convert_transition_matrix_to_ebunch,
                                         convert_ebunch_to_graph, make_transition_graphs, get_pos)
from moseq2_viz.model.label_util import get_syllable_mutation_ordering, get_sorted_syllable_stat_ordering
from moseq2_viz.scalars.util import (scalars_to_dataframe, compute_mean_syll_scalar, compute_session_centroid_speeds)

class InteractiveSyllableStats(SyllableStatWidgets):
    '''

    Interactive Syllable Statistics grapher class that holds the context for the current
     inputted session.

    '''

    def __init__(self, index_path, model_path, df_path, info_path, max_sylls):
        '''
        Initialize the main data inputted into the current context

        Parameters
        ----------
        index_path (str): Path to index file.
        model_path (str): Path to trained model file.
        info_path (str): Path to syllable information file.
        max_sylls (int): Maximum number of syllables to plot.
        '''

        super().__init__()

        self.model_path = model_path
        self.info_path = info_path
        self.max_sylls = max_sylls
        self.index_path = index_path
        self.df_path = df_path

        if df_path != None:
            if os.path.exists(df_path):
                self.label_df_path = df_path.replace('syll_df', 'label_time_df')
            else:
                self.df_path = None

        self.df = None

        self.ar_mats = None
        self.results = None
        self.icoord, self.dcoord = None, None
        self.cladogram = None

        # Load all the data
        self.interactive_stat_helper()
        self.df = self.df[self.df['syllable'] < self.max_sylls]

        # Update the widget values
        self.session_sel.options = sorted(list(self.df.SessionName.unique()))
        self.session_sel.value = [self.session_sel.options[0]]

        self.ctrl_dropdown.options = list(self.df.group.unique())
        self.exp_dropdown.options = list(self.df.group.unique())
        self.exp_dropdown.value = self.ctrl_dropdown.options[-1]

        self.dropdown_mapping = {
            'distance to center': 'dist_to_center',
            'centroid_speed': 'speed',
            '2d velocity': 'velocity_2d_mm',
            '3d velocity': 'velocity_3d_mm',
            'height': 'height_ave_mm'
        }

        self.clear_button.on_click(self.clear_on_click)

    def clear_on_click(self, b):
        '''
        Clears the cell output

        Parameters
        ----------
        b (button click)

        Returns
        -------
        '''

        clear_output()

    def compute_dendrogram(self):
        '''
        Computes the pairwise distances between the included model AR-states, and
        generates the graph information to be plotted after the stats.

        Returns
        -------
        '''

        # Get Pairwise distances
        X = pairwise_distances(self.ar_mats, metric='euclidean')
        Z = linkage(X, 'ward')

        # Get Dendrogram Metadata
        self.results = dendrogram(Z, distance_sort=True, no_plot=True, get_leaves=True)

        # Get Graph layout info
        icoord, dcoord = self.results['icoord'], self.results['dcoord']

        icoord = pd.DataFrame(icoord) - 5
        icoord = icoord * (self.df['syllable'].max() / icoord.max().max())
        self.icoord = icoord.values

        dcoord = pd.DataFrame(dcoord)
        dcoord = dcoord * (self.df['usage'].max() / dcoord.max().max())
        self.dcoord = dcoord.values

    def interactive_stat_helper(self):
        '''
        Computes and saves the all the relevant syllable information to be displayed.
         Loads the syllable information dict and merges it with the syllable statistics DataFrame.

        Returns
        -------
        '''

        warnings.filterwarnings('ignore')

        # Read syllable information dict
        with open(self.info_path, 'r') as f:
            syll_info = yaml.safe_load(f)

        # Getting number of syllables included in the info dict
        max_sylls = len(list(syll_info.keys()))
        for k in range(max_sylls):
            if 'group_info' in syll_info[str(k)].keys():
                del syll_info[str(k)]['group_info']

        info_df = pd.DataFrame(list(syll_info.values()), index=[int(k) for k in list(syll_info.keys())]).sort_index()
        info_df['syllable'] = info_df.index

        # Load the model
        model_data = parse_model_results(joblib.load(self.model_path))

        # Relabel the models, and get the order mapping
        labels, mapping = relabel_by_usage(model_data['labels'], count='usage')

        # Get max syllables if None is given
        if self.max_sylls == None:
            self.max_sylls = max_sylls

        # Read AR matrices and reorder according to the syllable mapping
        ar_mats = np.array(model_data['model_parameters']['ar_mat'])
        self.ar_mats = np.reshape(ar_mats, (100, -1))[mapping][:self.max_sylls]

        if self.df_path != None:
            print('Loading parquet files')
            df = pd.read_parquet(self.df_path, engine='fastparquet')
            label_df = pd.read_parquet(self.label_df_path, engine='fastparquet')
            label_df.columns = label_df.columns.astype(int)
        else:
            print('Syllable DataFrame not found. Computing syllable statistics...')
            # Read index file
            index, sorted_index = parse_index(self.index_path)

            # Load scalar Dataframe to compute syllable speeds
            scalar_df = scalars_to_dataframe(sorted_index)

            # Compute a syllable summary Dataframe containing usage-based
            # sorted/relabeled syllable usage and duration information from [0, max_syllable) inclusive
            df, label_df = results_to_dataframe(model_data, index, count='usage',
                                                max_syllable=self.max_sylls, sort=True, compute_labels=True)

            scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)

            # Compute and append additional syllable scalar data
            scalars = ['centroid_speed_mm', 'velocity_2d_mm', 'velocity_3d_mm', 'height_ave_mm', 'dist_to_center_px']
            for scalar in scalars:
                df = compute_mean_syll_scalar(df, scalar_df, label_df, scalar=scalar, max_sylls=self.max_sylls)

        self.df = df.merge(info_df, on='syllable')

    def interactive_syll_stats_grapher(self, stat, sort, groupby, errorbar, sessions, ctrl_group, exp_group):
        '''
        Helper function that is responsible for handling ipywidgets interactions and updating the currently
         displayed Bokeh plot.

        Parameters
        ----------
        stat (str or ipywidgets.DropDown): Statistic to plot: ['usage', 'speed', 'distance to center']
        sort (str or ipywidgets.DropDown): Statistic to sort syllables by (in descending order).
            ['usage', 'speed', 'distance to center', 'similarity', 'difference'].
        groupby (str or ipywidgets.DropDown): Data to plot; either group averages, or individual session data.
        errorbar (str or ipywidgets.DropDown): Error bar to display. ['SEM', 'STD']
        sessions (list or ipywidgets.MultiSelect): List of selected sessions to display data from.
        ctrl_group (str or ipywidgets.DropDown): Name of control group to compute group difference sorting with.
        exp_group (str or ipywidgets.DropDown): Name of comparative group to compute group difference sorting with.

        Returns
        -------
        '''

        # Get current dataFrame to plot
        df = self.df

        # Handle names to query DataFrame with
        stat = self.dropdown_mapping[stat.lower()]
        sortby = self.dropdown_mapping[sort.lower()]

        # Get selected syllable sorting
        if sort.lower() == 'difference':
            # display Text for groups to input experimental groups
            ordering = get_syllable_mutation_ordering(df, ctrl_group, exp_group, stat=stat)
        elif sort.lower() == 'similarity':
            ordering = self.results['leaves']
        elif sort.lower() != 'usage':
            ordering, _ = get_sorted_syllable_stat_ordering(df, stat=sortby)
        else:
            ordering = range(len(df.syllable.unique()))

        # Handle selective display for whether mutation sort is selected
        if sort.lower() == 'difference':
            self.mutation_box.layout.display = "block"
        else:
            self.mutation_box.layout.display = "none"

        # Handle selective display to select included sessions to graph
        if groupby == 'SessionName':
            self.session_sel.layout.display = "flex"
            self.session_sel.layout.align_items = 'stretch'

            mean_df = df.copy()
            df = df[df['SessionName'].isin(self.session_sel.value)]

        else:
            self.session_sel.layout.display = "none"
            mean_df = None

        # Compute cladogram if it does not already exist
        if self.cladogram == None:
            self.cladogram = graph_dendrogram(self)
            self.results['cladogram'] = self.cladogram

        self.stat_fig = bokeh_plotting(df, stat, ordering, mean_df=mean_df, groupby=groupby,
                                       errorbar=errorbar, syllable_families=self.results, sort_name=sort)


class InteractiveTransitionGraph(TransitionGraphWidgets):
    '''

    Interactive transition graph class used to facilitate interactive graph generation
    and thresholding functionality.

    '''

    def __init__(self, model_path, index_path, info_path, df_path, max_sylls):
        '''
        Initializes context variables

        Parameters
        ----------
        model_path (str): Path to trained model file
        index_path (str): Path to index file containing trained session metadata.
        info_path (str): Path to labeled syllable info file
        max_sylls (int): Maximum number of syllables to plot.
        '''

        super().__init__()

        self.model_path = model_path
        self.index_path = index_path
        self.info_path = info_path
        self.df_path = df_path
        self.max_sylls = max_sylls

        if df_path != None:
            if os.path.exists(df_path):
                self.label_df_path = df_path.replace('syll_df', 'label_time_df')
            else:
                self.df_path = None

        # Load and store transition graph data
        self.initialize_transition_data()

        self.set_range_widget_values()

        # Set color dropdown callback
        self.color_nodes_dropdown.observe(self.on_set_scalar, names='value')

        self.clear_button.on_click(self.clear_on_click)

    def clear_on_click(self, b):
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
        speed_threshold_stds = int(self.df['speed'].max() / self.df['speed'].std()) + 2

        self.edge_thresholder.options = [float('%.3f' % (np.std(self.trans_mats) * i)) for i in
                                         range(edge_threshold_stds)]
        self.edge_thresholder.index = (1, edge_threshold_stds - 1)

        self.usage_thresholder.options = [float('%.3f' % (self.df['usage'].std() * i)) for i in
                                          range(usage_threshold_stds)]
        self.usage_thresholder.index = (0, usage_threshold_stds - 1)

        self.speed_thresholder.options = [float('%.3f' % (self.df['speed'].std() * i)) for i in
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

        if event.new == 'Default' or event.new == 'Centroid Speed':
            key = 'speed'
            self.speed_thresholder.description = 'Threshold Nodes by Speed'
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
            key = 'dist_to_center'
            self.speed_thresholder.description = 'Threshold Nodes by Distance to Center'
        else:
            key = 'speed'
            self.speed_thresholder.description = 'Threshold Nodes by Speed'

        scalar_threshold_stds = int(self.df[key].max() / self.df[key].std()) + 2
        self.speed_thresholder.options = [float('%.3f' % (self.df[key].std() * i)) for i in
                                          range(scalar_threshold_stds)]
        self.speed_thresholder.index = (0, scalar_threshold_stds - 1)


    def initialize_transition_data(self):
        '''
        Performs all necessary pre-processing to compute the transition graph data and
         syllable metadata to display via HoverTool.
        Stores the syll_info dict, groups to explore, maximum number of syllables, and
         the respective transition graphs and syllable scalars associated.

        Returns
        -------
        '''

        warnings.filterwarnings('ignore')

        # Load Model
        model_fit = parse_model_results(joblib.load(self.model_path))

        # Load Index File
        index, sorted_index = parse_index(self.index_path)

        # Load scalar Dataframe to compute syllable speeds
        scalar_df = scalars_to_dataframe(sorted_index)

        # Load Syllable Info
        with open(self.info_path, 'r') as f:
            self.syll_info = yaml.safe_load(f)

        # Get labels and optionally relabel them by usage sorting
        labels = model_fit['labels']

        # get max_sylls
        if self.max_sylls == None:
            self.max_sylls = len(list(self.syll_info.keys()))

        if self.df_path != None:
            print('Loading parquet files')
            df = pd.read_parquet(self.df_path, engine='fastparquet')
            label_df = pd.read_parquet(self.label_df_path, engine='fastparquet')
            label_df.columns = label_df.columns.astype(int)
        else:
            print('Syllable DataFrame not found. Computing syllable statistics...')
            # Compute a syllable summary Dataframe containing usage-based
            # sorted/relabeled syllable usage and duration information from [0, max_syllable) inclusive
            df, label_df = results_to_dataframe(model_fit, index, count='usage',
                                                max_syllable=self.max_sylls, sort=True, compute_labels=True)
            scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)

            # Compute and append additional syllable scalar data
            scalars = ['centroid_speed_mm', 'velocity_2d_mm', 'velocity_3d_mm', 'height_ave_mm', 'dist_to_center_px']
            for scalar in scalars:
                df = compute_mean_syll_scalar(df, scalar_df, label_df, scalar=scalar, max_sylls=self.max_sylls)

        # Get groups and matching session uuids
        self.group, label_group, label_uuids = get_trans_graph_groups(model_fit, index, sorted_index)

        # Compute entropies
        entropies = []
        for g in self.group:
            use_labels = [lbl for lbl, grp in zip(labels, label_group) if grp == g]
            entropies.append(
                np.mean(entropy(use_labels, truncate_syllable=self.max_sylls, get_session_sum=False), axis=0))

        self.entropies = entropies

        # Compute entropy rates
        entropy_rates = []
        for g in self.group:
            use_labels = [lbl for lbl, grp in zip(labels, label_group) if grp == g]
            entropy_rates.append(
                np.mean(entropy_rate(use_labels, truncate_syllable=self.max_sylls, get_session_sum=False), axis=0))

        self.entropy_rates = entropy_rates

        labels = relabel_by_usage(labels, count='usage')[0]

        # Compute usages and transition matrices
        self.trans_mats, usages = get_group_trans_mats(labels, label_group, self.group, self.max_sylls)
        self.df = df.groupby(['group', 'syllable'], as_index=False).mean()
        self.df = self.df[self.df['syllable'] < self.max_sylls]

        # Get usage dictionary for node sizes
        self.usages = get_usage_dict(usages)

        # Compute entropy + entropy rate differences
        for i in range(len(self.group)):
            for j in range(i + 1, len(self.group)):
                self.entropies.append(self.entropies[j] - self.entropies[i])
                self.entropy_rates.append(self.entropy_rates[j] - self.entropy_rates[i])

        # Set entropy and entropy rate Ordered Dicts
        for i in range(len(self.entropies)):
            self.entropies[i] = get_usage_dict([self.entropies[i]])[0]
            self.entropy_rates[i] = get_usage_dict([self.entropy_rates[i]])[0]

    def interactive_transition_graph_helper(self, layout, scalar_color, edge_threshold, usage_threshold, speed_threshold):
        '''

        Helper function that generates all the transition graphs given the currently selected
        thresholding values, then displays them in a Jupyter notebook or web page.

        Parameters
        ----------
        edge_threshold (tuple or ipywidgets.FloatRangeSlider): Transition probability range to include in graphs.
        usage_threshold (tuple or ipywidgets.FloatRangeSlider): Syllable usage range to include in graphs.
        speed_threshold (tuple or ipywidgets.FloatRangeSlider): Syllable speed range to include in graphs.

        Returns
        -------
        '''

        # Get graph node anchors
        usages, anchor, usages_anchor, ngraphs = handle_graph_layout(self.trans_mats, self.usages, anchor=0)

        weights = self.trans_mats

        # Get anchored group scalars
        scalars = {
            'speed': [],
            'speeds_2d': [],
            'speeds_3d': [],
            'heights': [],
            'dists': []
        }

        for g in self.group:
            scalars['speed'].append(self.df[self.df['group'] == g]['speed'].to_numpy())
            scalars['speeds_2d'].append(self.df[self.df['group'] == g]['velocity_2d_mm'].to_numpy())
            scalars['speeds_3d'].append(self.df[self.df['group'] == g]['velocity_3d_mm'].to_numpy())
            scalars['heights'].append(self.df[self.df['group'] == g]['height_ave_mm'].to_numpy())
            scalars['dists'].append(self.df[self.df['group'] == g]['dist_to_center'].to_numpy())

        if scalar_color == 'Default' or scalar_color == 'Centroid Speed':
            key = 'speed'
        elif scalar_color == '2D velocity':
            key = 'speeds_2d'
        elif scalar_color == '3D velocity':
            key = 'speeds_3d'
        elif scalar_color == 'Height':
            key = 'heights'
        elif scalar_color == 'Distance to Center':
            key = 'dists'
        else:
            key = 'speed'

        scalar_anchor = get_usage_dict([scalars[key][anchor]])[0]

        # Create graph with nodes and edges
        ebunch_anchor, orphans = convert_transition_matrix_to_ebunch(
            weights[anchor], self.trans_mats[anchor], edge_threshold=edge_threshold,
            keep_orphans=True, usages=usages_anchor, speeds=scalar_anchor, speed_threshold=speed_threshold,
            usage_threshold=usage_threshold, max_syllable=self.max_sylls - 1)

        # Get graph anchor
        graph_anchor = convert_ebunch_to_graph(ebunch_anchor)

        pos = get_pos(graph_anchor, layout=layout, nnodes=self.max_sylls-1)

        # make transition graphs
        group_names = self.group.copy()

        # prepare transition graphs
        usages, group_names, _, _, _, graphs, scalars = make_transition_graphs(self.trans_mats,
                                                                               self.usages[:len(self.group)],
                                                                               self.group,
                                                                               group_names,
                                                                               usages_anchor,
                                                                               pos, ebunch_anchor,
                                                                               orphans=orphans,
                                                                               edge_threshold=edge_threshold,
                                                                               arrows=True,
                                                                               scalars=scalars)

        for key in scalars.keys():
            for i, scalar in enumerate(scalars[key]):
                if isinstance(scalar, (list, np.ndarray)):
                    scalars[key][i] = get_usage_dict([scalar])[0]

        # interactive plot transition graphs
        plot_interactive_transition_graph(graphs, pos, self.group,
                                          group_names, usages, self.syll_info,
                                          self.entropies, self.entropy_rates,
                                          scalars=scalars, scalar_color=scalar_color)
