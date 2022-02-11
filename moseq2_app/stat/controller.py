'''

Main interactive model syllable statistics results application functionality.
This module facilitates the interactive functionality for the statistics plotting, and
 transition graph features.

'''

import os
import io
import base64
import warnings
import numpy as np
import pandas as pd
from os.path import exists
from collections import defaultdict
from ipywidgets import interactive_output
from moseq2_viz.info.util import transition_entropy
from moseq2_app.util import merge_labels_with_scalars
from moseq2_viz.util import get_sorted_index, read_yaml
from scipy.cluster.hierarchy import linkage, dendrogram
from moseq2_viz.model.dist import get_behavioral_distance
from moseq2_viz.model.stat import run_kruskal, run_pairwise_stats
from moseq2_viz.model.util import (parse_model_results, relabel_by_usage, normalize_usages,
                                   sort_syllables_by_stat, sort_syllables_by_stat_difference)
from moseq2_app.stat.widgets import SyllableStatWidgets, TransitionGraphWidgets
from moseq2_app.stat.view import graph_dendrogram, bokeh_plotting, plot_interactive_transition_graph
from moseq2_viz.model.trans_graph import (get_trans_graph_groups, get_group_trans_mats,
                                         convert_transition_matrix_to_ebunch,
                                         convert_ebunch_to_graph, make_transition_graphs, get_pos)

class InteractiveSyllableStats(SyllableStatWidgets):
    '''

    Interactive Syllable Statistics grapher class that holds the context for the current
     inputted session.

    '''

    def __init__(self, index_path, model_path, df_path, info_path, max_sylls, load_parquet):
        '''
        Initialize the main data inputted into the current context

        Parameters
        ----------
        index_path (str): Path to index file.
        model_path (str): Path to trained model file.
        info_path (str): Path to syllable information file.
        max_sylls (int): Maximum number of syllables to plot.
        load_parquet (bool): Indicates to load previously loaded data
        '''

        super().__init__()

        self.model_path = model_path
        self.info_path = info_path
        self.max_sylls = max_sylls
        self.index_path = index_path
        self.df_path = df_path

        # If user inputs load_parquet=True in main.py function label_syllables()
        # then the self.df_path will be set to the inputted df_path (pointing to a pre-existing parquet file)
        # to load the data from.
        if load_parquet:
            if df_path is not None:
                if not os.path.exists(df_path):
                    self.df_path = None
        else:
            # If load_parquet=False, self.df_path will be set to None to compute the DataFrame from scratch
            self.df_path = None

        self.df = None

        # Load Syllable Info
        self.syll_info = read_yaml(self.info_path)

        self.results = None
        self.icoord, self.dcoord = None, None
        self.cladogram = None

        # Load all the data
        self.interactive_stat_helper()
        self.df = self.df[self.df['syllable'] < self.max_sylls]

        self.session_names = sorted(list(self.df.SessionName.unique()))
        self.subject_names = sorted(list(self.df.SubjectName.unique()))

        # Update the widget values
        self.session_sel.options = self.session_names
        self.session_sel.value = [self.session_sel.options[0]]

        self.ctrl_dropdown.options = list(self.df.group.unique())
        self.exp_dropdown.options = list(self.df.group.unique())
        self.exp_dropdown.value = self.ctrl_dropdown.options[-1]

        self.dropdown_mapping = {
            'usage': 'usage',
            'duration': 'duration',
            'distance to center': 'dist_to_center_px',
            '2d velocity': 'velocity_2d_mm',
            '3d velocity': 'velocity_3d_mm',
            'height': 'height_ave_mm',
            'similarity': 'similarity',
            'difference': 'difference',
            'KW & Dunn\'s': 'kw',
            'Z-Test': 'z_test',
            'T-Test': 't_test',
            'Mann-Whitney': 'mw'
        }

        self.clear_button.on_click(self.clear_on_click)
        self.grouping_dropdown.observe(self.on_grouping_update, names='value')

        # Compute the syllable dendrogram values
        self.compute_dendrogram()

        # Plot the Bokeh graph with the currently selected data.
        self.out = interactive_output(self.interactive_syll_stats_grapher, {
            'stat': self.stat_dropdown,
            'sort': self.sorting_dropdown,
            'groupby': self.grouping_dropdown,
            'errorbar': self.errorbar_dropdown,
            'sessions': self.session_sel,
            'ctrl_group': self.ctrl_dropdown,
            'exp_group': self.exp_dropdown,
            'hyp_test': self.hyp_test_dropdown,
            'thresh': self.thresholding_dropdown
        })

    def compute_dendrogram(self):
        '''
        Computes the pairwise distances between the included model AR-states, and
        generates the graph information to be plotted after the stats.

        Returns
        -------
        '''
        # Get Pairwise distances
        X = get_behavioral_distance(self.sorted_index,
                                    self.model_path,
                                    max_syllable=self.max_sylls,
                                    distances='ar[init]')['ar[init]']

        # Finding the first syllable where the distance between 2 states is np.nan
        # If/when that np.nan syllable-pair distance is found,
        # the nan distances, including the following syllables, will be removed from the matrix X.
        # The reason why the matrix X is cut off upon finding the first np.nan distance is because the
        # syllables are already relabeled by usage, therefore if a syllable is reported to be not used, then
        # the subsequent syllables will also not be used.
        is_missing = np.isnan(X)
        if is_missing.any():
            print('Existing model does not have equal amount of requested states.')
            max_states = int(max(np.where(is_missing.any(1), is_missing.argmax(1), np.nan)))
            print(f'Visualizing max number of available states: {max_states}')
            X = X[:max_states, :max_states]

        Z = linkage(X, 'complete')

        # Get Dendrogram Metadata
        self.results = dendrogram(Z, distance_sort=False, no_plot=True, get_leaves=True)

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
        # Read syllable information dict
        syll_info = read_yaml(self.info_path)

        # Getting number of syllables included in the info dict
        max_sylls = len(self.syll_info)
        for k in range(max_sylls):
            # remove group_info
            syll_info[k].pop('group_info', None)

            # Open videos in encoded urls
            # Implementation from: https://github.com/jupyter/notebook/issues/1024#issuecomment-338664139
            if exists(syll_info[k]['crowd_movie_path']):
                video = io.open(syll_info[k]['crowd_movie_path'], 'r+b').read()
                encoded = base64.b64encode(video)
                syll_info[k]['crowd_movie_path'] = encoded.decode('ascii')

        info_df = pd.DataFrame(syll_info).T.sort_index()
        info_df['syllable'] = info_df.index

        # Load the model and sort labels - also remaps the ar matrices
        model_data = parse_model_results(self.model_path, sort_labels_by_usage=True, count='usage')

        # Read index file
        self.sorted_index = get_sorted_index(self.index_path)

        if not set(model_data['metadata']['uuids']).issubset(set(self.sorted_index['files'])):
            print('Error: Index file UUIDs do not match model UUIDs.')

        # Get max syllables if None is given
        if self.max_sylls is None:
            self.max_sylls = max_sylls

        # if load_parquet=True, and self.df_path points to an existing parquet file,
        # then the syllable statistics DataFrame will be loaded from the parquet file.
        # otherwise, the DataFrame is computed from scratch
        if self.df_path is not None:
            print('Loading parquet files')
            df = pd.read_parquet(self.df_path, engine='fastparquet')
            if len(df.syllable.unique()) < self.max_sylls:
                print('Requested more syllables than the parquet file holds, recomputing requested dataset.')
                df, _ = merge_labels_with_scalars(self.sorted_index, self.model_path)
        else:
            print('Syllable DataFrame not found. Computing syllable statistics...')
            df, _ = merge_labels_with_scalars(self.sorted_index, self.model_path)

        self.df = df.merge(info_df, on='syllable')
        self.df['SubjectName'] = self.df['SubjectName'].astype(str)
        self.df['SessionName'] = self.df['SessionName'].astype(str)
        # When syllable usage is 0, the scalars will have nan affecting the interactive stat plot
        # Since syllable usage is 0, the nan in scalars will be filled with 0
        self.df.fillna(0, inplace=True)

    def run_selected_hypothesis_test(self, hyp_test_name, stat, ctrl_group, exp_group):
        '''
        Helper function that computes the significant syllables for a given pair of groups given
         a name for the type of hypothesis test to run, and the statistic to run the test on.

        Parameters
        ----------
        hyp_test_name (str): Name of hypothesis test to run. options=['KW & Dunn\s', 'Z-Test', 'T-Test', 'Mann-Whitney']
        stat (str): Syllable statistic to compute hypothesis test on. options=['usage', 'distance to center', 'similarity', 'difference']
        ctrl_group (str): Name of control group to compute group difference sorting with.
        exp_group (str): Name of comparative group to compute group difference sorting with.

        Returns
        -------
        sig_sylls (list): list of significant syllables to mark on plotted statistics figure
        '''

        if self.dropdown_mapping[hyp_test_name] == 'kw':
            # run KW and Dunn's Test
            sig_syll_dict = run_kruskal(self.df, statistic=stat, max_syllable=self.max_sylls)[2]

            # get corresponding computed significant syllables from pair dict
            if (ctrl_group, exp_group) in sig_syll_dict:
                sig_sylls = sig_syll_dict[(ctrl_group, exp_group)]
            elif (exp_group, ctrl_group) in sig_syll_dict:
                sig_sylls = sig_syll_dict[(exp_group, ctrl_group)]
            else:
                raise KeyError('Selected groups are not compatible with KW hypothesis test.')

        elif self.dropdown_mapping[hyp_test_name] == 'z_test':
            # run z-test
            intersect_sig_sylls = run_pairwise_stats(self.df, ctrl_group, exp_group,
                                                     test_type='z_test', statistic=stat, max_syllable=self.max_sylls)
        elif self.dropdown_mapping[hyp_test_name] == 't_test':
            # run t-test
            intersect_sig_sylls = run_pairwise_stats(self.df, ctrl_group, exp_group,
                                                     test_type='t_test', statistic=stat, max_syllable=self.max_sylls)
        elif self.dropdown_mapping[hyp_test_name] == 'mw':
            # run Mann-Whitney test
            intersect_sig_sylls = run_pairwise_stats(self.df, ctrl_group, exp_group,
                                                     test_type='mw', statistic=stat, max_syllable=self.max_sylls)

        if self.dropdown_mapping[hyp_test_name] != 'kw':
            sig_sylls = list(intersect_sig_sylls[intersect_sig_sylls["is_sig"] == True].index)

        return sig_sylls


    def interactive_syll_stats_grapher(self, stat, sort, groupby, errorbar, sessions, ctrl_group, exp_group, hyp_test='KW & Dunn\'s', thresh='usage'):
        '''
        Helper function that is responsible for handling ipywidgets interactions and updating the currently
         displayed Bokeh plot.

        Parameters
        ----------
        stat (str or ipywidgets.DropDown): Statistic to plot: ['usage', 'distance to center']
        sort (str or ipywidgets.DropDown): Statistic to sort syllables by (in descending order).
            ['usage', 'distance to center', 'similarity', 'difference'].
        groupby (str or ipywidgets.DropDown): Data to plot; either group averages, or individual session data.
        errorbar (str or ipywidgets.DropDown): Error bar to display. ['None', 'CI 95%' ,'SEM', 'STD']
        sessions (list or ipywidgets.MultiSelect): List of selected sessions to display data from.
        ctrl_group (str or ipywidgets.DropDown): Name of control group to compute group difference sorting with.
        exp_group (str or ipywidgets.DropDown): Name of comparative group to compute group difference sorting with.
        hyp_test (str or ipywidgets.DropDown): Name of hypothesis testing method to use to compute significant syllables.
        thresh (str or ipywidgets.DropDown): Name of statistic to threshold the graph by using the Bokeh Slider
        Returns
        -------
        '''

        # initialize sig_sylls list variable to prevent UnboundLocalError
        sig_sylls = []

        # Get current dataFrame to plot
        df = self.df.fillna(0)

        # Handle names to query DataFrame with
        stat = self.dropdown_mapping[stat.lower()]
        sortby = self.dropdown_mapping[sort.lower()]
        thresh = self.dropdown_mapping[thresh.lower()]

        # Get selected syllable sorting
        if sort.lower() == 'difference':
            # display Text for groups to input experimental groups
            ordering = sort_syllables_by_stat_difference(df, ctrl_group, exp_group, stat=stat)

            # run selected hypothesis test
            if ctrl_group != exp_group:
                sig_sylls_indices = self.run_selected_hypothesis_test(hyp_test, stat, ctrl_group, exp_group)

                # renumber the significant syllables s.t. they are plotted to match the current ordering.
                sig_sylls = np.argsort(ordering)[sig_sylls_indices]

        elif sort.lower() == 'similarity':
            ordering = self.results['leaves']
        elif sort.lower() != 'usage':
            ordering, _ = sort_syllables_by_stat(df, stat=sortby)
        else:
            ordering = range(len(df.syllable.unique()))

        # Handle selective display for whether mutation sort is selected
        if sort.lower() == 'difference':
            self.mutation_box.layout.display = "block"
            sort = f'Difference: {self.exp_dropdown.value} - {self.ctrl_dropdown.value}'
        else:
            self.mutation_box.layout.display = "none"

        # Handle selective display to select included sessions to graph
        if groupby == 'SessionName' or groupby == 'SubjectName':
            mean_df = df.copy()
            df = df[df[groupby].isin(self.session_sel.value)]
            # set error bar to None because error bars are not implemented corectly in SessionName and SubjectName
            errorbar = "None"
            self.error_box.layout.display = "none"
        else:
            self.error_box.layout.display = "block"
            mean_df = None

        # Compute cladogram if it does not already exist
        if self.cladogram is None:
            self.cladogram = graph_dendrogram(self, self.syll_info)
            self.results['cladogram'] = self.cladogram

        self.stat_fig = bokeh_plotting(df, stat, ordering, mean_df=mean_df, groupby=groupby, errorbar=errorbar,
                                       syllable_families=self.results, sort_name=sort, thresh=thresh, sig_sylls=sig_sylls)


class InteractiveTransitionGraph(TransitionGraphWidgets):
    '''

    Interactive transition graph class used to facilitate interactive graph generation
    and thresholding functionality.

    '''

    def __init__(self, model_path, index_path, info_path, df_path, max_sylls, plot_vertically, load_parquet):
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
        self.plot_vertically = plot_vertically

        # If user inputs load_parquet=True in main.py function label_syllables()
        # then the self.df_path will be set to the inputted df_path (pointing to a pre-existing parquet file)
        # to load the data from.
        if load_parquet:
            if df_path is not None:
                if not os.path.exists(df_path):
                    self.df_path = None
        else:
            # If load_parquet=False, self.df_path will be set to None to compute the DataFrame from scratch
            self.df_path = None

        # Load Model
        self.model_fit = parse_model_results(model_path)

        # Load Index File
        self.sorted_index = get_sorted_index(index_path)

        if set(self.sorted_index['files']) != set(self.model_fit['metadata']['uuids']):
            print('Warning: Index file UUIDs do not match model UUIDs.')

        # Load and store transition graph data
        self.initialize_transition_data()

        self.set_range_widget_values()

        self.clear_button.on_click(self.clear_on_click)

        # Manage dropdown menu values
        self.scalar_dict = {
            'Default': 'speeds_2d',
            'Duration': 'duration',
            '2D velocity': 'speeds_2d',
            '3D velocity': 'speeds_3d',
            'Height': 'heights',
            'Distance to Center': 'dists'
        }

    def compute_entropies(self, labels, label_group):
        '''
        Compute individual syllable entropy and transition entropy rates for all sessions with in a label_group.

        Parameters
        ----------
        labels (2d list): list of session syllable labels over time.
        label_group (list): list of groups computing entropies for.

        Returns
        -------
        '''

        self.incoming_transition_entropy, self.outgoing_transition_entropy = [], []

        for g in self.group:
            use_labels = [lbl for lbl, grp in zip(labels, label_group) if grp == g]

            self.incoming_transition_entropy.append(np.mean(transition_entropy(use_labels,
                                                    tm_smoothing=0,
                                                    truncate_syllable=self.max_sylls,
                                                    transition_type='incoming',
                                                    relabel_by='usage'), axis=0))

            self.outgoing_transition_entropy.append(np.mean(transition_entropy(use_labels,
                                                    tm_smoothing=0,
                                                    truncate_syllable=self.max_sylls,
                                                    transition_type='outgoing',
                                                    relabel_by='usage'), axis=0))

    def compute_entropy_differences(self):
        '''
        Computes cross group entropy/entropy-rate differences
         and casts them to OrderedDict objects

        Returns
        -------
        '''

        # Compute entropy + entropy rate differences
        for i in range(len(self.group)):
            for j in range(i + 1, len(self.group)):
                self.incoming_transition_entropy.append(self.incoming_transition_entropy[j] - self.incoming_transition_entropy[i])
                self.outgoing_transition_entropy.append(self.outgoing_transition_entropy[j] - self.outgoing_transition_entropy[i])

    def initialize_transition_data(self):
        '''
        Performs all necessary pre-processing to compute the transition graph data and
         syllable metadata to display via HoverTool.
        Stores the syll_info dict, groups to explore, maximum number of syllables, and
         the respective transition graphs and syllable scalars associated.

        Returns
        -------
        '''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Load Syllable Info
            self.syll_info = read_yaml(self.info_path)

            # Get labels and optionally relabel them by usage sorting
            labels = self.model_fit['labels']

            # get max_sylls
            if self.max_sylls is None:
                self.max_sylls = len(self.syll_info)

            for k in range(self.max_sylls):
                # Open videos in encoded urls
                # Implementation from: https://github.com/jupyter/notebook/issues/1024#issuecomment-338664139
                if exists(self.syll_info[k]['crowd_movie_path']):
                    video = io.open(self.syll_info[k]['crowd_movie_path'], 'r+b').read()
                    encoded = base64.b64encode(video)
                    self.syll_info[k]['crowd_movie_path'] = encoded.decode('ascii')

            if self.df_path is not None:
                print('Loading parquet files')
                df = pd.read_parquet(self.df_path, engine='fastparquet')
            else:
                print('Syllable DataFrame not found. Creating new dataframe and computing syllable statistics...')
                df, _ = merge_labels_with_scalars(self.sorted_index, self.model_path)
            self.df = df

            # Get groups and matching session uuids
            label_group, _ = get_trans_graph_groups(self.model_fit)
            self.group = list(set(label_group))

            labels = relabel_by_usage(labels, count='usage')[0]

            self.compute_entropies(labels, label_group)

            # Compute usages and transition matrices
            self.trans_mats, self.usages = get_group_trans_mats(labels, label_group, self.group, self.max_sylls)
            self.df = self.df[self.df['syllable'] < self.max_sylls]
            self.df = self.df.groupby(['group', 'syllable'], as_index=False).mean()

            self.compute_entropy_differences()

    def interactive_transition_graph_helper(self, layout, edge_threshold, usage_threshold):
        '''

        Helper function that generates all the transition graphs given the currently selected
        thresholding values, then displays them in a Jupyter notebook or web page.

        Parameters
        ----------
        layout (string)
        scalar_color (string)
        edge_threshold (tuple or ipywidgets.FloatRangeSlider): Transition probability range to include in graphs.
        usage_threshold (tuple or ipywidgets.FloatRangeSlider): Syllable usage range to include in graphs.
        speed_threshold (tuple or ipywidgets.FloatRangeSlider): Syllable speed range to include in graphs.

        Returns
        -------
        '''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Get graph node anchors
            anchor = 0
            # make a list of normalized usages
            usages = [normalize_usages(u) for u in self.usages]
            usages_anchor = usages[anchor]

            # Get anchored group scalars
            scalars = defaultdict(list)
            _scalar_map = {
                'duration': 'duration',
                'speeds_2d': 'velocity_2d_mm',
                'speeds_3d': 'velocity_3d_mm',
                'heights': 'height_ave_mm',
                'dists': 'dist_to_center_px'
            }
            # loop thru each group and append a syllable -> scalar value mapping to collection above
            for g in self.group:
                group_df = self.df.query('group == @g').set_index('syllable')
                for new_scalar, old_scalar in _scalar_map.items():
                    scalars[new_scalar].append(dict(group_df[old_scalar]))

            # key = self.scalar_dict.get(scalar_color, 'speeds_2d')
            # scalar_anchor = scalars[key][anchor]

            usage_kwargs = {
                'usages': usages_anchor,
                'usage_threshold': usage_threshold
            }

            # Create graph with nodes and edges
            ebunch_anchor, orphans = convert_transition_matrix_to_ebunch(
                self.trans_mats[anchor], self.trans_mats[anchor], edge_threshold=edge_threshold,
                keep_orphans=True, max_syllable=self.max_sylls, **usage_kwargs)
            indices = [e[:-1] for e in ebunch_anchor]

            # Get graph anchor
            graph_anchor = convert_ebunch_to_graph(ebunch_anchor)

            pos = get_pos(graph_anchor, layout=layout, nnodes=self.max_sylls)

            # make transition graphs
            group_names = self.group.copy()

            # prepare transition graphs
            usages, group_names, _, _, _, graphs, scalars = make_transition_graphs(self.trans_mats,
                                                                                usages[:len(self.group)],
                                                                                self.group,
                                                                                group_names,
                                                                                pos=pos,
                                                                                indices=indices,
                                                                                orphans=orphans,
                                                                                edge_threshold=edge_threshold,
                                                                                arrows=True,
                                                                                scalars=scalars,
                                                                                usage_kwargs=usage_kwargs,
                                                                                )

            # interactive plot transition graphs
            plot_interactive_transition_graph(graphs, pos, self.group,
                                            group_names, usages, self.syll_info,
                                            self.incoming_transition_entropy, self.outgoing_transition_entropy,
                                            scalars=scalars, plot_vertically=self.plot_vertically)
