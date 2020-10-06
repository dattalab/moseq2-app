import os
import joblib
import warnings
import numpy as np
import pandas as pd
import ruamel.yaml as yaml
from moseq2_viz.util import parse_index
from IPython.display import clear_output
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from moseq2_app.stat.widgets import SyllableStatWidgets
from moseq2_app.stat.view import graph_dendrogram, bokeh_plotting
from moseq2_viz.model.label_util import get_syllable_mutation_ordering, get_sorted_syllable_stat_ordering
from moseq2_viz.model.util import (parse_model_results, relabel_by_usage, results_to_dataframe)
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

        self.clear_button.on_click(self.clear_on_click)

    def clear_on_click(self, b):
        '''
        Clears the cell output

        Parameters
        ----------
        b

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
        stat (list or ipywidgets.DropDown): Statistic to plot: ['usage', 'speed', 'distance to center']
        sort (list or ipywidgets.DropDown): Statistic to sort syllables by (in descending order).
            ['usage', 'speed', 'distance to center', 'similarity', 'difference'].
        groupby (list or ipywidgets.DropDown): Data to plot; either group averages, or individual session data.
        sessions (list or ipywidgets.MultiSelect): List of selected sessions to display data from.
        ctrl_group (str or ipywidgets.DropDown): Name of control group to compute group difference sorting with.
        exp_group (str or ipywidgets.DropDown): Name of comparative group to compute group difference sorting with.

        Returns
        -------
        '''

        # Get current dataFrame to plot
        df = self.df

        # Handle names to query DataFrame with
        if stat.lower() == 'distance to center':
            stat = 'dist_to_center'
        elif stat.lower() == 'centroid speed':
            stat = 'speed'
        elif stat.lower() == '2d velocity':
            stat = 'velocity_2d_mm'
        elif stat.lower() == '3d velocity':
            stat = 'velocity_3d_mm'
        elif stat.lower() == 'height':
            stat = 'height_ave_mm'

        if sort.lower() == 'distance to center':
            sortby = 'dist_to_center'
        elif sort.lower() == 'centroid speed':
            sortby = 'speed'
        elif sort.lower() == '2d velocity':
            sortby = 'velocity_2d_mm'
        elif sort.lower() == '3d velocity':
            sortby = 'velocity_3d_mm'
        elif sort.lower() == 'height':
            sortby = 'height_ave_mm'

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