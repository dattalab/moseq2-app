import os
import qgrid
import shutil
import joblib
import warnings
import numpy as np
import pandas as pd
from bokeh.io import show
import ruamel.yaml as yaml
import ipywidgets as widgets
from ipywidgets import interactive_output
from moseq2_app.util import index_to_dataframe
from IPython.display import display, clear_output
from moseq2_app.gui.progress import get_session_paths
from moseq2_app.gui.widgets import GroupSettingWidgets
from moseq2_viz.helpers.wrappers import init_wrapper_function
from moseq2_app.viz.controller import SyllableLabeler, CrowdMovieComparison
from moseq2_app.roi.controller import InteractiveFindRoi, InteractiveExtractionViewer
from moseq2_app.stat.controller import InteractiveSyllableStats, InteractiveTransitionGraph
from moseq2_viz.model.util import get_syllable_usages, relabel_by_usage, parse_model_results
from moseq2_app.roi.validation import (make_session_status_dicts, get_iqr_anomaly_sessions, get_scalar_df,
                                       get_anomaly_dict, print_validation_results)

warnings.filterwarnings('ignore')

def interactive_roi_wrapper(data_path, config_file, session_config=None, compute_bgs=True):
    '''

    Interactive ROI detection wrapper function. Users can use run this wrapper
    to find the required extraction parameters, as well as preview examples of the extraction
    with the found parameters.

    Parameters
    ----------
    data_path (str): Path to base directory containing session folders.
    config_data (dict): ROI and Extraction configuration parameters
    session_parameters (str): Path to file containing individual session parameter sets.

    Returns
    -------
    '''

    roi_app = InteractiveFindRoi(data_path, config_file, session_config, compute_bgs=compute_bgs)

    # Run interactive application
    selout = widgets.interactive_output(roi_app.interactive_find_roi_session_selector,
                                        {'session': roi_app.checked_list})
    display(roi_app.clear_button, roi_app.ui_tools, selout)

def interactive_extraction_preview_wrapper(input_dir):
    '''

    Parameters
    ----------
    input_dir

    Returns
    -------

    '''

    viewer = InteractiveExtractionViewer(data_path=input_dir)

    # Run interactive application
    selout = widgets.interactive_output(viewer.get_extraction,
                                        {'input_file': viewer.sess_select})
    display(viewer.clear_button, viewer.sess_select, selout)

def validate_extractions_wrapper(input_dir):
    '''
    Wrapper function to test the measured scalar values to determine whether some sessions should be
     flagged and diagnosed before aggregating the sessions.

    Parameters
    ----------
    input_dir (str): path to parent directory containing all the extracted session folders
    Returns
    -------
    '''

    # Get paths to extracted sessions
    paths = get_session_paths(input_dir, extracted=True)

    # Make status dictionaries containing all the validation flags, also check for dropped frames
    status_dicts = make_session_status_dicts(paths)

    # Get scalar dataframe including all sessions
    scalar_df = get_scalar_df(paths)

    # Flag sessions with mean scalar values that are outside the inter-quartile range (.25-.75)
    status_dicts = get_iqr_anomaly_sessions(scalar_df, status_dicts)

    # Run all validation tests
    anomaly_dict = get_anomaly_dict(scalar_df, status_dicts)

    # Print Results
    print_validation_results(anomaly_dict)

def interactive_group_setting_wrapper(index_filepath):
    '''

    Parameters
    ----------
    index_filepath

    Returns
    -------

    '''

    index_grid = GroupSettingWidgets()

    index_dict, df = index_to_dataframe(index_filepath)
    qgrid_widget = qgrid.show_grid(df[['SessionName', 'SubjectName', 'group', 'uuid']], column_options=index_grid.col_opts,
                                   column_definitions=index_grid.col_defs, show_toolbar=False)

    def update_table(b):
        '''

        Parameters
        ----------
        b

        Returns
        -------

        '''

        index_grid.update_index_button.button_style = 'info'
        index_grid.update_index_button.icon = 'none'

        selected_rows = qgrid_widget.get_selected_df()
        x = selected_rows.index

        for i in x:
            qgrid_widget.edit_cell(i, 'group', index_grid.group_input.value)

    def update_clicked(b):
        '''

        Parameters
        ----------
        b

        Returns
        -------

        '''

        files = index_dict['files']
        meta = [f['metadata'] for f in files]
        meta_cols = pd.DataFrame(meta).columns

        latest_df = qgrid_widget.get_changed_df()
        df.update(latest_df)

        updated_index = {'files': list(df.drop(meta_cols, axis=1).to_dict(orient='index').values()),
                         'pca_path': index_dict['pca_path']}

        with open(index_filepath, 'w+') as f:
            yaml.safe_dump(updated_index, f)

        index_grid.update_index_button.button_style = 'success'
        index_grid.update_index_button.icon = 'check'

    def clear_clicked(b):
        clear_output()

    index_grid.clear_button.on_click(clear_clicked)
    index_grid.update_index_button.on_click(update_clicked)

    index_grid.save_button.on_click(update_table)

    display(index_grid.clear_button, index_grid.group_set)
    display(qgrid_widget)

def interactive_syllable_labeler_wrapper(model_path, config_file, index_file, crowd_movie_dir, output_file,
                                         max_syllables=None, n_explained=99):
    '''
    Wrapper function to launch a syllable crowd movie preview and interactive labeling application.

    Parameters
    ----------
    model_path (str): Path to trained model.
    crowd_movie_dir (str): Path to crowd movie directory
    output_file (str): Path to syllable label information file
    max_syllables (int): Maximum number of syllables to preview and label.

    Returns
    -------
    '''

    # Load the config file
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # Copy index file to modeling session directory
    modeling_session_dir = os.path.dirname(model_path)
    new_index_path = os.path.join(modeling_session_dir, os.path.basename(index_file))
    if os.path.dirname(index_file) != os.path.dirname(new_index_path):
        shutil.copy2(index_file, new_index_path)

    # Load the model
    model = parse_model_results(joblib.load(model_path))

    # Compute the sorted labels
    model['labels'] = relabel_by_usage(model['labels'], count='usage')[0]

    # Get Maximum number of syllables to include
    if max_syllables == None:
        syllable_usages = get_syllable_usages(model, 'usage')
        cumulative_explanation = 100 * np.cumsum(syllable_usages)
        max_sylls = np.argwhere(cumulative_explanation >= n_explained)[0][0]
        print(f'Number of syllables explaining {n_explained}% variance: {max_sylls}')
    else:
        max_sylls = max_syllables

    # Make initial syllable information dict
    labeler = SyllableLabeler(model_fit=model, index_file=index_file, max_sylls=max_sylls, save_path=output_file)

    # Populate syllable info dict with relevant syllable information
    labeler.get_crowd_movie_paths(index_file, model_path, config_data, crowd_movie_dir)

    # Set the syllable dropdown options
    labeler.syll_select.options = labeler.syll_info

    # Launch and display interactive API
    output = widgets.interactive_output(labeler.interactive_syllable_labeler, {'syllables': labeler.syll_select})
    display(labeler.clear_button, labeler.syll_select, output)

    def on_syll_change(change):
        '''
        Callback function for when user selects a different syllable number
        from the Dropdown menu

        Parameters
        ----------
        change (ipywidget DropDown select event): User changes current value of DropDownMenu

        Returns
        -------
        '''

        clear_output()
        display(labeler.syll_select, output)

    # Update view when user selects new syllable from DropDownMenu
    output.observe(on_syll_change, names='value')

def interactive_syllable_stat_wrapper(index_path, model_path, info_path, df_path=None, max_syllables=None):
    '''
    Wrapper function to launch the interactive syllable statistics API. Users will be able to view different
    syllable statistics, sort them according to their metric of choice, and dynamically group the data to
    view individual sessions or group averages.

    Parameters
    ----------
    index_path (str): Path to index file.
    model_path (str): Path to trained model file.
    info_path (str): Path to syllable information file.
    max_syllables (int): Maximum number of syllables to plot.

    Returns
    -------
    '''

    # Initialize the statistical grapher context
    istat = InteractiveSyllableStats(index_path=index_path, model_path=model_path, df_path=df_path,
                                     info_path=info_path, max_sylls=max_syllables)

    # Compute the syllable dendrogram values
    istat.compute_dendrogram()

    # Plot the Bokeh graph with the currently selected data.
    out = interactive_output(istat.interactive_syll_stats_grapher, {
        'stat': istat.stat_dropdown,
        'sort': istat.sorting_dropdown,
        'groupby': istat.grouping_dropdown,
        'errorbar': istat.errorbar_dropdown,
        'sessions': istat.session_sel,
        'ctrl_group': istat.ctrl_dropdown,
        'exp_group': istat.exp_dropdown
    })

    display(istat.clear_button, istat.stat_widget_box, out)
    show(istat.cladogram)

def interactive_crowd_movie_comparison_preview_wrapper(config_filepath, index_path, model_path, syll_info_path, output_dir,
                                               df_path=None, get_pdfs=True):
    '''
    Wrapper function that launches an interactive crowd movie comparison application.
    Uses ipywidgets and Bokeh to facilitate real time user interaction.

    Parameters
    ----------
    config_data (dict): dict containing crowd movie creation parameters
    index_path (str): path to index file with paths to all the extracted sessions
    model_path (str): path to trained model containing syllable labels.
    syll_info_path (str): path to syllable information file containing syllable labels
    output_dir (str): path to directory to store crowd movies

    Returns
    -------
    '''

    with open(config_filepath, 'r') as f:
        config_data = yaml.safe_load(f)

    with open(syll_info_path, 'r') as f:
        syll_info = yaml.safe_load(f)

    index, sorted_index, model_fit = init_wrapper_function(index_file=index_path, model_fit=model_path,
                                                           output_dir=output_dir)

    cm_compare = CrowdMovieComparison(config_data=config_data, index_path=index_path, df_path=df_path,
                                      model_path=model_path, syll_info=syll_info, output_dir=output_dir,
                                      get_pdfs=get_pdfs)

    # Set Syllable select widget options
    cm_compare.cm_syll_select.options = syll_info

    # Set Session MultipleSelect widget options
    sessions = list(set(model_fit['metadata']['uuids']))
    cm_compare.cm_session_sel.options = sorted([sorted_index['files'][s]['metadata']['SessionName'] for s in sessions])

    cm_compare.get_session_mean_syllable_info_df(model_fit, sorted_index)

    out = interactive_output(cm_compare.crowd_movie_preview, {'syllable': cm_compare.cm_syll_select,
                                                              'groupby': cm_compare.cm_sources_dropdown,
                                                              'nexamples': cm_compare.num_examples})
    display(cm_compare.clear_button, out)


def interactive_plot_transition_graph_wrapper(model_path, index_path, info_path, df_path=None, max_syllables=None):
    '''
    Wrapper function that works as a background process that prepares the data
    for the interactive graphing function.

    Parameters
    ----------
    model_path (str): Path to trained model.
    index_path (str): Path to index file containined trained data metadata.
    info_path (str): Path to user-labeled syllable information file.

    Returns
    -------
    '''

    # Initialize Transition Graph data structure
    i_trans_graph = InteractiveTransitionGraph(model_path=model_path, index_path=index_path,
                                               info_path=info_path, df_path=df_path, max_sylls=max_syllables)

    # Make graphs
    out = interactive_output(i_trans_graph.interactive_transition_graph_helper,
                             {'layout': i_trans_graph.graph_layout_dropdown,
                              'scalar_color': i_trans_graph.color_nodes_dropdown,
                              'edge_threshold': i_trans_graph.edge_thresholder,
                              'usage_threshold': i_trans_graph.usage_thresholder,
                              'speed_threshold': i_trans_graph.speed_thresholder,
                              })

    # Display widgets and bokeh network plots
    display(i_trans_graph.clear_button, i_trans_graph.thresholding_box, out)