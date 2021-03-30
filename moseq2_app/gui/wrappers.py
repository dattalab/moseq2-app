'''

Wrapper functions for all the functionality included in moseq2-app. These functions are executed directly from
main.py.

'''

import os
import shutil
import ipywidgets as widgets
from bokeh.io import show
from moseq2_viz.util import read_yaml
from ipywidgets import interactive_output
from IPython.display import display, clear_output
from moseq2_app.flip.controller import FlipRangeTool
from moseq2_app.gui.progress import get_session_paths
from moseq2_app.gui.widgets import GroupSettingWidgets
from moseq2_app.scalars.controller import InteractiveScalarViewer
from moseq2_viz.model.util import (relabel_by_usage, parse_model_results,
                                   compute_syllable_explained_variance)
from moseq2_app.viz.controller import SyllableLabeler, CrowdMovieComparison
from moseq2_app.roi.controller import InteractiveFindRoi, InteractiveExtractionViewer
from moseq2_app.stat.controller import InteractiveSyllableStats, InteractiveTransitionGraph
from moseq2_app.roi.validation import (make_session_status_dicts, get_scalar_anomaly_sessions,
                                       get_scalar_df, print_validation_results)

def interactive_roi_wrapper(data_path, config_file, session_config=None, compute_bgs=True, autodetect_depths=False):
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

    roi_app = InteractiveFindRoi(data_path,
                                 config_file,
                                 session_config,
                                 compute_bgs=compute_bgs,
                                 autodetect_depths=autodetect_depths)

    # Run interactive application
    roi_app.interactive_find_roi_session_selector(roi_app.checked_list.value)


def interactive_extraction_preview_wrapper(input_dir):
    '''

    Interactive extraction previewing tool. Upon extracted session selection, function automatically displays
    the extraction mp4 video file.

    Parameters
    ----------
    input_dir (str): path to base directory containing extraction directories

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
    status_dicts = get_scalar_anomaly_sessions(scalar_df, status_dicts)

    # Print Results
    print_validation_results(scalar_df, status_dicts)

def interactive_group_setting_wrapper(index_filepath):
    '''

    Wrapper function that handles the interactive group display and value updating.

    Parameters
    ----------
    index_filepath (str): Path to index file.

    Returns
    -------
    '''

    index_grid = GroupSettingWidgets(index_filepath)

    # Add callback functions
    index_grid.clear_button.on_click(index_grid.clear_clicked)
    index_grid.update_index_button.on_click(index_grid.update_clicked)
    index_grid.save_button.on_click(index_grid.update_table)

    # Display output
    display(index_grid.clear_button, index_grid.group_set)
    display(index_grid.qgrid_widget)

    return index_grid

def interactive_scalar_summary_wrapper(index_filepath):
    '''
    Wrapper function to launch the session scalar summary plot.

    Parameters
    ----------
    index_filepath (str): Path to index file to plot scalars from.

    Returns
    -------
    viewer (InteractiveScalarViewer obj): Scalar summary viewer object.
    '''

    viewer = InteractiveScalarViewer(index_filepath)

    return viewer

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

    # Copy index file to modeling session directory
    modeling_session_dir = os.path.dirname(model_path)
    new_index_path = os.path.join(modeling_session_dir, os.path.basename(index_file))
    if os.path.dirname(index_file) != os.path.dirname(new_index_path):
        shutil.copy2(index_file, new_index_path)

    # Load the model
    model = parse_model_results(model_path)

    # Compute the sorted labels
    model['labels'] = relabel_by_usage(model['labels'], count='usage')[0]

    # Get Maximum number of syllables to include
    if max_syllables is None:
        max_sylls = compute_syllable_explained_variance(model, n_explained=n_explained)
    else:
        max_sylls = max_syllables

    # Make initial syllable information dict
    labeler = SyllableLabeler(model_fit=model,
                              model_path=model_path,
                              index_file=index_file,
                              config_file=config_file,
                              max_sylls=max_sylls,
                              crowd_movie_dir=crowd_movie_dir,
                              save_path=output_file)

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

        clear_output(wait=True)
        display(labeler.syll_select, output)

    # Update view when user selects new syllable from DropDownMenu
    output.observe(on_syll_change, names='value')

def interactive_syllable_stat_wrapper(index_path, model_path, info_path, df_path=None, max_syllables=None, load_parquet=False):
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
    load_parquet (bool): Indicates to load previously loaded data

    Returns
    -------
    '''

    # Initialize the statistical grapher context
    istat = InteractiveSyllableStats(index_path=index_path, model_path=model_path, df_path=df_path,
                                     info_path=info_path, max_sylls=max_syllables, load_parquet=load_parquet)

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
                                               df_path=None, get_pdfs=True, load_parquet=False):
    '''
    Wrapper function that launches an interactive crowd movie comparison application.
    Uses ipywidgets and Bokeh to facilitate real time user interaction.

    Parameters
    ----------
    config_filepath (str): path to config file containing crowd movie generation parameters
    index_path (str): path to index file with paths to all the extracted sessions
    model_path (str): path to trained model containing syllable labels.
    syll_info_path (str): path to syllable information file containing syllable labels
    output_dir (str): path to directory to store crowd movies
    df_path (str): optional path to pre-existing syllable information to plot
    get_pdfs (bool): indicates whether to compute and display position heatmaps
    load_parquet (bool): Indicates to load previously saved syllable data.

    Returns
    -------
    '''

    config_data = read_yaml(config_filepath)
    syll_info = read_yaml(syll_info_path)

    cm_compare = CrowdMovieComparison(config_data=config_data, index_path=index_path, df_path=df_path,
                                      model_path=model_path, syll_info=syll_info, output_dir=output_dir,
                                      get_pdfs=get_pdfs, load_parquet=load_parquet)

    out = interactive_output(cm_compare.crowd_movie_preview, {'syllable': cm_compare.cm_syll_select,
                                                              'groupby': cm_compare.cm_sources_dropdown,
                                                              'nexamples': cm_compare.num_examples})
    display(cm_compare.clear_button, out)


def interactive_plot_transition_graph_wrapper(model_path, index_path, info_path, df_path=None, max_syllables=None, plot_vertically=False, load_parquet=False):
    '''
    Wrapper function that works as a background process that prepares the data
    for the interactive graphing function.

    Parameters
    ----------
    model_path (str): Path to trained model.
    index_path (str): Path to index file containined trained data metadata.
    info_path (str): Path to user-labeled syllable information file.
    df_path (str): Path to pre-saved syllable information.
    max_syllables (int or None): Limit maximum number of displayed syllables.
    load_parquet (bool): Indicates to load previously saved data.

    Returns
    -------
    '''

    # Initialize Transition Graph data structure
    i_trans_graph = InteractiveTransitionGraph(model_path=model_path, index_path=index_path,
                                               info_path=info_path, df_path=df_path,
                                               max_sylls=max_syllables, plot_vertically=plot_vertically,
                                               load_parquet=load_parquet)

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

def get_frame_flips_wrapper(input_dir,
                            output_file,
                            max_frames=1e6,
                            tail_filter_iters=1,
                            space_filter_size=3,
                            continuous_slider_update=True):
    '''

    Wrapper function that facilitates the interactive

    Parameters
    ----------
    input_dir (str): Input directory containing extracted session folders.
    output_file (str): Path to save flip classifier in.
    max_frames (int): Maximum number of frames to load from the extracted data.
    tail_filter_iters (int): Number of tail filtering iterations
    prefilter_kernel_size (int): Size of the median spatial filter.

    Returns
    -------
    flip_finder (FlipRangeTool): Flip Classifier Training object that will be used throughout the notebook to
     hold the labeled accepted frame ranges and selected paths/info.
    '''

    flip_finder = FlipRangeTool(input_dir=input_dir,
                                max_frames=max_frames,
                                output_file=output_file,
                                tail_filter_iters=tail_filter_iters,
                                prefilter_kernel_size=space_filter_size,
                                continuous_slider_update=continuous_slider_update)

    return flip_finder
