"""
Wrapper functions for all the functionality included in moseq2-app.
"""

import os
import shutil
from bokeh.io import show
import ipywidgets as widgets
from moseq2_viz.util import read_yaml
from ipywidgets import interactive_output
from IPython.display import display, clear_output
from moseq2_app.gui.progress import get_session_paths
from moseq2_viz.model.util import (relabel_by_usage, parse_model_results,
                                   compute_syllable_explained_variance)
from moseq2_app.viz.controller import SyllableLabeler, CrowdMovieComparison
from moseq2_app.stat.controller import InteractiveSyllableStats, InteractiveTransitionGraph
from moseq2_app.roi.validation import (make_session_status_dicts, get_scalar_anomaly_sessions,
                                       get_scalar_df, print_validation_results)

def validate_extractions_wrapper(input_dir):
    """
    test the measured scalar values to determine whether some sessions should be flagged and diagnosed before aggregating the sessions.

    Args:
    input_dir (str): path to parent directory containing all the extracted session folders
    """

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

def interactive_syllable_labeler_wrapper(model_path, config_file, index_file, crowd_movie_dir, output_file, fig_dir,
                                         max_syllables=None, n_explained=99, select_median_duration_instances=False, max_examples=20):
    """
    launch a syllable crowd movie preview and interactive labeling application.

    Args:
    model_path (str): Path to trained model.
    crowd_movie_dir (str): Path to crowd movie directory
    output_file (str): Path to syllable label information file
    max_syllables (int): Maximum number of syllables to preview and label.
    """

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
        max_sylls = compute_syllable_explained_variance(model, fig_dir, n_explained=n_explained)
    else:
        max_sylls = max_syllables

    # Make initial syllable information dict
    labeler = SyllableLabeler(model_fit=model,
                              model_path=model_path,
                              index_file=index_file,
                              config_file=config_file,
                              max_sylls=max_sylls,
                              select_median_duration_instances=select_median_duration_instances,
                              max_examples=max_examples,
                              crowd_movie_dir=crowd_movie_dir,
                              save_path=output_file)

    # Launch and display interactive API
    output = widgets.interactive_output(labeler.interactive_syllable_labeler, {'syllables': labeler.syll_select})
    display(labeler.clear_button, labeler.syll_select, output)
    return max_sylls

    def on_syll_change(change):
        """
        select a different syllable number from the Dropdown menu

        Args:
        change (ipywidget DropDown select event): User changes current value of DropDownMenu
        """

        clear_output(wait=True)
        display(labeler.syll_select, output)

    # Update view when user selects new syllable from DropDownMenu
    output.observe(on_syll_change, names='value')

def interactive_crowd_movie_comparison_preview_wrapper(config_filepath, index_path, model_path, syll_info_path, output_dir,
                                               df_path=None, get_pdfs=True, load_parquet=False):
    """
    launch an interactive crowd movie comparison application.

    Args:
    config_filepath (str): path to config file containing crowd movie generation parameters
    index_path (str): path to index file with paths to all the extracted sessions
    model_path (str): path to trained model containing syllable labels.
    syll_info_path (str): path to syllable information file containing syllable labels
    output_dir (str): path to directory to store crowd movies
    df_path (str): optional path to pre-existing syllable information to plot
    get_pdfs (bool): indicates whether to compute and display position heatmaps
    load_parquet (bool): Indicates to load previously saved syllable data.
    """

    config_data = read_yaml(config_filepath)
    syll_info = read_yaml(syll_info_path)

    cm_compare = CrowdMovieComparison(config_data=config_data, index_path=index_path, df_path=df_path,
                                      model_path=model_path, syll_info=syll_info, output_dir=output_dir,
                                      get_pdfs=get_pdfs, load_parquet=load_parquet)

    out = interactive_output(cm_compare.crowd_movie_preview, {'syllable': cm_compare.cm_syll_select,
                                                              'groupby': cm_compare.cm_sources_dropdown,
                                                              'nexamples': cm_compare.num_examples})
    display(cm_compare.clear_button, out)


def interactive_plot_transition_graph_wrapper(model_path, index_path, info_path, df_path=None, 
                                              max_syllables=None, plot_vertically=False, load_parquet=False):
    """
    prepare the data for the interactive graphing function.

    Args:
    model_path (str): Path to trained model.
    index_path (str): Path to index file containined trained data metadata.
    info_path (str): Path to user-labeled syllable information file.
    df_path (str): Path to pre-saved syllable information.
    max_syllables (int or None): Limit maximum number of displayed syllables.
    load_parquet (bool): Indicates to load previously saved data.
    """

    # Initialize Transition Graph data structure
    i_trans_graph = InteractiveTransitionGraph(model_path=model_path, index_path=index_path,
                                               info_path=info_path, df_path=df_path,
                                               max_sylls=max_syllables, plot_vertically=plot_vertically,
                                               load_parquet=load_parquet)

    # Make graphs
    out = interactive_output(i_trans_graph.interactive_transition_graph_helper,
                             {'layout': i_trans_graph.graph_layout_dropdown,
                              # 'scalar_color': i_trans_graph.color_nodes_dropdown,
                              'edge_threshold': i_trans_graph.edge_thresholder,
                              'usage_threshold': i_trans_graph.usage_thresholder,
                              })

    # Display widgets and bokeh network plots
    display(i_trans_graph.clear_button, i_trans_graph.thresholding_box, out)