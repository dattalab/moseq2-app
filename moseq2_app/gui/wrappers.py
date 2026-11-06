"""
Wrapper functions for all the functionality included in moseq2-app.
"""

import os
import html
import numpy as np
import ipywidgets as widgets
from moseq2_app.labels import (LABELS_FOR, STOP, WIPE, read_syllable_labels, stale, stale_labels_message,
                               stopped_message)
from moseq2_common.yaml import read_yaml
from ipywidgets import interactive_output
from IPython.display import display
from moseq2_app.gui.progress import get_session_paths
from moseq2_viz.model.util import (relabel_by_usage, parse_model_results,
                                   compute_syllable_explained_variance)
from moseq2_app.viz.controller import SyllableLabeler, CrowdMovieComparison
from moseq2_app.stat.controller import InteractiveTransitionGraph
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

def interactive_syllable_labeler_wrapper(model_path, index, crowd_movie_dir, output_file, fig_dir,
                                         crowd_movie_settings, max_syllables=None, n_explained=99):
    """
    launch a syllable crowd movie preview and interactive labeling application.

    Args:
    model_path (str): Path to trained model.
    index (dict): the project's sessions (``moseq2_common.bookkeeping.session_index``).
    crowd_movie_dir (str): Path to crowd movie directory
    output_file (str): Path to syllable label information file
    fig_dir (str): folder the explained-variance figure is saved in.
    crowd_movie_settings (dict): config.yaml's ``crowd_movies`` section, the crowd movies' settings.
    max_syllables (int): Maximum number of syllables to preview and label.
    n_explained (int): Percentage of explained variance that sets the number of syllables when max_syllables is None.
    """

    # Load the model
    model = parse_model_results(model_path)

    # Compute the sorted labels
    model['labels'] = relabel_by_usage(model['labels'], count='usage')[0]

    # Get Maximum number of syllables to include
    if max_syllables is None:
        max_sylls = compute_syllable_explained_variance(model, fig_dir, n_explained=n_explained)
    else:
        n_used = len({s for labels in model['labels'] for s in np.unique(labels) if s >= 0})
        if max_syllables > n_used:
            raise ValueError(f'max_syllables is {max_syllables}, but this model uses {n_used} syllables. Set '
                             f'max_syllables to {n_used} or fewer, or to None to choose the number from '
                             'explained_variance.')
        max_sylls = max_syllables

    def open_labeler():
        # Make initial syllable information dict
        labeler = SyllableLabeler(model_fit=model,
                                  model_path=model_path,
                                  index=index,
                                  crowd_movie_settings=crowd_movie_settings,
                                  max_sylls=max_sylls,
                                  crowd_movie_dir=crowd_movie_dir,
                                  save_path=output_file)

        # Launch and display interactive API
        output = widgets.interactive_output(labeler.interactive_syllable_labeler, {'syllables': labeler.syll_select})
        display(labeler.clear_button, labeler.syll_select, output)

    # labels made for another model, or before this one was fit, are not attached to it
    found = None
    if os.path.exists(output_file):
        found = stale((read_yaml(output_file) or {}).get(LABELS_FOR), model_path, 'is a different model')
    if found is None:
        open_labeler()
    else:
        def wipe():
            os.remove(output_file)
            open_labeler()

        display(stale_labels_prompt(stale_labels_message(output_file, [found]), wipe,
                                    lambda: print(stopped_message(output_file))))
    return max_sylls


def stale_labels_prompt(message, on_wipe, on_stop):
    """
    The message for labels made for something else, with the two choices: wipe the labels and continue, or stop.

    Args:
    message (str): what the labels were made for and what they would label now (``stale_labels_message``).
    on_wipe (callable): run when the person chooses to wipe the labels; its output is shown under the buttons.
    on_stop (callable): run when the person chooses to stop.

    Returns:
    (ipywidgets.VBox): the message, the two buttons and the area their output is shown in.
    """
    wipe_button = widgets.Button(description=WIPE, button_style='danger', layout=widgets.Layout(width='auto'))
    stop_button = widgets.Button(description=STOP, layout=widgets.Layout(width='auto'))
    output = widgets.Output()

    def choose(action):
        def clicked(_):
            wipe_button.disabled = stop_button.disabled = True
            with output:
                action()
        return clicked

    wipe_button.on_click(choose(on_wipe))
    stop_button.on_click(choose(on_stop))
    return widgets.VBox([widgets.HTML(f'<p style="color: red;">{html.escape(message)}</p>'),
                         widgets.HBox([wipe_button, stop_button]), output])

def interactive_crowd_movie_comparison_preview_wrapper(crowd_movie_settings, index, model_path, syll_info_path, output_dir,
                                               get_pdfs=True):
    """
    launch an interactive crowd movie comparison application.

    Args:
    crowd_movie_settings (dict): config.yaml's ``crowd_movies`` section, the crowd movies' settings.
    index (dict): the project's sessions (``moseq2_common.bookkeeping.session_index``).
    model_path (str): path to trained model containing syllable labels.
    syll_info_path (str): path to syllable information file containing syllable labels
    output_dir (str): path to directory to store crowd movies
    get_pdfs (bool): indicates whether to compute and display position heatmaps
    """

    syll_info = read_syllable_labels(syll_info_path)

    cm_compare = CrowdMovieComparison(crowd_movie_settings=crowd_movie_settings, index=index,
                                      model_path=model_path, syll_info=syll_info, output_dir=output_dir,
                                      get_pdfs=get_pdfs)

    out = interactive_output(cm_compare.crowd_movie_preview, {'syllable': cm_compare.cm_syll_select,
                                                              'groupby': cm_compare.cm_sources_dropdown,
                                                              'nexamples': cm_compare.num_examples})
    display(cm_compare.clear_button, out)


def interactive_plot_transition_graph_wrapper(model_path, index, info_path,
                                              max_syllables=None, plot_vertically=False):
    """
    prepare the data for the interactive graphing function.

    Args:
    model_path (str): Path to trained model.
    index (dict): the project's sessions (``moseq2_common.bookkeeping.session_index``).
    info_path (str): Path to user-labeled syllable information file.
    max_syllables (int or None): Limit maximum number of displayed syllables.
    """

    # Initialize Transition Graph data structure
    i_trans_graph = InteractiveTransitionGraph(model_path=model_path, index=index,
                                               info_path=info_path,
                                               max_sylls=max_syllables, plot_vertically=plot_vertically)

    # Make graphs
    out = interactive_output(i_trans_graph.interactive_transition_graph_helper,
                             {'layout': i_trans_graph.graph_layout_dropdown,
                              # 'scalar_color': i_trans_graph.color_nodes_dropdown,
                              'edge_threshold': i_trans_graph.edge_thresholder,
                              'usage_threshold': i_trans_graph.usage_thresholder,
                              })

    # Display widgets and bokeh network plots
    display(i_trans_graph.clear_button, i_trans_graph.thresholding_box, out)