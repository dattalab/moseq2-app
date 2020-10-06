'''

Wrapper functions for all the functionality included in moseq2-app. These functions are executed directly from
main.py.

'''

import os
import qgrid
import shutil
import joblib
import warnings
import numpy as np
import pandas as pd
import ruamel.yaml as yaml
import ipywidgets as widgets
from moseq2_app.util import index_to_dataframe
from IPython.display import display, clear_output
from moseq2_app.gui.progress import get_session_paths
from moseq2_app.viz.controller import SyllableLabeler
from moseq2_app.gui.widgets import GroupSettingWidgets
from moseq2_app.roi.controller import InteractiveFindRoi, InteractiveExtractionViewer
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
    status_dicts = get_iqr_anomaly_sessions(scalar_df, status_dicts)

    # Run all validation tests
    anomaly_dict = get_anomaly_dict(scalar_df, status_dicts)

    # Print Results
    print_validation_results(anomaly_dict)

def interactive_group_setting_wrapper(index_filepath):
    '''

    Wrapper function that handles the interactive group display and value updating.

    Parameters
    ----------
    index_filepath (str): Path to index file.

    Returns
    -------
    '''

    index_grid = GroupSettingWidgets()

    index_dict, df = index_to_dataframe(index_filepath)
    qgrid_widget = qgrid.show_grid(df[['SessionName', 'SubjectName', 'group', 'uuid']], column_options=index_grid.col_opts,
                                   column_definitions=index_grid.col_defs, show_toolbar=False)

    def update_table(b):
        '''
        Updates table upon "Set Button" click

        Parameters
        ----------
        b (button click)

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
        Updates the index file with the current table state upon Save button click.

        Parameters
        ----------
        b (button click)

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
        # Clear the display
        clear_output()

    # Add callback functions
    index_grid.clear_button.on_click(clear_clicked)
    index_grid.update_index_button.on_click(update_clicked)
    index_grid.save_button.on_click(update_table)

    # Display output
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