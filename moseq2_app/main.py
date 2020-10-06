'''

Main functions that facilitate all jupyter notebook functionality. All functions will call a wrapper function
 to handle any non front-end settings.

'''

from bokeh.io import output_notebook
from moseq2_extract.helpers.data import get_selected_sessions
from moseq2_app.gui.wrappers import interactive_roi_wrapper, interactive_extraction_preview_wrapper, \
     validate_extractions_wrapper

def view_extraction(extractions, default=0):
    '''
    Prompts user to select which extracted video(s) to preview.

    Parameters
    ----------
    extractions (list): list of paths to all extracted avi videos.
    default (int): index of the default extraction to display

    Returns
    -------
    extractions (list): list of selected extractions.
    '''

    if len(extractions) == 0:
        print('no sessions to view')
        return []

    if default < 0:
        for i, sess in enumerate(extractions):
            print(f'[{str(i + 1)}] {sess}')
        extractions = get_selected_sessions(extractions, False)
    else:
        print(f"Displaying {extractions[default]}")
        return [extractions[default]]

    return extractions

def interactive_roi_detector(input_dir, progress_paths, compute_all_bgs=True):
    '''
    Function to launch ROI detector interactive GUI in jupyter notebook

    Parameters
    ----------
    input_dir (str): path to parent directory containing session folders
    progress_paths (dict): dictionary of notebook progress paths.
    compute_all_bgs (bool): if True, computes all the sessions' background images to speed up the UI.

    Returns
    -------
    '''

    output_notebook()
    config_file = progress_paths['config_file']
    session_config = progress_paths['session_config']

    interactive_roi_wrapper(input_dir, config_file, session_config, compute_bgs=compute_all_bgs)

def preview_extractions(input_dir):
    '''
    Function to launch a dynamic video loader that displays extraction session mp4s.

    Parameters
    ----------
    input_dir (str): Path to parent directory containing extracted sessions folders

    Returns
    -------
    '''

    output_notebook()
    interactive_extraction_preview_wrapper(input_dir)

def validate_extractions(input_dir):
    '''
    Wrapper function that facilitates the extraction validation step from `main.py`.
     Prints all the flagged session outlier details.

    Parameters
    ----------
    input_dir (str): Path to parent directory containing extracted sessions folders

    Returns
    -------
    '''

    validate_extractions_wrapper(input_dir)
