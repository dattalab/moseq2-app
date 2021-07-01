'''

Main functions that facilitate all jupyter notebook functionality. All functions will call a wrapper function
 to handle any non front-end settings.

'''

from os.path import exists
from bokeh.io import output_notebook
from moseq2_extract.util import filter_warnings
from moseq2_extract.gui import get_selected_sessions
from moseq2_app.scalars.controller import InteractiveScalarViewer
from moseq2_app.gui.wrappers import interactive_roi_wrapper, interactive_extraction_preview_wrapper, \
     validate_extractions_wrapper, interactive_group_setting_wrapper, \
     interactive_syllable_labeler_wrapper, interactive_syllable_stat_wrapper, \
     interactive_crowd_movie_comparison_preview_wrapper, interactive_plot_transition_graph_wrapper,\
     get_frame_flips_wrapper

output_notebook()

def validate_inputs(inputs, progress_paths):

    error = False
    for i in inputs:
        if not exists(progress_paths[i]):
            error = True
            print(f'ERROR: {i} not found.')

    return error

@filter_warnings
def flip_classifier_tool(input_dir,
                         output_file,
                         max_frames=1e6,
                         tail_filter_iters=1,
                         space_filter_size=3,
                         continuous_slider_update=True,
                         launch_gui=True):
    '''

    Flip Classifier Notebook main functionality access point.

    Parameters
    ----------
    input_dir (str): Path to base directory containing extraction session folders
    max_frames (int): Maximum number of frames to include in the dataset.
    output_file (str): Path to save the outputted flip classifier.
    tail_filter_iters (int): Number of tail filtering iterations
    prefilter_kernel_size (int): Size of the median spatial filter.
    continuous_slider_update (bool): Indicates whether to continuously update the view upon slider widget interactions.
    launch_gui (bool): Indicates whether to launch the labeling gui or just create the FlipClassifier instance.

    Returns
    -------
    flip_obj (FlipRangeTool): Object that holds all saved interactive functionality and data to be used throughout the
     notebook.
    '''

    flip_obj = get_frame_flips_wrapper(input_dir=input_dir,
                                       output_file=output_file,
                                       max_frames=max_frames,
                                       tail_filter_iters=tail_filter_iters,
                                       space_filter_size=space_filter_size,
                                       continuous_slider_update=continuous_slider_update,
                                       launch_gui=launch_gui
                                       )
    return flip_obj

@filter_warnings
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

@filter_warnings
def interactive_roi_detector(progress_paths, compute_all_bgs=True, autodetect_depths=False, overwrite=False):
    '''
    Function to launch ROI detector interactive GUI in jupyter notebook

    Parameters
    ----------
    progress_paths (dict): dictionary of notebook progress paths.
    compute_all_bgs (bool): if True, computes all the sessions' background images to speed up the UI.
    overwrite (bool): if True, will overwrite the previously saved session_config.yaml file

    Returns
    -------
    '''

    config_file = progress_paths['config_file']
    session_config = progress_paths['session_config']

    interactive_roi_wrapper(progress_paths.get('base_dir', './'),
                            config_file,
                            session_config,
                            compute_bgs=compute_all_bgs,
                            autodetect_depths=autodetect_depths,
                            overwrite=overwrite)

@filter_warnings
def preview_extractions(input_dir, flipped=False):
    '''
    Function to launch a dynamic video loader that displays extraction session mp4s.

    Parameters
    ----------
    input_dir (str): Path to parent directory containing extracted sessions folders
    flipped (bool): indicates whether to show corrected flip videos

    Returns
    -------
    '''
    output_notebook()
    interactive_extraction_preview_wrapper(input_dir, flipped=flipped)

@filter_warnings
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

@filter_warnings
def interactive_group_setting(index_file):
    '''

    Interactive group setting wrapper function that displays a excel-like table to update
    the current group selection

    Parameters
    ----------
    index_file (str): Path to index file to update.

    Returns
    -------
    '''

    interactive_group_setting_wrapper(index_file)

@filter_warnings
def interactive_scalar_summary(index_file):
    '''
    Interactive Scalar summary visualization tool accessible from jupyter notebook.

    Parameters
    ----------
    index_file (str): Path to index file containing session paths to plot scalars for.

    Returns
    -------
    '''

    if not exists(index_file):
        print('Index file does not exist. Input path to an existing file and run the function again.')
        return

    viewer = InteractiveScalarViewer(index_file)
    return viewer

@filter_warnings
def label_syllables(progress_paths, max_syllables=None, n_explained=99):
    '''
    Interactive syllable labeling tool accessible from the jupyter notebook.

    Parameters
    ----------
    progress_paths (dict): dictionary of notebook progress paths.
    max_syllables (int or None): manual maximum number of syllables to label.
    n_explained (int): Percentage of explained variance to use to compute max_syllables to compute.

    Returns
    -------
    '''

    # Get proper input paths
    model_path = progress_paths['model_path']
    config_file = progress_paths['config_file']
    index_file = progress_paths['index_file']
    crowd_dir = progress_paths['crowd_dir']
    syll_info_path = progress_paths['syll_info']

    inputs = ['model_path', 'config_file', 'index_file']

    error = validate_inputs(inputs, progress_paths)

    if error:
        print('Set the correct paths to the missing variables and run the function again.')
        return

    interactive_syllable_labeler_wrapper(model_path, config_file,
                                         index_file, crowd_dir, syll_info_path,
                                         max_syllables=max_syllables, n_explained=n_explained)

@filter_warnings
def interactive_syllable_stats(progress_paths, max_syllable=None, load_parquet=False):
    '''
    Generates the interactive syllable statistics viewer, consisting of a dot-line plot and a dendrogram.

    Parameters
    ----------
    progress_paths (dict): dictionary of notebook progress paths.
    max_syllables (int or None): manual maximum number of syllables to label.
    load_parquet (bool): Indicates to load previously saved data.

    Returns
    -------
    '''

    # Get proper input parameters
    index_file = progress_paths['index_file']
    model_path = progress_paths['model_path']
    syll_info_path = progress_paths['syll_info']
    syll_info_df_path = progress_paths['df_info_path']

    inputs = ['model_path', 'index_file', 'syll_info']

    error = validate_inputs(inputs, progress_paths)

    if error:
        print('Set the correct paths to the missing variables and run the function again.')
        return

    interactive_syllable_stat_wrapper(index_file, model_path, syll_info_path,
                                      syll_info_df_path, max_syllables=max_syllable, load_parquet=load_parquet)

@filter_warnings
def interactive_crowd_movie_comparison(progress_paths, group_movie_dir, get_pdfs=True, load_parquet=False):
    '''
    Interactive crowd movie/position heatmap comparison function. Launched via the notebook.

    Parameters
    ----------
    progress_paths (dict): dictionary of notebook progress paths.
    group_movie_dir (str): path to generate new grouped crowd movies in.
    get_pdfs (bool): indicates whether to also generate position heatmaps.
    load_parquet (bool): Indicates to load previously saved data.

    Returns
    -------
    '''

    # Get proper input paths
    model_path = progress_paths['model_path']
    config_file = progress_paths['config_file']
    index_file = progress_paths['index_file']
    syll_info_path = progress_paths['syll_info']
    syll_info_df_path = progress_paths['df_info_path']

    inputs = ['model_path', 'config_file', 'index_file', 'syll_info']

    error = validate_inputs(inputs, progress_paths)

    if error:
        print('Set the correct paths to the missing variables and run the function again.')
        return

    interactive_crowd_movie_comparison_preview_wrapper(config_file, index_file, model_path,
                                               syll_info_path, group_movie_dir, syll_info_df_path,
                                               get_pdfs=get_pdfs, load_parquet=load_parquet)

@filter_warnings
def interactive_transition_graph(progress_paths, max_syllables=None, plot_vertically=False, load_parquet=False):
    '''

    Displays group transition graphs with a configurable number of syllables. Launched via the
     the jupyter notebook.

    Parameters
    ----------
    progress_paths (dict): dictionary of notebook progress paths.
    max_syllables (int or None): manual maximum number of syllables to label.
    load_parquet (bool): Indicates to load previously saved data.

    Returns
    -------
    '''

    # Get proper input paths
    model_path = progress_paths['model_path']
    index_file = progress_paths['index_file']
    syll_info_path = progress_paths['syll_info']
    syll_info_df_path = progress_paths['df_info_path']

    inputs = ['model_path', 'index_file', 'syll_info']

    error = validate_inputs(inputs, progress_paths)
    
    if error:
        print('Set the correct paths to the missing variables and run the function again.')
        return

    interactive_plot_transition_graph_wrapper(model_path,
                                              index_file,
                                              syll_info_path,
                                              syll_info_df_path,
                                              max_syllables=max_syllables,
                                              load_parquet=load_parquet,
                                              plot_vertically=plot_vertically)
