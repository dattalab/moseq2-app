from bokeh.io import output_notebook
from moseq2_extract.helpers.data import get_selected_sessions
from moseq2_app.gui.wrappers import interactive_roi_wrapper, interactive_extraction_preview_wrapper, \
     validate_extractions_wrapper, interactive_group_setting_wrapper, interactive_syllable_labeler_wrapper, \
     interactive_syllable_stat_wrapper, interactive_crowd_movie_comparison_preview_wrapper, \
     interactive_plot_transition_graph_wrapper

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

    Parameters
    ----------
    input_dir

    Returns
    -------

    '''

    validate_extractions_wrapper(input_dir)

def interactive_group_setting(index_file):
    '''

    Parameters
    ----------
    index_file

    Returns
    -------
    '''

    interactive_group_setting_wrapper(index_file)

def label_syllables(progress_paths, max_syllables=None, n_explained=90):
    '''

    Parameters
    ----------
    progress_paths
    max_syllables
    n_explained

    Returns
    -------
    '''

    # Get proper input paths
    model_path = progress_paths['model_path']
    config_file = progress_paths['config_file']
    index_file = progress_paths['index_file']
    crowd_dir = progress_paths['crowd_dir']
    syll_info_path = progress_paths['syll_info']

    output_notebook()
    interactive_syllable_labeler_wrapper(model_path, config_file,
                                         index_file, crowd_dir, syll_info_path,
                                         max_syllables=max_syllables, n_explained=n_explained)

def interactive_syllable_stats(progress_paths, max_syllable=None):
    '''

    Parameters
    ----------
    progress_paths
    max_syllable

    Returns
    -------

    '''

    # Get proper input parameters
    index_file = progress_paths['index_file']
    model_path = progress_paths['model_path']
    syll_info_path = progress_paths['syll_info']
    syll_info_df_path = progress_paths['df_info_path']

    output_notebook()
    interactive_syllable_stat_wrapper(index_file, model_path, syll_info_path,
                                      syll_info_df_path, max_syllables=max_syllable)

def interactive_crowd_movie_comparison(progress_paths, group_movie_dir, get_pdfs=True):
    '''

    Parameters
    ----------
    progress_paths
    group_movie_dir
    get_pdfs

    Returns
    -------
    '''

    # Get proper input paths
    model_path = progress_paths['model_path']
    config_file = progress_paths['config_file']
    index_file = progress_paths['index_file']
    syll_info_path = progress_paths['syll_info']
    syll_info_df_path = progress_paths['df_info_path']

    output_notebook()
    interactive_crowd_movie_comparison_preview_wrapper(config_file, index_file, model_path,
                                               syll_info_path, group_movie_dir, syll_info_df_path,
                                               get_pdfs=get_pdfs)

def interactive_transition_graph(progress_paths, max_syllables=None):
    '''

    Parameters
    ----------
    progress_paths
    max_syllables (int)

    Returns
    -------

    '''

    # Get proper input paths
    model_path = progress_paths['model_path']
    index_file = progress_paths['index_file']
    syll_info_path = progress_paths['syll_info']
    syll_info_df_path = progress_paths['df_info_path']

    output_notebook()
    interactive_plot_transition_graph_wrapper(model_path,
                                              index_file,
                                              syll_info_path,
                                              syll_info_df_path,
                                              max_syllables=max_syllables)

