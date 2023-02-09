'''

Main functions that facilitate all jupyter notebook functionality. All functions will call a wrapper function
 to handle any non front-end settings.

'''
from os.path import exists
import ipywidgets as widgets
from IPython.display import display
from bokeh.io import output_notebook, show
from moseq2_extract.util import filter_warnings
from moseq2_extract.gui import get_selected_sessions
from moseq2_app.flip.controller import FlipRangeTool
from moseq2_app.gui.widgets import GroupSettingWidgets
from moseq2_app.scalars.controller import InteractiveScalarViewer
from moseq2_app.stat.controller import InteractiveSyllableStats
from moseq2_app.stat.view import plot_dendrogram
from moseq2_app.roi.controller import InteractiveExtractionViewer
from moseq2_app.gui.wrappers import validate_extractions_wrapper, \
    interactive_syllable_labeler_wrapper, interactive_crowd_movie_comparison_preview_wrapper, \
    interactive_plot_transition_graph_wrapper

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
                         clean_parameters,
                         max_frames=1e6,
                         continuous_slider_update=True,
                         launch_gui=True):
    '''

    Flip Classifier Notebook main functionality access point.

    Parameters
    ----------
    input_dir (str): Path to base directory containing extraction session folders
    max_frames (int): Maximum number of frames to include in the dataset.
    output_file (str): Path to save the outputted flip classifier.
    clean_parameters (dict): Parameters passed to moseq2_extract.extract.proc.clean_frames 
    continuous_slider_update (bool): Indicates whether to continuously update the view upon slider widget interactions.
    launch_gui (bool): Indicates whether to launch the labeling gui or just create the FlipClassifier instance.

    Returns
    -------
    flip_obj (FlipRangeTool): Flip Classifier Training object that will be used throughout the notebook to
     hold the labeled accepted frame ranges and selected paths/info.
    '''

    flip_finder = FlipRangeTool(input_dir=input_dir,
                                max_frames=max_frames,
                                output_file=output_file,
                                clean_parameters=clean_parameters,
                                launch_gui=launch_gui,
                                continuous_slider_update=continuous_slider_update)

    return flip_finder

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
def preview_extractions(input_dir, flipped=False):
    '''
    Function to launch a dynamic video loader that displays extraction session mp4s.
    Upon extracted session selection, function automatically displays the extraction mp4 video file.

    Parameters
    ----------
    input_dir (str): Path to parent directory containing extracted sessions folders
    flipped (bool): indicates whether to show corrected flip videos

    Returns
    -------
    '''
    output_notebook()
    viewer = InteractiveExtractionViewer(data_path=input_dir, flipped=flipped)

    # Run interactive application
    selout = widgets.interactive_output(viewer.get_extraction,
                                        {'input_file': viewer.sess_select})
    display(viewer.clear_button, viewer.sess_select, selout)

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

    index_grid = GroupSettingWidgets(index_file)

    # Display output
    display(index_grid.clear_button, index_grid.group_set)
    display(index_grid.qgrid_widget)

    return index_grid

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
def label_syllables(progress_paths, max_syllables=None, n_explained=99, select_median_duration_instances=False, max_examples=20):
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
    fig_dir = progress_paths['plot_path']

    inputs = ['model_path', 'config_file', 'index_file']

    if validate_inputs(inputs, progress_paths):
        print('Set the correct paths to the missing variables and run the function again.')
        return

    max_sylls = interactive_syllable_labeler_wrapper(model_path, config_file,
                                         index_file, crowd_dir, syll_info_path, fig_dir,
                                         max_syllables=max_syllables, n_explained=n_explained, 
                                         select_median_duration_instances=select_median_duration_instances, max_examples=max_examples)
    return max_sylls


@filter_warnings
def show_dendrogram(progress_paths, max_syllable=None, color_by_cluster=False):

    # get input paths
    index_file = progress_paths['index_file'] # Path to index file.
    model_path = progress_paths['model_path'] # Path to trained model file.
    syll_info_path = progress_paths['syll_info'] # Path to syllable information file.
    save_dir = progress_paths['plot_path']

    plot_dendrogram(index_file=index_file, model_path=model_path, syll_info_path=syll_info_path, 
                    save_dir=save_dir, max_syllable=max_syllable, color_by_cluster=color_by_cluster)

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
    index_file = progress_paths['index_file'] # Path to index file.
    model_path = progress_paths['model_path'] # Path to trained model file.
    syll_info_path = progress_paths['syll_info'] # Path to syllable information file.
    syll_info_df_path = progress_paths['df_info_path'] # relavant data frame for plotting and stats

    inputs = ['model_path', 'index_file', 'syll_info']

    error = validate_inputs(inputs, progress_paths)

    if error:
        print('Set the correct paths to the missing variables and run the function again.')
        return

    # Initialize the statistical grapher context
    istat = InteractiveSyllableStats(index_path=index_file, model_path=model_path, df_path=syll_info_df_path,
                                     info_path=syll_info_path, max_sylls=max_syllable, load_parquet=load_parquet)

    display(istat.clear_button, istat.stat_widget_box, istat.out)

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
