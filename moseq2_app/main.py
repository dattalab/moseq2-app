"""
Main functions that facilitate all jupyter notebook functionality.
"""
import re
from os.path import exists, join
import ipywidgets as widgets
from IPython.display import display
from bokeh.io import output_notebook
from moseq2_common.bookkeeping import project_index, session_index
from moseq2_common.options import read_settings
from moseq2_common.project import Project
from moseq2_viz.options import CROWD_MOVIE_STAGE, CrowdMovieOptions
from moseq2_extract.util import filter_warnings
from moseq2_app.gui.widgets import GroupSettingWidgets
from moseq2_app.scalars.controller import InteractiveScalarViewer
from moseq2_app.stat.controller import InteractiveSyllableStats
from moseq2_app.stat.view import plot_dendrogram
from moseq2_app.roi.controller import InteractiveExtractionViewer
from moseq2_app.util import current_analysis
from moseq2_app.gui.wrappers import validate_extractions_wrapper, \
    interactive_syllable_labeler_wrapper, interactive_crowd_movie_comparison_preview_wrapper, \
    interactive_plot_transition_graph_wrapper

output_notebook()


_SELECTION_ITEM = re.compile(r"(e?)\s*(\d+)(?:\s*-\s*(\d+))?", re.IGNORECASE)


def _parse_session_selection(answer, n_sessions):
    """
    Turn an answer at the session prompt into the session numbers it selects.

    The answer is a comma separated list of items. Each item is a session number
    "N", an inclusive range "N-M", or either of those prefixed with "e" to leave
    those sessions out. Without an item that selects sessions, the answer starts
    from all of them. Sessions are numbered from 1.

    Args:
    answer (str): what the user typed.
    n_sessions (int): number of sessions listed.

    Returns:
    selected (list): session numbers in the order first named, each once.

    Raises:
    ValueError: naming the item that is not a number or a range, names a session
    that is not listed or runs backwards, or the answer if it leaves no sessions.
    """
    items = [item.strip() for item in answer.split(",") if item.strip()]
    if not items:
        raise ValueError("Invalid input.")

    selected, excluded = [], set()
    for item in items:
        match = _SELECTION_ITEM.fullmatch(item)
        if match is None:
            raise ValueError(f'Invalid input "{item}": not a number or a range.')
        first, last = int(match[2]), int(match[3] or match[2])
        for number in (first, last):
            if not 1 <= number <= n_sessions:
                raise ValueError(f'Invalid input "{item}": there is no session {number}.')
        if first > last:
            raise ValueError(f'Invalid input "{item}": a range runs from the lower number to the higher.')
        if match[1]:
            excluded.update(range(first, last + 1))
        else:
            selected += [n for n in range(first, last + 1) if n not in selected]

    selected = [n for n in selected or range(1, n_sessions + 1) if n not in excluded]
    if not selected:
        raise ValueError(f'Invalid input "{answer.strip()}": that leaves no sessions to extract.')
    return selected


def _selected_sessions(to_extract, extract_all):
    """
    The sessions a person picks from ``to_extract`` at a numbered prompt, or all of them.

    Args:
    to_extract (list): list of paths to sessions to extract
    extract_all (bool): boolean to include all sessions and skip user-input prompt.

    Returns:
    to_extract (list): new list of selected sessions to extract.
    """

    if len(to_extract) > 1 and not extract_all:
        for i, sess in enumerate(to_extract):
            print(f"[{str(i + 1)}] {sess}")

        print("You may input comma separated values for individual sessions")
        print(
            'Or you can input a hyphen separated range. E.g. "1-10" selects 10 sessions, including sessions 1 and 10'
        )
        print(
            'You can also exclude a range by prefixing the range selection with the letter "e"; e.g.: "e1-5".'
        )
        print("Press q to quit.")
        while True:
            sessions = input("Input your selected sessions to extract: ")
            if sessions.strip().lower() == "q":
                return []
            try:
                return [to_extract[n - 1] for n in _parse_session_selection(sessions, len(to_extract))]
            except ValueError as error:
                print(f"{error} Try again or press q to quit.")
    else:
        return to_extract


@filter_warnings
def view_extraction(extractions, default=0):
    """
    Prompt user to select which extracted video(s) to preview.

    Args:
    extractions (list): list of paths to all extracted avi videos.
    default (int): index of the default extraction to display

    Returns:
    extractions (list): list of selected extractions.
    """

    if len(extractions) == 0:
        print('no sessions to view')
        return []

    if default < 0:
        for i, sess in enumerate(extractions):
            print(f'[{str(i + 1)}] {sess}')
        extractions = _selected_sessions(extractions, False)
    else:
        print(f"Displaying {extractions[default]}")
        return [extractions[default]]

    return extractions

@filter_warnings
def preview_extractions(input_dir, flipped=False):
    """
    launch a dynamic video loader that displays extraction session mp4s.

    Args:
    input_dir (str): Path to parent directory containing extracted sessions folders
    flipped (bool): indicates whether to show corrected flip videos
    """
    output_notebook()
    viewer = InteractiveExtractionViewer(data_path=input_dir, flipped=flipped)

    # Run interactive application
    selout = widgets.interactive_output(viewer.get_extraction,
                                        {'input_file': viewer.sess_select})
    display(viewer.clear_button, viewer.sess_select, selout)

@filter_warnings
def validate_extractions(input_dir):
    """
    validate extracted sessions and print validation results.

    Args:
    input_dir (str): Path to parent directory containing extracted sessions folders
    """

    validate_extractions_wrapper(input_dir)

@filter_warnings
def interactive_group_setting(project_folder):
    """
    display a excel-like table to set each session's group, saved in the project's config.yaml

    Args:
    project_folder (str): the project folder.
    """

    index_grid = GroupSettingWidgets(project_folder)

    # Display output
    display(index_grid.clear_button, index_grid.group_set)
    display(index_grid.qgrid_widget)

    return index_grid

@filter_warnings
def interactive_scalar_summary(project_folder):
    """
    launch Interactive Scalar summary visualization tool.

    Args:
    project_folder (str): the project folder, whose sessions it plots the scalars of.
    """

    project = Project.at(project_folder)
    index = project_index(project)
    if index is None:
        print(f'{project.bookkeeping_file} lists no sessions yet. Run `moseq2 extract` (or the notebook\'s '
              'Extract step) first, then run this cell again.')
        return

    viewer = InteractiveScalarViewer(index)
    display(viewer.ui_tools)
    return viewer

def _crowd_movie_settings(project):
    """The settings of ``project``'s config.yaml ``crowd_movies`` section, as ``CrowdMovieOptions.resolve`` takes them."""
    return read_settings(project.config_file, CrowdMovieOptions, stage_classes=CROWD_MOVIE_STAGE)


def _labels_needed(analysis):
    """Stop, saying what to run, when the syllable labeler has not written the analysis's label file yet."""
    if not exists(analysis.syll_info):
        raise FileNotFoundError(
            f"{analysis.syll_info} does not exist yet. The syllable labeler writes it for {analysis.model}: "
            "run the notebook's labeling cell (label_syllables) first, then run this cell again."
        )


def _folder(path):
    """``path`` as a folder name the analysis tools join file names onto."""
    return join(str(path), '')


@filter_warnings
def label_syllables(project_folder, max_syllables=None, n_explained=99):
    """
    launch Interactive syllable labeling tool.

    The model is the project's current model (``moseq2_app.util.choose_model``); the crowd
    movies are rendered into its analysis folder with the settings of config.yaml's
    ``crowd_movies`` section, and the labels are saved there as ``syll_info.yaml``.

    Args:
    project_folder (str): the project folder.
    max_syllables (int or None): manual maximum number of syllables to label.
    n_explained (int): Percentage of explained variance to use to compute max_syllables to compute.

    Returns:
    max_sylls (int): the number of syllables offered for labeling.
    """
    analysis = current_analysis(project_folder)
    return interactive_syllable_labeler_wrapper(str(analysis.model), session_index(analysis.project),
                                                _folder(analysis.crowd_movies), str(analysis.syll_info),
                                                _folder(analysis.plots), _crowd_movie_settings(analysis.project),
                                                max_syllables=max_syllables, n_explained=n_explained)


@filter_warnings
def show_dendrogram(project_folder, max_syllable=None, color_by_cluster=False):
    """
    show syllable similarity dendrogram, saved in the analysis folder's ``plots``.

    Args:
        project_folder (str): the project folder.
        max_syllable (int, optional): the number of syllables to include.
        color_by_cluster (bool, optional): color the syllables by similarity cluster.
    """
    analysis = current_analysis(project_folder)
    plot_dendrogram(index=session_index(analysis.project), model_path=str(analysis.model),
                    syll_info_path=str(analysis.syll_info), save_dir=_folder(analysis.plots),
                    max_syllable=max_syllable, color_by_cluster=color_by_cluster)


@filter_warnings
def interactive_syllable_stats(project_folder, max_syllable=None):
    """
    launch the interactive syllable statistics viewer.

    Args:
    project_folder (str): the project folder.
    max_syllable (int or None): manual maximum number of syllables to plot.
    """
    analysis = current_analysis(project_folder)
    _labels_needed(analysis)

    # Initialize the statistical grapher context
    istat = InteractiveSyllableStats(index=session_index(analysis.project), model_path=str(analysis.model),
                                     info_path=str(analysis.syll_info), max_sylls=max_syllable)

    display(istat.clear_button, istat.stat_widget_box, istat.out)


@filter_warnings
def interactive_crowd_movie_comparison(project_folder, group_movie_dir, get_pdfs=True):
    """
    launch interactive crowd movie/position heatmap comparison function

    Args:
    project_folder (str): the project folder.
    group_movie_dir (str): path to generate new grouped crowd movies in.
    get_pdfs (bool): indicates whether to also generate position heatmaps.
    """
    analysis = current_analysis(project_folder)
    _labels_needed(analysis)

    interactive_crowd_movie_comparison_preview_wrapper(_crowd_movie_settings(analysis.project),
                                                       session_index(analysis.project), str(analysis.model),
                                                       str(analysis.syll_info), group_movie_dir, get_pdfs=get_pdfs)


@filter_warnings
def interactive_transition_graph(project_folder, max_syllables=None, plot_vertically=False):
    """
    Display group transition graphs with a configurable number of syllables.

    Args:
    project_folder (str): the project folder.
    max_syllables (int or None): manual maximum number of syllables to plot.
    plot_vertically (bool): stack the groups' graphs vertically.
    """
    analysis = current_analysis(project_folder)
    _labels_needed(analysis)

    interactive_plot_transition_graph_wrapper(str(analysis.model), session_index(analysis.project),
                                              str(analysis.syll_info), max_syllables=max_syllables,
                                              plot_vertically=plot_vertically)
