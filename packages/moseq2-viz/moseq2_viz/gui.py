"""
GUI functions for visualizing model results.

"""

from os.path import join, exists
from functools import wraps, partial
from moseq2_viz.util import read_yaml
from moseq2_viz.cli import plot_transition_graph, make_crowd_movies
from moseq2_viz.helpers.wrappers import (
    add_group_wrapper,
    plot_syllable_stat_wrapper,
    plot_scalar_summary_wrapper,
    plot_transition_graph_wrapper,
    copy_h5_metadata_to_yaml_wrapper,
    plot_verbose_pdfs_wrapper,
    plot_mean_group_position_pdf_wrapper,
    get_best_fit_model_wrapper,
    make_crowd_movies_wrapper,
)


def _alias(func, dec_func=None):
    """
    Copy documentation and function signatures across re-used functions (but with different names).

    Args:
    func (function): The function to add alias or to wrap
    dec_func (function): the function to add alias when used a wrapper function.

    Returns:
    inner (function): wrapped function.
    """

    @wraps(func if dec_func is None else dec_func)
    def inner(*args, **kwargs):
        return func(*args, **kwargs)

    if dec_func is None:
        inner.__doc__ = f"This is an alias of {func.__name__}\n" + func.__doc__
    else:
        inner.__doc__ = f"This is an alias of {dec_func.__name__}\n" + dec_func.__doc__
    return inner


def get_groups_command(index_file):
    """
    Return the group names and print the number of groups, sessions and group names.

    Args:
    index_file (str): path to index file

    Returns:
    (int): number of unique groups
    """

    index_data = read_yaml(index_file)

    groups, uuids = [], []
    subjectNames, sessionNames = [], []
    for f in index_data["files"]:
        if f["uuid"] not in uuids:
            uuids.append(f["uuid"])
            groups.append(f["group"])
            subjectNames.append(f["metadata"]["SubjectName"])
            sessionNames.append(f["metadata"]["SessionName"])

    print("Total number of unique subject names:", len(set(subjectNames)))
    print("Total number of unique session names:", len(set(sessionNames)))
    print("Total number of unique groups:", len(set(groups)))

    for i in range(len(subjectNames)):
        print(
            "Session Name:",
            sessionNames[i],
            "; Subject Name:",
            subjectNames[i],
            "; group:",
            groups[i],
        )

    return len(set(groups))


def add_group(
    index_file,
    by="SessionName",
    value="default",
    group="default",
    exact=False,
    lowercase=False,
    negative=False,
):
    """
    Update the index file (moseq2-index.yaml) group names with user defined group names.

    Args:
    index_file (str): path to index file
    by (str): session metadata field to find the match for the value
    value (str or list): value(s) to search for in the by field
    group (str or list): Respective group name(s) to set corresponding sessions as.
    exact (bool): indicate whether to search for exact match.
    lowercase (bool): indicate whether to convert all searched for names to lowercase.
    negative (bool): whether to update the inverse of the found selection.
    """

    gui_data = {"key": by, "exact": exact, "lowercase": lowercase, "negative": negative}

    if isinstance(value, str):
        gui_data.update(
            {
                "value": value,
                "group": group,
            }
        )
        add_group_wrapper(index_file, gui_data)
    elif isinstance(value, list) and isinstance(group, list):
        if len(value) == len(group):
            for v, g in zip(value, group):
                gui_data.update(
                    {
                        "value": v,
                        "group": g,
                    }
                )
                add_group_wrapper(index_file, gui_data)
        else:
            print("ERROR, did not enter equal number of substring values -> groups.")


copy_h5_metadata_to_yaml_command = _alias(copy_h5_metadata_to_yaml_wrapper)


def get_best_fit_model(
    progress_paths,
    output_file=None,
    plot_all=False,
    fps=30,
    ext="p",
    objective="duration (mean match)",
):
    """
    Return the best model in the model foder that is closest to model free changepoint in the given objective.

    Args:
    progress_paths (dict): Dictionary containing paths the to model directory and pc scores file
    output_file (str): fiename for the comparison plot
    plot_all (bool): flag that Indicates whether to plot all the models.
    fps (int): Frames per second.
    ext (str): File extension to search for models with
    objective (str): The objective to compare the results between the models and the model free changepoint.

    Returns:
    best_fit_model (str): Path tp best fit model
    """

    # Check output file path
    if output_file is None:
        output_file = join(progress_paths["plot_path"], "model_vs_pc_changepoints")

    # Get paths to required parameters
    model_dir = progress_paths["base_model_path"]
    if not exists(progress_paths["changepoints_path"]):
        changepoint_path = join(
            progress_paths["pca_dirname"], progress_paths["changepoints_path"] + ".h5"
        )
    else:
        changepoint_path = progress_paths["changepoints_path"]
    # Get best fit model and plot requested curves
    best_fit_model, fig = get_best_fit_model_wrapper(
        model_dir,
        changepoint_path,
        output_file,
        plot_all=plot_all,
        ext=ext,
        fps=fps,
        objective=objective,
    )
    fig.show(warn=False)

    return best_fit_model


@partial(_alias, dec_func=make_crowd_movies_wrapper)
def make_crowd_movies_command(*args, **kwargs):
    # Get default CLI params
    objs = make_crowd_movies.params
    defaults = {tmp.name: tmp.default for tmp in objs if not tmp.required}

    if len(args) == 4:
        *args, config_data = args
    else:
        config_data = kwargs.pop("config_data", None)

    if config_data is None:
        config_data = defaults
    elif isinstance(config_data, dict):
        config_data = {**defaults, **config_data}
    else:
        raise TypeError("config_data needs to be a dictionary")

    return make_crowd_movies_wrapper(*args, config_data=config_data, **kwargs)


plot_stats_command = _alias(plot_syllable_stat_wrapper)

plot_scalar_summary_command = _alias(plot_scalar_summary_wrapper)


@partial(_alias, dec_func=plot_transition_graph_wrapper)
def plot_transition_graph_command(*args, **kwargs):
    # Get default CLI params
    params = {
        tmp.name: tmp.default
        for tmp in plot_transition_graph.params
        if not tmp.required
    }

    if len(args) == 4:
        *args, config_data = args
    else:
        config_data = kwargs.pop("config_data", None)

    if config_data is not None:
        config_data = {**params, **config_data}
    else:
        config_data = params

    return plot_transition_graph_wrapper(*args, config_data=config_data, **kwargs)


plot_mean_group_position_heatmaps_command = _alias(plot_mean_group_position_pdf_wrapper)

plot_verbose_position_heatmaps = _alias(plot_verbose_pdfs_wrapper)
