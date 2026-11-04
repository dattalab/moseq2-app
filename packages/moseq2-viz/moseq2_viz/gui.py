"""
GUI functions for visualizing model results.

"""

from os.path import join, exists

from moseq2_viz import steps
from moseq2_viz.options import BestModelOptions


def get_best_fit_model(
    progress_paths,
    output_file=None,
    plot_all=None,
    fps=None,
    ext=None,
    objective=None,
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
    Each of the four left as None takes ``BestModelOptions``' default.

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
    result = steps.get_best_model(
        model_dir,
        changepoint_path,
        output_file,
        BestModelOptions.resolve(explicit={
            name: value
            for name, value in dict(plot_all=plot_all, ext=ext, fps=fps, objective=objective).items()
            if value is not None
        }),
    )
    result.figure.show(warn=False)

    return result.best_model_info
