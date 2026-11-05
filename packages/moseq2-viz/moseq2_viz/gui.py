"""
GUI functions for visualizing model results.

Each function takes the project folder and reads the project's
``moseq2-project.yaml`` for the files it needs, on every call.
"""

from os.path import join

from moseq2_common.bookkeeping import BookkeepingError, load, resolve_path
from moseq2_common.project import Project
from moseq2_viz import steps
from moseq2_viz.options import BestModelOptions

#: The folder inside the project holding the fitted models.
MODELS_DIRNAME = "models/"


def shim_progress_paths(progress_paths, output_file):
    """The Analysis notebook's first cell's 1.5 paths (removed by PR 4.3b, which moves that notebook).

    Its command-line branch hands ``get_best_fit_model`` the progress dictionary:
    the models in ``base_model_path``, the changepoints at ``changepoints_path``
    (or that name in ``pca_dirname``), and no project to record the choice in.
    """
    from os.path import exists

    if output_file is None:
        output_file = join(progress_paths["plot_path"], "model_vs_pc_changepoints")
    changepoints = progress_paths["changepoints_path"]
    if not exists(changepoints):
        changepoints = join(progress_paths["pca_dirname"], changepoints + ".h5")
    return None, progress_paths["base_model_path"], changepoints, output_file


def get_best_fit_model(
    project_folder,
    output_file=None,
    *,
    plot_all=None,
    fps=None,
    ext=None,
    objective=None,
):
    """
    Return the best model in the project's ``models/`` folder, closest to the model-free changepoints in the given
    objective, and make it the project's current model.

    Args:
    project_folder (str): the project folder.
    output_file (str): fiename for the comparison plot; None for ``plots/model_vs_pc_changepoints`` in the project.
    plot_all (bool): flag that Indicates whether to plot all the models.
    fps (int): Frames per second.
    ext (str): File extension to search for models with
    objective (str): The objective to compare the results between the models and the model free changepoint.
    Each of the four left as None takes ``BestModelOptions``' default.

    Returns:
    best_fit_model (str): Path tp best fit model
    """
    if isinstance(project_folder, dict):
        project, model_dir, changepoints, output_file = shim_progress_paths(project_folder, output_file)
    else:
        project = Project.at(project_folder)
        if output_file is None:
            output_file = join(project_folder, "plots", "model_vs_pc_changepoints")
        stored = load(project).pca.get("changepoints")
        if not stored:
            raise BookkeepingError(
                f"{project.bookkeeping_file} names no changepoints yet. Compute them first, with the notebook's "
                "changepoints cell or `moseq2-pca compute-changepoints` in the project folder."
            )
        model_dir, changepoints = join(project_folder, MODELS_DIRNAME), str(resolve_path(project, stored))
    result = steps.get_best_model(
        model_dir,
        changepoints,
        output_file,
        BestModelOptions.resolve(explicit={
            name: value
            for name, value in dict(plot_all=plot_all, ext=ext, fps=fps, objective=objective).items()
            if value is not None
        }),
        project=project,
    )
    result.figure.show(warn=False)

    return result.best_model_info
