# The first project cell (cell 4) of notebooks/MoSeq2-Analysis-Visualization-Notebook.ipynb at the v1.5.0 tag, unchanged.

from os.path import join, dirname, abspath
from moseq2_app.gui.progress import update_progress, restore_progress_vars, progress_path_sanity_check

progress_filepath = './progress.yaml' # Add the path to your progress.yaml here.

# set to True if you are coming from the CLI and did not use the extraction and modeling notebook.
# keep False if you came from the extraction and modeling notebook. Most users will keep this False
from_CLI = False
# keep False if this is not a model object from applying a pre-trained model to new data. Most users will keep this False
pretrained_model_used = False

progress_paths = restore_progress_vars(progress_file=progress_filepath, init=from_CLI, overwrite=False)

# Find the best model when coming from extraction and modeling from the CLI.
if from_CLI and not pretrained_model_used:
    # Find the best model given the objective
    from moseq2_viz.gui import get_best_fit_model
    output_file = join(progress_paths['base_dir'], 'model_vs_pc_changepoints')
    objective = 'median_loglikelihood'  # objective could be either duration, jsd or median_loglikelihood
    best_model_fit = get_best_fit_model(progress_paths, output_file, plot_all=True, objective=objective)
    progress_paths = update_progress(progress_filepath, 'model_path', best_model_fit[f'best model - {objective}'])

progress_path_sanity_check(progress_paths, progress_filepath) # check all necessary paths exist

progress_paths
