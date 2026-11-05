# The first project cell (cell 3) of notebooks/[OPTIONAL]MoSeq2-Apply-Pre-Trained-Model-Notebook.ipynb at the v1.5.0 tag, unchanged.

from os.path import join
from moseq2_app.gui.progress import check_progress, restore_progress_vars

base_dir = './' # Add the path to your data folder here.
progress_filepath = join(base_dir, 'progress.yaml')

progress_paths = restore_progress_vars(progress_filepath, init=True, overwrite=False)
check_progress(progress_filepath)
