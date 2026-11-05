"""
GUI front-end operations for PCA.

Each function takes the project folder, reads the ``pca`` section of its
``config.yaml``, resolves the step's settings and calls the step in
``moseq2_pca.steps``. The steps read the sessions from the project's
``aggregate_results/`` and write into its ``_pca/``; each records what it wrote
in the project's ``moseq2-project.yaml``, which the next function reads (on
every call, so a step run on the command line in between is what it finds).
"""

import warnings
from os.path import join

from moseq2_common.bookkeeping import sessions_folder
from moseq2_common.options import read_settings
from moseq2_common.project import Project
from moseq2_pca import steps
from moseq2_pca.cli import recorded_paths
from moseq2_pca.options import PCA_STAGE, ApplyPCAOptions, ChangepointsOptions, TrainPCAOptions

#: The folder inside the project the notebook's PCA steps write to.
PCA_DIRNAME = '_pca'


def _options(cls, project):
    return cls.resolve(config=read_settings(project.config_file, cls, stage_classes=PCA_STAGE))


def train_pca_command(project_folder, output_file='pca'):
    """
    Train PCA on the project's sessions from within a Jupyter notebook.

    Args:
    project_folder (str): the project folder.
    output_file (str): name of the components file written into the project's ``_pca/`` (``pca`` writes ``pca.h5``).
    """
    project = Project.at(project_folder)
    opts = _options(TrainPCAOptions, project)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        steps.train_pca(str(sessions_folder(project)), join(project_folder, PCA_DIRNAME), output_file, opts,
                        project=project)


def apply_pca_command(project_folder, output_file='pca_scores', components_file=None):
    """
    Compute the PC scores of the project's sessions from within a Jupyter notebook.

    Args:
    project_folder (str): the project folder.
    output_file (str): name of the scores file written into the project's ``_pca/``.
    components_file (str): the PCA components to use; None for the ones the project last trained.

    Returns:
    (str): success string.
    """
    project = Project.at(project_folder)
    opts = _options(ApplyPCAOptions, project)

    result = steps.apply_pca(str(sessions_folder(project)), join(project_folder, PCA_DIRNAME), output_file, opts,
                             components_file=components_file or recorded_paths(project).get('pca_file'),
                             project=project)

    if result.scores_file is not None:
        return 'PCA Scores have been successfully computed.'
    else:
        return 'PCA Scores have not been computed'


def compute_changepoints_command(project_folder, output_file='changepoints'):
    """
    Compute the model-free changepoints of the project's sessions from within a Jupyter notebook.

    Args:
    project_folder (str): the project folder.
    output_file (str): name of the changepoints file written into the project's ``_pca/``.

    Returns:
    (str): success string.
    """
    project = Project.at(project_folder)
    opts = _options(ChangepointsOptions, project)
    recorded = recorded_paths(project)

    steps.compute_changepoints(str(sessions_folder(project)), join(project_folder, PCA_DIRNAME), output_file, opts,
                               components_file=recorded.get('pca_file'),
                               scores_file=recorded.get('pca_file_scores'), project=project)

    return 'Model-free syllable changepoints have been successfully computed.'
