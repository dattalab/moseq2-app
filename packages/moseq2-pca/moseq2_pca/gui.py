"""
GUI front-end operations for PCA.

Each function takes the project folder, reads the ``pca`` section of its
``config.yaml``, resolves the step's settings and calls the step in the project
layer, ``moseq2_pca.project``. The steps read the sessions the project's
``moseq2-project.yaml`` lists, less those under ``exclude`` in ``config.yaml``,
and write into its ``_pca/``; each records what it wrote in
``moseq2-project.yaml``, which the next function reads (on every call, so a
step run on the command line in between is what it finds). A step that would
write over an output that already exists, with its ``overwrite_pca_*`` setting
off, asks first whether to overwrite it, in the words the command line uses
(``moseq2_common.interface.answered``).
"""

import warnings
from os.path import join

from moseq2_common.interface import answered
from moseq2_common.options import read_settings
from moseq2_common.project import Project
from moseq2_pca import project as project_steps
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

        answered(project_steps.train_pca, project, None, join(project_folder, PCA_DIRNAME), output_file, opts)


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

    result = answered(project_steps.apply_pca, project, None, join(project_folder, PCA_DIRNAME), output_file, opts,
                      components_file=components_file)

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

    answered(project_steps.compute_changepoints, project, None, join(project_folder, PCA_DIRNAME), output_file, opts)

    return 'Model-free syllable changepoints have been successfully computed.'
