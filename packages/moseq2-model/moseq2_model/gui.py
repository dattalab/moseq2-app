"""
GUI front-end functions for training ARHMM.

Each function takes the project folder, reads the ``model`` section of its
``config.yaml``, resolves the step's settings and calls the step in
``moseq2_model.steps``. The PC scores and the sessions with their groups come
from the project's ``moseq2-project.yaml`` and ``config.yaml``, read on every
call. The one-run switches (``get_cmd``, ``run_cmd``, ``verbose``) are the
functions' arguments, not settings.
"""

from os.path import basename, join

from moseq2_common.bookkeeping import BookkeepingError, load, project_index, resolve_path
from moseq2_common.options import read_settings
from moseq2_common.project import Project
from moseq2_model import steps
from moseq2_model.options import MODEL_STAGE, ApplyModelOptions, KappaScanOptions, LearnModelOptions

#: The folders inside the project the notebook writes fitted and applied models to.
MODELS_DIRNAME = "models"
APPLIED_MODELS_DIRNAME = "applied_models"


def _scores(project):
    """The PC scores the project's bookkeeping file names."""
    scores = load(project).pca.get("scores")
    if not scores:
        raise BookkeepingError(
            f"{project.bookkeeping_file} names no PC scores yet. Compute them first, with the notebook's "
            "apply-PCA cell or `moseq2-pca apply-pca` in the project folder."
        )
    return str(resolve_path(project, scores))


def learn_model_command(project_folder, model_name="model.p", *, get_cmd=True, run_cmd=False, verbose=False):
    """
    Train ARHMM from within a Jupyter notebook, on the project's PC scores, into its ``models/`` folder.

    Args:
    project_folder (str): the project folder.
    model_name (str): the model's file name in ``models/``.
    get_cmd (bool): flag to return the kappa scan learn-model commands.
    run_cmd (bool): run the kappa scan's learn-model commands after writing them; otherwise run the script yourself.
    verbose (bool): compute modeling summary - can slow down training.

    Returns:
    None or kappa scan command
    """
    project = Project.at(project_folder)
    input_file = _scores(project)
    output_dir = join(project_folder, MODELS_DIRNAME)
    config = read_settings(project.config_file, LearnModelOptions, stage_classes=MODEL_STAGE)
    index = project_index(project)

    if config.get("kappa") == "scan":
        scan = KappaScanOptions.resolve(config=config)
        assert scan.scan_scale in (
            "log",
            "linear",
        ), 'scan scale must be "log" or "linear"'

        opts = KappaScanOptions.resolve(
            config=config,
            explicit={"get_cmd": get_cmd, "run_cmd": run_cmd, "out_script": join(output_dir, scan.out_script)},
        )
        return steps.kappa_scan(input_file, output_dir, opts, index=index,
                                learn=LearnModelOptions.resolve(config=config), project=project).commands
    else:
        opts = LearnModelOptions.resolve(config=config, explicit={"verbose": verbose})
        steps.learn_model(input_file, join(output_dir, model_name), opts, index=index, project=project)


def apply_model_command(project_folder, model_file, dest_name=None):
    """Apply a pre-trained ARHMM to the project's PC scores from within a Jupyter notebook.

    Args:
        project_folder (str): the project folder.
        model_file (str): path to the pre-trained ARHMM.
        dest_name (str): the applied model's file name in the project's ``applied_models/``; None for
            ``model_file``'s name.
    """
    project = Project.at(project_folder)
    config = read_settings(project.config_file, ApplyModelOptions, stage_classes=MODEL_STAGE)
    opts = ApplyModelOptions.resolve(config=config)

    # apply model to data, each session in its group from config.yaml
    steps.apply_model(
        model_file,
        _scores(project),
        join(project_folder, APPLIED_MODELS_DIRNAME, dest_name or basename(model_file)),
        opts,
        index=project_index(project),
        project=project,
    )
