"""
General utility functions.
"""
import pandas as pd
from moseq2_common.yaml import write_config
from copy import deepcopy
from dataclasses import dataclass
from os.path import basename, splitext
from pathlib import Path, PurePosixPath
from contextlib import contextmanager
from typing import Iterator
from moseq2_common.bookkeeping import Bookkeeping, BookkeepingError, load, relative_path, resolve_path, update
from moseq2_common.options import SECTIONS, check_top_level, read_config
from moseq2_common.project import Project, outdated_names
from IPython.display import clear_output
from moseq2_viz.scalars.util import scalars_to_dataframe
from moseq2_extract.util import check_filter_sizes
from moseq2_extract.session_config import extract_settings
from moseq2_viz.model.util import compute_behavioral_statistics

#: What the 1.5 notebooks import from here that 1.6.0 removed: each stops with a message to download
#: the 1.6 notebooks.
__getattr__ = outdated_names(__name__, ("read_yaml", "setup_model_folders", "update_model_paths",
                                        "camera_for_pca"))

#: The folder inside the project holding the fitted models.
MODELS_DIRNAME = "models"


class ClearOutputMixin:
    """
    The Clear Output button's handler, shared by the notebook tools that show one.
    """

    def clear_on_click(self, b=None):
        """
        Clear the cell output

        Args:
        b (button click)
        """

        clear_output()


def read_and_clean_config(config_file):
    """read config.yaml's extraction settings (its ``extract`` section and ``camera_type``) and reset incorrect ones

    Args:
        config_file (dict): path to the config file

    Returns:
        config_data (dict): dictionary of config data.
    """
    config_data = extract_settings(config_file)
    config_data = check_filter_sizes(config_data)
    config_data['threads'] = max(1, config_data.get('threads', 8))

    # TODO: see if this is necessary
    # set defaults if the keys don't exist
    config_data = {
        'bg_roi_erode': (1, 1),
        'bg_roi_dilate': (1, 1),
        **config_data
    }

    return config_data


def merge_labels_with_scalars(sorted_index, model_path):
    """
    Compute all the syllable statistics to plot, including syllable scalars.

    Args:
    sorted_index (dict): Sorted dict of modeled sessions
    model_fit (dict): Trained AR-HMM results dict
    model_path (str): Respective path to the AR-HMM model in use.
    max_sylls (int): Maximum number of syllables to include

    Returns:
    df (pd.DataFrame): Dataframe containing all of the mean syllable statistics
    scalar_df (pd.DataFrame): Dataframe containing the frame-by-frame scalar and label data
    """

    # Load scalar Dataframe to compute syllable speeds
    scalar_df = scalars_to_dataframe(sorted_index, model_path=model_path)

    df = compute_behavioral_statistics(scalar_df, count='usage',
                                       groupby=['group', 'uuid', 'SessionName', 'SubjectName'])

    return df, scalar_df

def index_to_dataframe(index_data):
    """
    Turn the project's sessions into an editable DataFrame.

    Args:
    index_data (dict): the sessions (``moseq2_common.bookkeeping.session_index``)

    Returns:
    index_data (dict): the sessions, as given
    df (pd.DataFrame): Formatted dict in DataFrame form including each session's metadata
    """

    files = index_data['files']
    meta = [f['metadata'] for f in files]

    meta_df = pd.DataFrame(meta)
    tmp_df = pd.DataFrame(files)

    df = pd.concat([meta_df, tmp_df], axis=1)

    def apply_filename(n):
        return basename(n[0])

    df['filename'] = df['path'].apply(apply_filename)

    return index_data, df

class bcolors:
    """
    color UNICODE values used to color printed output.
    """

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def uuid_lookup(target_uuid, uuid_dict_source):
    """
    Look up session infomtion with full/partial uuid. Helper function for users to look up uuid after running interactive_scalar_summary

    Args:
    target_uuid (str): full or partial uuid the user wants to look up.
    uuid_dict (dict): dictionary from interactive_scalar_summary widget that has all the session information 
    """
    
    # deep copy the input dictionary
    uuid_dict = deepcopy(uuid_dict_source)

    for uuid, info in uuid_dict.items():
        if target_uuid in uuid:
            print('UUID:', uuid)
            # Printed a field at a time rather than as a dict, so that a path
            # arrives as a path. Rendering the dict quotes every value and
            # escapes each separator, which on Windows turns a path the user
            # is being handed in order to go and open it into one they have to
            # edit first.
            metadata = info['metadata']
            metadata['h5Path'] = info['path'][0]
            for field in ('SessionName', 'SubjectName', 'StartTime', 'h5Path'):
                print(f'  {field}: {metadata[field]}')

@dataclass(frozen=True)
class Analysis:
    """The model the Analysis notebook analyses, and the folder its analysis is written in.

    The folder is the model's own, beside it and named after it (``models/model/``
    for ``models/model.p``), so two models analysed in one project keep their
    outputs apart.
    """

    project: Project
    #: The model file, where it was fit.
    model: Path
    #: The folder the analysis of this model writes into.
    folder: Path

    @property
    def plots(self) -> Path:
        return self.folder / "plots"

    @property
    def crowd_movies(self) -> Path:
        return self.folder / "crowd_movies"

    @property
    def syll_info(self) -> Path:
        """The syllable labels the labeler writes."""
        return self.folder / "syll_info.yaml"


def _no_current_model(project: Project) -> BookkeepingError:
    return BookkeepingError(
        f"{project.bookkeeping_file} names no current model yet, so there is no model to analyze. Fit one "
        "with the Extract-Modeling notebook's modeling cell or `moseq2-model learn-model`, or pass the file "
        "name of a model in the project's models folder to choose_model."
    )


def _analysis(project: Project, bookkeeping: Bookkeeping, model: str) -> Analysis:
    folder = (bookkeeping.models.get("analysis") or {}).get(model) or splitext(model)[0]
    return Analysis(project=project, model=resolve_path(project, model), folder=resolve_path(project, folder))


def current_analysis(project_folder) -> Analysis:
    """The project's current model and its analysis folder, read from ``moseq2-project.yaml``.

    Args:
    project_folder (str): the project folder.

    Returns:
    (Analysis): the model and the folder its analysis is written in, which exists.
    """
    project = Project.at(project_folder)
    bookkeeping = load(project)
    model = bookkeeping.models.get("current")
    if not model:
        raise _no_current_model(project)
    analysis = _analysis(project, bookkeeping, model)
    if not analysis.model.is_file():
        raise FileNotFoundError(
            f"{project.bookkeeping_file} names {analysis.model} as the current model, and it does not exist. "
            "Fit it again, or choose another of the project's models with choose_model."
        )
    analysis.folder.mkdir(parents=True, exist_ok=True)
    return analysis


def _model_named(project: Project, bookkeeping: Bookkeeping, model_name: str) -> str:
    """The model ``model_name`` names, as the bookkeeping file stores it."""
    listed = [*(bookkeeping.models.get("fitted") or []), *bookkeeping.applied_models]
    found = [model for model in listed if model == model_name or PurePosixPath(model).name == model_name]
    if len(found) == 1:
        return found[0]
    for candidate in (resolve_path(project, model_name), Path(project.folder) / MODELS_DIRNAME / model_name):
        if candidate.is_file():
            return relative_path(project, candidate)
    models_folder = Path(project.folder) / MODELS_DIRNAME
    in_folder = [relative_path(project, path) for path in sorted(models_folder.glob("*.p"))]
    choices = sorted(set(listed) | set(in_folder))
    if found:
        problem = f"{len(found)} of the project's models are named {model_name!r}: {', '.join(found)}"
    else:
        problem = f"The project has no model {model_name!r}"
    raise BookkeepingError(
        f"{problem}. Its models are: {', '.join(choices) if choices else 'none yet'} "
        f"({project.bookkeeping_file} and {models_folder}). Pass one of those to choose_model as model_name."
    )


def choose_model(project_folder, model_name=None) -> Analysis:
    """
    Choose the model the Analysis notebook analyses, and give its analysis a folder.

    The model becomes the project's current model in ``moseq2-project.yaml``, which
    every later analysis cell reads. Nothing is copied: the analysis reads the model
    where it was fit, so a model fit again under the same name is what it analyses.
    Its analysis folder is the model's own folder beside it, named after it.

    Args:
    project_folder (str): the project folder.
    model_name (str): the model's file name in the project's ``models`` folder, or
        its path in the project; None for the project's current model.

    Returns:
    (Analysis): the model and the folder its analysis is written in.
    """
    project = Project.at(project_folder)
    bookkeeping = load(project)
    if model_name is None:
        model = bookkeeping.models.get("current")
        if not model:
            raise _no_current_model(project)
    else:
        model = _model_named(project, bookkeeping, str(model_name))
    folder = (bookkeeping.models.get("analysis") or {}).get(model) or splitext(model)[0]

    def change(bookkeeping):
        fitted = bookkeeping.models.setdefault("fitted", [])
        if model not in fitted and model not in bookkeeping.applied_models:
            fitted.append(model)
        bookkeeping.models["current"] = model
        bookkeeping.models.setdefault("analysis", {})[model] = folder

    if (bookkeeping.models.get("current") != model
            or (bookkeeping.models.get("analysis") or {}).get(model) != folder):
        update(project, change)
    analysis = current_analysis(project_folder)
    print(f"Analyzing {analysis.model}; its analysis is saved in {analysis.folder}")
    return analysis


@contextmanager
def update_config(path: str, section: str | None = None) -> Iterator[dict]:
    """Edit config.yaml in a notebook cell: yield one section (or, with ``section`` None, the whole file).

    What the cell leaves in the yielded dict is written back when the ``with``
    block ends, the rest of the file as it was::

        with update_config(config_file, section='pca') as config_data:
            config_data['nworkers'] = 4

    Args:
        path (str): path to config file.
        section (str): ``extract``, ``pca``, ``model`` or ``crowd_movies``; None for the whole file.
    """
    if section is not None and section not in SECTIONS:
        raise ValueError(f"config.yaml has no section {section!r}; its sections are {', '.join(SECTIONS)}. "
                         "Pass one of those as update_config's section.")
    config = read_config(path)
    if section is None:
        edited = config
    else:
        check_top_level(path, config)
        edited = dict(config.get(section) or {})
    try:
        yield edited
    finally:
        if section is not None:
            config[section] = edited
        write_config(config, path)
