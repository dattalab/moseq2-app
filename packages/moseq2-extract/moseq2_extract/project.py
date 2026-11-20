"""The extraction steps in a project: the project layer.

Each function here is one step run in a project. It takes the project
(``moseq2_common.project.Project``) first, then what a person can type for the
step and the step's resolved options object, and returns the step's result. It
resolves what the project decides for the step: where the recording's outputs go
(its session's folder in ``aggregate_results/``, ``session_outputs``), and the
settings files the step reads (``settings_files``). It starts the run's record
with the project, so the record names the project folder and describes each
input relative to it, and calls the file layer (``moseq2_extract.steps``) with
explicit paths and that record. The file layer fills the record in and writes it
with the outputs.

The file layer is reached through its module (``steps.find_roi``), never by a
name imported from it.
"""

from __future__ import annotations

from moseq2_common.project import Project, session_key, settings_files
from moseq2_common.records import start_record
from moseq2_extract import steps
from moseq2_extract.options import RoiOptions
from moseq2_extract.recordings import session_outputs

__all__ = ["find_roi"]


def find_roi(project: Project, input_file: str, opts: RoiOptions, *,
             output_dir: str | None = None) -> steps.FindRoiResult:
    """Find the arena and background of ``input_file`` in ``project``; write the background and arena images.

    They go into ``output_dir`` when given, else into the recording's session folder in the project's
    ``aggregate_results/``, where ``extract`` writes its outputs. The record lists the recording, then the
    project's ``config.yaml`` and ``session_config.yaml``.
    """
    record = start_record("find-roi", opts, [input_file, *settings_files(project)], project=project)
    if output_dir is None:
        output_dir = str(session_outputs(project, session_key(project, input_file)).folder)
    return steps.find_roi(input_file, output_dir, opts, record=record)
