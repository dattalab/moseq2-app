"""The extraction pipeline steps, one function each.

``find-roi``, ``extract``, ``batch-extract`` and ``aggregate-results`` on the
command line (``cli.py``) and the Extract notebook's functions (``gui.py``)
resolve their settings into one options object and call these. A step needs no
``config.yaml``: from a script, pass the paths and an options object, such as
``extract("session_1/depth.dat", "proc", ExtractOptions(num_frames=900))``.
``batch_extract`` is the exception, since each recording's settings come from
``config.yaml``'s ``extract`` section and its entry in ``session_config.yaml``.

Each step hands its work function (``get_roi_wrapper``, ``extract_wrapper``,
``aggregate_extract_results_wrapper`` in ``moseq2_extract.helpers.wrappers``)
the options as a dict, and extraction records that dict among the extraction's
parameters (``metadata/extraction/parameters`` in the results ``.h5``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os.path import abspath, join
from pathlib import Path

from moseq2_common.project import Project
from moseq2_extract.helpers.extract import run_local_extract, run_slurm_extract
from moseq2_extract.helpers.wrappers import aggregate_extract_results_wrapper, extract_wrapper, get_roi_wrapper
from moseq2_extract.options import AggregateOptions, BatchExtractOptions, ExtractOptions, RoiOptions
from moseq2_extract.recordings import find_recordings

#: ExtractOptions fields extract_wrapper takes as arguments rather than in its dict.
_WRAPPER_ARGUMENTS = ("num_frames", "skip_completed")


@dataclass(frozen=True)
class FindRoiResult:
    #: The arena masks written (``roi_XX.tiff``), one per ``bg_roi_index`` found.
    roi_files: list


@dataclass(frozen=True)
class ExtractResult:
    #: The extraction's ``results_XX.h5``; None when the recording was skipped as already extracted.
    h5_file: Path | None


@dataclass(frozen=True)
class BatchExtractResult:
    #: The recordings the search found, before any were picked.
    found: list
    #: The recordings extracted locally, or skipped there as already extracted.
    extracted: list
    #: (recording, exception) for each local extraction that failed, in the order tried.
    failures: list
    #: The script of sbatch commands, when the cluster type is slurm.
    script: Path | None


@dataclass(frozen=True)
class AggregateResult:
    #: The index file written for the aggregated sessions.
    index_file: Path | None


def _settings(opts, leave_out=()):
    """The dict a wrapper takes: the options, but ``leave_out``."""
    return {name: value for name, value in opts.as_dict().items() if name not in leave_out}


def find_roi(input_file: str, output_dir: str | None, opts: RoiOptions) -> FindRoiResult:
    """Find the arena and background of ``input_file``; write the background and arena images."""
    settings = _settings(opts)
    get_roi_wrapper(input_file, settings, output_dir)
    # the wrapper records in settings the folder it wrote to
    folder = settings.get("output_dir")
    indices = opts.bg_roi_index if isinstance(opts.bg_roi_index, (list, tuple)) else [opts.bg_roi_index]
    written = [Path(folder) / f"roi_{idx:02d}.tiff" for idx in indices] if folder else []
    return FindRoiResult(roi_files=[path for path in written if path.is_file()])


def extract(input_file: str, output_dir: str | None, opts: ExtractOptions) -> ExtractResult:
    """Extract ``input_file`` into ``<recording folder>/<output_dir>`` (``proc`` when None)."""
    settings = _settings(opts, leave_out=_WRAPPER_ARGUMENTS)
    if output_dir is None:
        output_dir = "proc"
    folder = extract_wrapper(input_file, output_dir, settings, num_frames=opts.num_frames,
                             skip=opts.skip_completed)
    if folder is None:
        return ExtractResult(h5_file=None)
    index = opts.bg_roi_index[0] if isinstance(opts.bg_roi_index, (list, tuple)) else opts.bg_roi_index
    return ExtractResult(h5_file=Path(folder) / f"results_{index:02d}.h5")


def batch_extract(input_dir: str, opts: BatchExtractOptions, *, config_file: str,
                  output_dir: str | None = None, explicit: Mapping | None = None,
                  select: Callable[[Sequence[str]], Sequence[str]] | None = None,
                  project: Project | None = None) -> BatchExtractResult:
    """Find the recordings under ``input_dir`` and extract each, here or as SLURM jobs.

    The search looks for ``opts.extensions`` down to ``opts.max_depth`` folders
    below ``input_dir`` and, unless ``opts.skip_checks`` is set, leaves out each
    recording whose ``results_00.yaml`` in its output folder records a finished
    extraction. The command line and the notebook both list recordings this way.

    Each recording is extracted with ``config.yaml``'s ``extract`` settings, the
    arena tool's settings from its entry in ``session_config.yaml``, and
    ``explicit`` (options typed on the command line) over both
    (``session_config.recording_configs``).

    Args:
    input_dir: the folder to search.
    opts: the search, the cluster type and the SLURM job settings; ``num_frames`` and
        ``skip_completed`` apply to every recording.
    config_file: the project's ``config.yaml``.
    output_dir: each recording's output folder, below its own folder; None for ``proc``.
    explicit: options typed on the command line, applied over each recording's settings.
    select: picks the recordings to extract from those found (the notebook's prompt).
    project: the project each SLURM job's settings file names (``moseq2_common.project.write_job_settings``).
    """
    folder = output_dir if output_dir is not None else "proc"
    search = find_recordings(input_dir, extensions=opts.extensions, max_depth=opts.max_depth,
                             skip_extracted=not opts.skip_checks,
                             results_yaml=join(folder, "results_00.yaml"))
    message = search.report()
    if message:
        print(message)
    if not search.recordings:
        return BatchExtractResult(found=[], extracted=[], failures=[], script=None)
    to_extract = list(select(search.recordings) if select is not None else search.recordings)

    if opts.cluster_type == "local":
        failures = run_local_extract(to_extract, config_file, opts.skip_completed, num_frames=opts.num_frames,
                                     output_dir=output_dir, cli_options=dict(explicit) if explicit else None)
        failed = {path for path, _ in failures}
        return BatchExtractResult(found=list(search.recordings),
                                  extracted=[Path(p) for p in to_extract if p not in failed],
                                  failures=failures, script=None)

    job = {**opts.as_dict(), "config_file": abspath(config_file), "output_dir": folder}
    run_slurm_extract(input_dir, to_extract, job, opts.skip_completed, num_frames=opts.num_frames,
                      cli_options=dict(explicit) if explicit else None, project=project)
    # run_slurm_extract records where it wrote the script
    return BatchExtractResult(found=list(search.recordings), extracted=[], failures=[],
                              script=Path(job["extract_out_script"]))


def aggregate_results(input_dir: str, output_dir: str, opts: AggregateOptions) -> AggregateResult:
    """Copy every finished extraction under ``input_dir`` into ``output_dir`` and write the index."""
    index = aggregate_extract_results_wrapper(input_dir, opts.format, output_dir, opts.mouse_threshold)
    return AggregateResult(index_file=Path(index) if index else None)
