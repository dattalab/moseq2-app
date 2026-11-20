"""Extraction's steps on the files they are given: the file layer.

Each function here is one step on explicit files, and knows nothing of
projects: ``find_roi`` and ``extract`` take a recording, an output folder and a
resolved options object (``moseq2_extract.options``), ``generate_config`` writes
the config file it is given, and ``download_flip_file`` downloads a flip
classifier into a folder. None of them reads or writes a project's
``config.yaml``, ``session_config.yaml`` or ``moseq2-project.yaml``, and none
asks a person anything: a choice a step needs (which flip classifier) is a
``moseq2_common.project.ChoiceNeeded`` raised before any work. From a script,
pass the paths and an options object, such as
``extract("session_1/depth.dat", "proc", ExtractOptions(num_frames=900))``.

The project layer, ``moseq2_extract.project``, runs these steps in a project:
it decides where a recording's outputs go (its session's folder in
``aggregate_results/``), the uuid ``moseq2-project.yaml`` gave the session and
the settings files the run reads, starts the run's record with the project, and
calls the function here with that record. Batch extraction and aggregation are
project steps and live there.

Each step hands its work function (``get_roi_wrapper``, ``extract_wrapper`` in
``moseq2_extract.helpers.wrappers``) the options as a dict. Each output carries
the record of the run that wrote it (``moseq2_common.records``): extraction's is
the ``/provenance`` attribute of the session's ``results.h5``, holding the
options after detection; ``find_roi``'s is a ``.yaml`` beside each arena image
(``sidecar_path``). Called without a record, a step starts its own, with no
project. Without an output folder, ``find_roi`` and ``extract`` write into
``proc/`` beside the recording.

Before any work, ``find_roi`` and ``extract`` stop on a recording that does not
exist, is empty or cannot be read (``moseq2_common.records.check_inputs``), and
extraction on a recording whose frames and timestamps disagree in number.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from os.path import abspath, basename, dirname, exists, join
from pathlib import Path

from moseq2_common.project import ChoiceNeeded
from moseq2_common.records import Record, check_inputs, start_record, write_record
from moseq2_common.yaml import write_config
from moseq2_extract.helpers.wrappers import extract_wrapper, get_roi_wrapper
from moseq2_extract.options import ExtractOptions, RoiOptions
from moseq2_extract.recordings import standalone_outputs
from moseq2_extract.setup import _CAMERA_TYPE, CAMERA_PRESETS, _defaults, _stages, project_camera
from moseq2_extract.util import filter_warnings

#: ExtractOptions fields extract_wrapper takes as arguments rather than in its dict.
_WRAPPER_ARGUMENTS = ("num_frames", "skip_completed")


@dataclass(frozen=True)
class FindRoiResult:
    #: The arena masks written (``roi_XX.tiff``), one per ``bg_roi_index`` found.
    roi_files: list


@dataclass(frozen=True)
class ExtractResult:
    #: The extraction's ``results.h5``; None when the recording was skipped as already extracted.
    h5_file: Path | None


#: What to do about a recording that is missing, empty or unreadable.
RECORDING_ADVICE = ("Check that it is the recording you meant and that it was copied completely, or copy it again "
                    "from where it was recorded.")

# The classifiers published before 2026 are scikit-learn 0.20 pickles, and a
# pickle records the module path of the class it was made from --
# sklearn.ensemble.forest, renamed to sklearn.ensemble._forest in 0.22. No
# modern scikit-learn can unpickle them, and no amount of care at this end
# changes that, because the incompatibility is inside the file.
#
# These are the same four forests transcribed into pickle-free .npz archives
# of plain arrays, read by moseq2_extract.extract.flip_classifier. Nothing
# about them names a class or a library version, so they cannot rot the same
# way, and the package no longer imports scikit-learn at all.
#
# The originals are still served at the URLs they always were; these were
# published beside them under new names rather than replacing them, so an
# older moseq2-extract keeps downloading exactly what it downloaded before.
#: The published flip classifiers: each one's label and the URL it downloads from, numbered from 0 in this order.
FLIP_CLASSIFIERS: dict[str, str] = {
    "large mice with fibers (K2)": "https://moseq-data.s3.amazonaws.com/flip-classifiers/flip_classifier_k2_largemicewithfiber.npz",
    "adult male c57s (K2)": "https://moseq-data.s3.amazonaws.com/flip-classifiers/flip_classifier_k2_c57_10to13weeks.npz",
    "mice with Inscopix cables (K2)": "https://moseq-data.s3.amazonaws.com/flip-classifiers/flip_classifier_k2_inscopix.npz",
    "adult male c57s (Azure)": "https://moseq-data.s3.amazonaws.com/flip-classifiers/flip_classifier_azure_temp.npz",
}


def _settings(opts, leave_out=()):
    """The dict a wrapper takes: the options, but ``leave_out``."""
    return {name: value for name, value in opts.as_dict().items() if name not in leave_out}


def find_roi(input_file: str, output_dir: str | None, opts: RoiOptions, *,
             record: Record | None = None) -> FindRoiResult:
    """Find the arena and background of ``input_file``; write the background and arena images.

    They go into ``output_dir`` when given, else into ``proc/`` beside the recording. ``record`` is the run's
    record, started by the caller (``moseq2_extract.project.find_roi`` starts it with the project); without
    one, the step starts its own, its one input the recording.
    """
    check_inputs("find-roi", {input_file: RECORDING_ADVICE})
    if record is None:
        record = start_record("find-roi", opts, [input_file])
    settings = _settings(opts)
    if output_dir is None:
        output_dir = str(standalone_outputs(input_file).folder)
    get_roi_wrapper(input_file, settings, output_dir)
    # the wrapper records in settings the folder it wrote to
    folder = settings.get("output_dir")
    indices = opts.bg_roi_index if isinstance(opts.bg_roi_index, (list, tuple)) else [opts.bg_roi_index]
    written = [Path(folder) / f"roi_{idx:02d}.tiff" for idx in indices] if folder else []
    result = FindRoiResult(roi_files=[path for path in written if path.is_file()])
    record.ran_with({"bg_roi_depth_range": settings["bg_roi_depth_range"]})
    finished = record.finish({"files": sorted(str(p) for p in Path(folder).glob("*.tiff")) if folder else []})
    for roi_file in result.roi_files:
        write_record(roi_file, finished)
    return result


def _output_folder(input_file, output_dir):
    """Where ``input_file``'s outputs go: ``output_dir`` when given (a relative one below the recording's
    folder), else ``proc/`` beside it."""
    if output_dir is not None:
        return Path(abspath(input_file)).parent / output_dir
    return standalone_outputs(input_file).folder


def extract(input_file: str, output_dir: str | None, opts: ExtractOptions, *, uuid: str | None = None,
            record: Record | None = None) -> ExtractResult:
    """Extract ``input_file``.

    Its outputs go into ``output_dir`` when given (a relative one is relative to the recording's folder), else
    into ``proc/`` beside it. ``uuid`` is the session's uuid when it has one already
    (``moseq2_extract.project.extract`` passes the one ``moseq2-project.yaml`` lists); None draws a new one.
    ``record`` is the run's record, started by the caller (the project layer starts it with the project and
    the settings files); without one, the step starts its own. The extraction adds the recording and the
    files read beside it to the record's inputs.
    """
    check_inputs("extract", {input_file: RECORDING_ADVICE})
    if record is None:
        record = start_record("extract", opts, [])
    settings = _settings(opts, leave_out=_WRAPPER_ARGUMENTS)
    folder = _output_folder(input_file, output_dir)
    written = extract_wrapper(input_file, str(folder), settings, num_frames=opts.num_frames,
                              skip=opts.skip_completed, uuid=uuid, record=record)
    if written is None:
        return ExtractResult(h5_file=None)
    return ExtractResult(h5_file=Path(written) / "results.h5")


def generate_config(output_file: str, camera_type: str, *, overwrite: bool) -> Path:
    """Write the config file ``output_file``: every setting at its default, by section, with the camera's preset.

    ``camera_type`` is ``kinect`` (or ``k2``), ``azure``, or ``auto``, which
    takes ``azure`` when every recording in ``output_file``'s folder is an
    ``.mkv`` and ``kinect`` otherwise. An existing file is replaced only when
    ``overwrite`` is True; otherwise it is kept as it is. The sections and
    presets are ``moseq2_extract.setup``'s.
    """
    if exists(output_file) and not overwrite:
        return Path(output_file)
    camera = {"k2": "kinect"}.get(camera_type, camera_type)
    if camera == "auto":
        camera = project_camera(dirname(abspath(output_file)))
    if camera not in CAMERA_PRESETS:
        raise ValueError(f"generate-config takes the camera types auto, kinect (or k2) and azure, not "
                         f"{camera_type!r}. Pass one of those as --camera-type.")
    preset = CAMERA_PRESETS[camera]
    data = {"camera_type": _CAMERA_TYPE[camera]}
    for section, classes in _stages().items():
        data[section] = {**_defaults(classes), **preset.get(section, {})}
    data["exclude"] = []
    write_config(data, output_file)
    return Path(output_file)


def download_flip_file(output_dir: str, selection: int | None = None) -> Path:
    """Download flip classifier number ``selection`` of ``FLIP_CLASSIFIERS`` into ``output_dir``; its path.

    Without a ``selection`` it raises ``ChoiceNeeded`` (answered as ``selection=``) before anything is
    downloaded or written. A negative number, or one past the list, raises ``IndexError``. ``output_dir`` is
    made when missing. ``moseq2_extract.project.download_flip_file`` also names the classifier in a
    project's ``config.yaml``.
    """
    if selection is None:
        raise ChoiceNeeded(FLIP_CLASSIFIERS, what="flip classifier", step="download_flip_file")
    if selection < 0:
        raise IndexError("list index out of range")
    return Path(_fetch(FLIP_CLASSIFIERS[list(FLIP_CLASSIFIERS)[selection]], output_dir))


@filter_warnings
def _fetch(url, output_dir):
    """Download ``url`` into ``output_dir``, made when missing; the downloaded file's path, joined to ``output_dir``
    as given."""
    if not exists(output_dir):
        os.makedirs(output_dir)

    output_filename = join(output_dir, basename(url))

    urllib.request.urlretrieve(url, output_filename)
    print("Successfully downloaded flip file to", output_filename)
    return output_filename
