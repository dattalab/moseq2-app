"""
Contains helper functions for handling/storing data during extraction.
"""

import tarfile
import warnings
import numpy as np
from importlib.metadata import version
from moseq2_common.timestamps import FRAME_RATE_PATH, rate_from_timestamps
from moseq2_extract.io.video import load_timestamps_from_movie
from os.path import exists, join, basename
from moseq2_extract.util import (
    load_timestamps,
    load_metadata,
    load_textdata,
    dict_to_h5,
    click_param_annot,
)


#: Files an acquisition can leave beside a recording, which extraction copies into the session's ``.h5``
#: under ``/metadata/misc/<name>``: (file, name, data type).
MISC_FILES = (
    # behavior reinforcement experiments
    ("feedback_ts.txt", "realtime_feedback", bool),
    # a pre-trained model's real-time syllable classification
    ("predictions.txt", "realtime_predictions", int),
    # real-time PC scores
    ("pc_scores.txt", "realtime_pc_scores", np.float32),
)


def write_misc_metadata(h5_file, recording_folder):
    """
    Copy the real-time files beside a recording (``MISC_FILES``) into its extraction.

    Args:
    h5_file (h5py.File): the extraction, open for writing.
    recording_folder (str): the folder holding the recording.
    """
    for filename, name, dtype in MISC_FILES:
        path = join(recording_folder, filename)
        if not exists(path):
            continue
        try:
            data, timestamps = load_textdata(path, dtype=dtype)
        except Exception as error:  # noqa: BLE001 -- a file that cannot be read is named and left out
            warnings.warn(f"{path} could not be read ({error}), so it is not copied into the extraction. "
                          "Check that each line holds a timestamp, a space and comma-separated values, then "
                          "extract the recording again to include it.")
            continue
        h5_file.create_dataset(f"/metadata/misc/{name}/data", data=data)
        h5_file.create_dataset(f"/metadata/misc/{name}/timestamps", data=timestamps)


TIMESTAMP_UNIT = "milliseconds"


def _no_timestamps_message(recording, looked_in, empty=None, where="next to it"):
    """The error for a recording with no per-frame timestamps from any source."""
    places = "\n".join(f"  {place}" for place in looked_in)
    found = f" ({empty} is empty)" if empty else ""
    return (
        f"No frame timestamps were found for {recording}{found}. Looked in:\n{places}\n"
        "Without the time of each frame, the frames the camera dropped cannot be "
        "found, and every later step (PCA, changepoints, modeling) would assume "
        f"none were dropped. Put the recording's depth_ts.txt back {where}, as "
        "the acquisition software wrote it, and run the extraction again."
    )


def handle_extract_metadata(input_file, dirname, with_source=False):
    """
    Extract metadata and timestamp in the extraction.

    Args:
    input_file (str): path to input file to extract
    dirname (str): path to directory where extraction files reside.
    with_source (bool): also return where the timestamps were read from.

    Returns:
    acquisition_metadata (dict): key-value pairs of JSON contents
    timestamps (1D array): the time of every frame, in milliseconds
    tar (bool): indicator for whether the file is compressed.
    source (str): where the timestamps were read from: the timestamp file's
        path, ``depth_ts.txt inside <tarball>``, or ``the frame timestamps
        inside <file>`` for an ``.mkv`` read without a timestamp file.
        Returned only when ``with_source`` is True.
    """

    tar = None
    tar_members = None
    alternate_correct = False
    from_depth_file = False
    timestamp_path = None
    recording = input_file
    where = "next to it"

    # Handle TAR files
    if input_file.endswith((".tar.gz", ".tgz")):
        print(f"Scanning tarball {input_file} (this will take a minute)")
        # compute NEW psuedo-dirname now, `input_file` gets overwritten below with test_vid.dat tarinfo...
        dirname = join(
            dirname, basename(input_file).replace(".tar.gz", "").replace(".tgz", "")
        )

        tar = tarfile.open(input_file, "r:gz")
        tar_members = tar.getmembers()
        tar_names = [_.name for _ in tar_members]

    if tar is not None:
        # Handling tar paths
        where = "inside the tarball"
        looked_in = [
            f"depth_ts.txt inside {input_file}",
            f"timestamps.csv inside {input_file}",
        ]
        if "depth_ts.txt" in tar_names:
            timestamp_path = tar.extractfile(
                tar_members[tar_names.index("depth_ts.txt")]
            )
            source = looked_in[0]
        elif "timestamps.csv" in tar_names:
            timestamp_path = tar.extractfile(
                tar_members[tar_names.index("timestamps.csv")]
            )
            source = looked_in[1]
            alternate_correct = True
        # an archive without a metadata.json is named after itself (load_metadata)
        metadata_path = (tar.extractfile(tar_members[tar_names.index("metadata.json")])
                         if "metadata.json" in tar_names else join(dirname, "metadata.json"))
    else:
        # Handling non-compressed session paths
        metadata_path = join(dirname, "metadata.json")
        depth_ts_path = join(dirname, "depth_ts.txt")
        alternate_timestamp_path = join(dirname, "timestamps.csv")
        looked_in = [depth_ts_path, alternate_timestamp_path]
        # Checks for alternative timestamp file if original .txt extension does not exist
        if exists(depth_ts_path):
            timestamp_path = depth_ts_path
        elif exists(alternate_timestamp_path):
            timestamp_path = alternate_timestamp_path
            alternate_correct = True
        elif input_file.endswith(".mkv"):
            from_depth_file = True

    if timestamp_path is None and not from_depth_file:
        raise FileNotFoundError(_no_timestamps_message(recording, looked_in, where=where))

    if not from_depth_file:
        if tar is None:
            source = timestamp_path
        timestamps = load_timestamps(timestamp_path, col=0, alternate=alternate_correct)
    else:
        # the container's frame times are in seconds; extraction writes milliseconds
        timestamps = np.asarray(load_timestamps_from_movie(input_file)) * 1000
        source = f"the frame timestamps inside {input_file}"
        looked_in = looked_in + [source]

    if len(timestamps) == 0:
        raise FileNotFoundError(
            _no_timestamps_message(recording, looked_in, empty=source, where=where)
        )

    acquisition_metadata = load_metadata(metadata_path)

    if with_source:
        return acquisition_metadata, timestamps, tar, source
    return acquisition_metadata, timestamps, tar


def recording_frame_rate(recording, timestamps, finfo):
    """
    A recording's frame rate, in frames per second: from its timestamps (in
    milliseconds), or the rate its video declares when the timestamps give none.

    Args:
    recording (str): the recording's path, for the error message.
    timestamps (numpy.array): the time of every frame, in milliseconds, or None.
    finfo (dict): the recording's video information (``get_movie_info``).

    Returns:
    fps (float): frames per second.
    """
    rate = rate_from_timestamps(timestamps, "milliseconds") if timestamps is not None else None
    if rate is None:
        rate = finfo.get("fps")
    if not rate or not np.isfinite(rate) or rate <= 0:
        first = ", ".join(f"{d:g}" for d in np.diff(np.asarray(timestamps if timestamps is not None else [],
                                                               dtype="float64"))[:5])
        raise ValueError(
            f"Cannot tell the frame rate of {recording}: its timestamps give no frame period "
            f"(the first differences are [{first}] milliseconds) and the video declares no frame "
            f"rate. Check that the recording's timestamp file (depth_ts.txt or timestamps.csv) "
            f"holds one increasing time per frame, then extract the recording again."
        )
    return float(rate)


# extract h5 helper function
def create_extract_h5(
    h5_file,
    acquisition_metadata,
    config_data,
    status_dict,
    scalars_attrs,
    nframes,
    roi,
    bground_im,
    first_frame,
    first_frame_idx,
    last_frame_idx,
    **kwargs,
):
    """
    write acquisition metadata, extraction metadata, computed scalars, timestamps, and original frames/frames_mask to extracted h5.

    Args:
    h5_file (h5py.File object): opened h5 file object to write to.
    acquisition_metadata (dict): Dictionary containing extracted session acquisition metadata.
    config_data (dict): dictionary object containing all required extraction parameters. (auto generated)
    status_dict (dict): dictionary that helps indicate if the session has been extracted fully.
    scalars_attrs (dict): dict of computed scalar attributes and descriptions to save.
    nframes (int): number of frames being recorded
    roi (np.ndarray): Computed 2D ROI Image.
    bground_im (np.ndarray): Computed 2D Background Image.
    first_frame (np.ndarray): Computed 2D First Frame Image.
    timestamps (numpy.array): Array of session timestamps.
    kwargs (dict): additional keyword arguments.

    """

    h5_file.create_dataset("metadata/uuid", data=status_dict["uuid"])
    h5_file.create_dataset(FRAME_RATE_PATH, data=config_data["fps"])
    h5_file[FRAME_RATE_PATH].attrs["description"] = (
        "Frame rate of the recording in frames per second, from its timestamps or the video's own rate"
    )

    # Creating scalar dataset
    for scalar in list(scalars_attrs.keys()):
        h5_file.create_dataset(
            f"scalars/{scalar}", (nframes,), "float32", compression="gzip"
        )
        h5_file[f"scalars/{scalar}"].attrs["description"] = scalars_attrs[scalar]

    # Timestamps
    if config_data.get("timestamps") is not None:
        h5_file.create_dataset(
            "timestamps",
            compression="gzip",
            data=config_data["timestamps"][first_frame_idx:last_frame_idx],
        )
        h5_file["timestamps"].attrs["description"] = "Depth video timestamps"
        h5_file["timestamps"].attrs["unit"] = TIMESTAMP_UNIT

    # Cropped Frames
    h5_file.create_dataset(
        "frames",
        (nframes, config_data["crop_size"][0], config_data["crop_size"][1]),
        config_data["frame_dtype"],
        compression="gzip",
    )
    h5_file["frames"].attrs["description"] = (
        "3D Numpy array of depth frames (nframes x w x h)." + " Depth values are in mm."
    )
    # Frame Masks for EM Tracking
    if config_data["use_tracking_model"]:
        h5_file.create_dataset(
            "frames_mask",
            (nframes, config_data["crop_size"][0], config_data["crop_size"][1]),
            "float32",
            compression="gzip",
        )
        h5_file["frames_mask"].attrs[
            "description"
        ] = "Log-likelihood values from the tracking model (nframes x w x h)"
    else:
        h5_file.create_dataset(
            "frames_mask",
            (nframes, config_data["crop_size"][0], config_data["crop_size"][1]),
            "bool",
            compression="gzip",
        )
        h5_file["frames_mask"].attrs[
            "description"
        ] = "Boolean mask, false=not mouse, true=mouse"

    # Flip Classifier
    if config_data["flip_classifier"] is not None:
        h5_file.create_dataset(
            "metadata/extraction/flips", (nframes,), "bool", compression="gzip"
        )
        h5_file["metadata/extraction/flips"].attrs[
            "description"
        ] = "Output from flip classifier, false=no flip, true=flip"

    # True Depth
    h5_file.create_dataset(
        "metadata/extraction/true_depth", data=config_data["true_depth"]
    )
    h5_file["metadata/extraction/true_depth"].attrs[
        "description"
    ] = "Detected true depth of arena floor in mm"

    # ROI
    h5_file.create_dataset("metadata/extraction/roi", data=roi, compression="gzip")
    h5_file["metadata/extraction/roi"].attrs["description"] = "ROI mask"

    # First Frame
    h5_file.create_dataset(
        "metadata/extraction/first_frame", data=first_frame[0], compression="gzip"
    )
    h5_file["metadata/extraction/first_frame"].attrs[
        "description"
    ] = "First frame of depth dataset"

    # First Frame index
    h5_file.create_dataset(
        "metadata/extraction/first_frame_idx",
        data=[first_frame_idx],
        compression="gzip",
    )
    h5_file["metadata/extraction/first_frame_idx"].attrs[
        "description"
    ] = "First frame index of this dataset"

    # Last Frame index
    h5_file.create_dataset(
        "metadata/extraction/last_frame_idx", data=[last_frame_idx], compression="gzip"
    )
    h5_file["metadata/extraction/last_frame_idx"].attrs[
        "description"
    ] = "Last frame index of this dataset"

    # Background
    h5_file.create_dataset(
        "metadata/extraction/background", data=bground_im, compression="gzip"
    )
    h5_file["metadata/extraction/background"].attrs[
        "description"
    ] = "Computed background image"

    # Extract Version
    extract_version = np.bytes_(version("moseq2-extract"))
    h5_file.create_dataset("metadata/extraction/extract_version", data=extract_version)
    h5_file["metadata/extraction/extract_version"].attrs[
        "description"
    ] = "Version of moseq2-extract"

    # Extraction Parameters
    from moseq2_extract.cli import extract

    dict_to_h5(
        h5_file,
        status_dict["parameters"],
        "metadata/extraction/parameters",
        click_param_annot(extract),
    )

    # Acquisition Metadata
    for key, value in acquisition_metadata.items():
        if type(value) is list and len(value) > 0 and type(value[0]) is str:
            value = [n.encode("utf8") for n in value]

        if value is not None:
            h5_file.create_dataset(f"metadata/acquisition/{key}", data=value)
        else:
            h5_file.create_dataset(f"metadata/acquisition/{key}", dtype="f")
