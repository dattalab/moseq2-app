"""
Contains helper functions for handling/storing data during extraction.
"""

import os
import h5py
import shutil
import tarfile
import warnings
import numpy as np
from moseq2_extract._yaml import write_yaml
from tqdm.auto import tqdm
from cytoolz import keymap
from importlib.metadata import version
from moseq2_extract.io.video import load_timestamps_from_movie
from os.path import exists, join, dirname, basename, splitext
from moseq2_extract.util import (
    h5_to_dict,
    load_timestamps,
    load_metadata,
    read_yaml,
    camel_to_snake,
    load_textdata,
    build_path,
    dict_to_h5,
    click_param_annot,
)


def check_completion_status(status_filename):
    """
    Read a results_00.yaml (status file) and checks whether the session has been
    fully extracted.

    Args:
    status_filename (str): path to results_00.yaml

    Returns:
    complete (bool): If True, data has been extracted to completion.
    """

    if exists(status_filename):
        return read_yaml(status_filename)["complete"]
    return False


def build_index_dict(files_to_use):
    """
    Create a dictionary for the index file from a list of files and respective metadatas.

    Args:
    files_to_use (list): list of paths to extracted h5 files.

    Returns:
    output_dict (dict): index-file dictionary containing all aggregated extractions.
    """

    output_dict = {"files": [], "pca_path": ""}

    index_uuids = []
    for i, file_tup in enumerate(files_to_use):
        if file_tup[2]["uuid"] not in index_uuids:
            tmp = {
                "path": (file_tup[0], file_tup[1]),
                "uuid": file_tup[2]["uuid"],
                "group": "default",
                "metadata": {
                    "SessionName": f"default_{i}",
                    "SubjectName": f"default_{i}",
                },  # fallback metadata
            }

            # handling metadata sub-dictionary values
            if "metadata" in file_tup[2]:
                tmp["metadata"].update(file_tup[2]["metadata"])
            else:
                warnings.warn(
                    f"Could not locate metadata for {file_tup[0]}! File will be listed with minimal default metadata."
                )

            index_uuids.append(file_tup[2]["uuid"])
            # appending file with default information
            output_dict["files"].append(tmp)

    return output_dict


def load_extraction_meta_from_h5s(to_load, snake_case=True):
    """
    Load extraction metadata from h5 files.

    Args:
    to_load (list): list of paths to h5 files.
    snake_case (bool): whether to save the files using snake_case

    Returns:
    loaded (list): list of loaded h5 dicts.
    """

    loaded = []
    for _dict, _h5f in tqdm(to_load, desc="Scanning data"):
        try:
            # v0.1.3 introduced a change - acq. metadata now here
            tmp = h5_to_dict(_h5f, "/metadata/acquisition")
        except KeyError:
            # if it doesn't exist it's likely from an older moseq version. Try loading it here
            try:
                tmp = h5_to_dict(_h5f, "/metadata/extraction")
            except KeyError:
                # if all else fails, abandon all hope
                tmp = {}

        # note that everything going into here must be a string (no bytes!)
        tmp = {k: str(v) for k, v in tmp.items()}
        if snake_case:
            tmp = keymap(camel_to_snake, tmp)

        # Specific use case block: Behavior reinforcement experiments
        feedback_file = join(dirname(_h5f), "..", "feedback_ts.txt")
        if exists(feedback_file):
            timestamps = map(int, load_timestamps(feedback_file, 0))
            feedback_status = map(int, load_timestamps(feedback_file, 1))
            _dict["feedback_timestamps"] = list(zip(timestamps, feedback_status))

        _dict["extraction_metadata"] = tmp
        loaded += [(_dict, _h5f)]

    return loaded


def build_manifest(loaded, format, snake_case=True):
    """
    Build a manifest file used to contain extraction result metadata from h5 and yaml files.

    Args:
    loaded (list of dicts): list of dicts containing loaded h5 data.
    format (str): filename format indicating the new name for the metadata files in the aggregate_results dir.
    snake_case (bool): whether to save the files using snake_case

    Returns:
    manifest (dict): dictionary of extraction metadata.
    """

    manifest = {}
    fallback = "session_{:03d}"
    fallback_count = 0

    # Additional metadata for certain use cases
    additional_meta = []

    # Behavior reinforcement metadata
    additional_meta.append(
        {
            "filename": "feedback_ts.txt",
            "var_name": "realtime_feedback",
            "dtype": np.bool,
        }
    )

    # Pre-trained model real-time syllable classification results
    additional_meta.append(
        {
            "filename": "predictions.txt",
            "var_name": "realtime_predictions",
            "dtype": np.int,
        }
    )

    # Real-Time Recorded/Computed PC Scores
    additional_meta.append(
        {
            "filename": "pc_scores.txt",
            "var_name": "realtime_pc_scores",
            "dtype": np.float32,
        }
    )

    for _dict, _h5f in loaded:
        print_format = f"{format}_{splitext(basename(_h5f))[0]}"
        if not _dict["extraction_metadata"]:
            copy_path = fallback.format(fallback_count)
            fallback_count += 1
        else:
            try:
                copy_path = build_path(
                    _dict["extraction_metadata"], print_format, snake_case=snake_case
                )
            except:
                copy_path = fallback.format(fallback_count)
                fallback_count += 1
                pass

        # add a bonus dictionary here to be copied to h5 file itself
        manifest[_h5f] = {
            "copy_path": copy_path,
            "yaml_dict": _dict,
            "additional_metadata": {},
        }
        for meta in additional_meta:
            filename = join(dirname(_h5f), "..", meta["filename"])
            if exists(filename):
                try:
                    data, timestamps = load_textdata(filename, dtype=meta["dtype"])
                    manifest[_h5f]["additional_metadata"][meta["var_name"]] = {
                        "data": data,
                        "timestamps": timestamps,
                    }
                except:
                    warnings.warn(
                        "WARNING: Did not load timestamps! This may cause issues if total dropped frames > 2% of the session."
                    )

    return manifest


def copy_manifest_results(manifest, output_dir):
    """
    Copy all consolidated manifest results to their respective output files.

    Args:
    manifest (dict): manifest dictionary containing all extraction h5 metadata to save
    output_dir (str): path to directory where extraction results will be aggregated.

    """

    if not exists(output_dir):
        os.makedirs(output_dir)

    # now the key is the source h5 file and the value is the path to copy to
    for k, v in tqdm(manifest.items(), desc="Copying files"):

        if exists(join(output_dir, f'{v["copy_path"]}.h5')):
            continue

        in_basename = splitext(basename(k))[0]
        in_dirname = dirname(k)

        h5_path = k
        mp4_path = join(in_dirname, f"{in_basename}.mp4")

        if exists(h5_path):
            new_h5_path = join(output_dir, f'{v["copy_path"]}.h5')
            shutil.copyfile(h5_path, new_h5_path)

        # if we have additional_meta then crack open the h5py and write to a safe place
        if len(v["additional_metadata"]) > 0:
            for k2, v2 in v["additional_metadata"].items():
                new_key = f"/metadata/misc/{k2}"
                with h5py.File(new_h5_path, "a") as f:
                    f.create_dataset(f"{new_key}/data", data=v2["data"])
                    f.create_dataset(f"{new_key}/timestamps", data=v2["timestamps"])

        if exists(mp4_path):
            shutil.copyfile(mp4_path, join(output_dir, f'{v["copy_path"]}.mp4'))

        v["yaml_dict"].pop("extraction_metadata", None)
        with open(f'{join(output_dir, v["copy_path"])}.yaml', "w") as f:
            write_yaml(v["yaml_dict"], f)


def handle_extract_metadata(input_file, dirname):
    """
    Extract metadata and timestamp in the extraction.

    Args:
    input_file (str): path to input file to extract
    dirname (str): path to directory where extraction files reside.

    Returns:
    acquisition_metadata (dict): key-value pairs of JSON contents
    timestamps (1D array): list of loaded timestamps
    tar (bool): indicator for whether the file is compressed.
    """

    tar = None
    tar_members = None
    alternate_correct = False
    from_depth_file = False

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
        metadata_path = tar.extractfile(tar_members[tar_names.index("metadata.json")])
        if "depth_ts.txt" in tar_names:
            timestamp_path = tar.extractfile(
                tar_members[tar_names.index("depth_ts.txt")]
            )
        elif "timestamps.csv" in tar_names:
            timestamp_path = tar.extractfile(
                tar_members[tar_names.index("timestamps.csv")]
            )
            alternate_correct = True
    else:
        # Handling non-compressed session paths
        metadata_path = join(dirname, "metadata.json")
        timestamp_path = join(dirname, "depth_ts.txt")
        alternate_timestamp_path = join(dirname, "timestamps.csv")
        # Checks for alternative timestamp file if original .txt extension does not exist
        if not exists(timestamp_path) and exists(alternate_timestamp_path):
            timestamp_path = alternate_timestamp_path
            alternate_correct = True
        elif not (
            exists(timestamp_path) or exists(alternate_timestamp_path)
        ) and input_file.endswith(".mkv"):
            from_depth_file = True

    acquisition_metadata = load_metadata(metadata_path)
    if not from_depth_file:
        timestamps = load_timestamps(timestamp_path, col=0, alternate=alternate_correct)
    else:
        timestamps = load_timestamps_from_movie(input_file)

    return acquisition_metadata, timestamps, tar


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
    extract_version = np.string_(version("moseq2-extract"))
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
