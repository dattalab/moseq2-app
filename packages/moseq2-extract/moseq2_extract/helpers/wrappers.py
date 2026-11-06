"""
Wrapper functions for data processing in extraction.
"""

import os
import sys
import math
from uuid import uuid4
import h5py
import warnings
import numpy as np
import urllib.request
from moseq2_common.yaml import read_yaml, write_config
from tqdm.auto import tqdm
from moseq2_extract.io.image import write_image
from moseq2_extract.helpers.extract import process_extract_batches
from moseq2_extract.extract.proc import get_roi, get_bground_im_file
from os.path import join, exists, dirname, basename, abspath, splitext
from moseq2_extract.io.video import load_movie_data, get_movie_info, write_frames
from moseq2_extract.extract.flip_classifier import load_flip_classifier, training_settings
from moseq2_extract.util import mouse_threshold_filter, filter_warnings
from moseq2_extract.helpers.data import (
    recording_frame_rate,
    handle_extract_metadata,
    create_extract_h5,
    write_misc_metadata,
)
from moseq2_extract.recordings import find_recordings, outputs_in, project_keys, session_outputs
from moseq2_common.project import session_key
from dataclasses import fields
from moseq2_common.frames import CleaningOptions, select_strel
from moseq2_common.records import atomic_output, start_record, write_h5_record
from moseq2_extract.util import (
    gen_batch_sequence,
    scalar_attributes,
    convert_raw_to_avi_function,
    set_bground_to_plane_fit,
    graduate_dilated_wall_area,
    get_bucket_center,
    _floor_depth,
    camera_for,
    detect_and_set_camera_parameters,
    get_frame_range_indices,
    check_filter_sizes,
)


def aggregate_extract_results_wrapper(input_dir, output_dir, mouse_threshold=0.0, project=None):
    """
    The finished extractions of the recordings under ``input_dir``, each in its session's folder.

    Nothing is copied: extraction writes each session's files into its folder. A recording counts
    when its ``results.h5`` exists (``moseq2_extract.recordings.session_outputs``) and its frames'
    mean depth is above ``mouse_threshold``.

    Args:
    input_dir (str): the folder searched for recordings.
    output_dir (str): the folder holding the session folders; None for each session's folder in the
        project (the one moseq2-project.yaml names, else in ``aggregate_results/``).
    mouse_threshold (float): the mean frame depth a session's extraction must exceed to be included.
    project (Project): the project.

    Returns:
    found (list): (session key, its folder, its uuid) for each session included, sorted by key.
    """
    recordings = find_recordings(input_dir).recordings
    keys = sorted(session_key(project, recording) for recording in recordings)
    everything = set(keys) | set(project_keys(project))
    found = []
    for key in keys:
        outputs = session_outputs(project, key, keys=everything, sessions_dir=output_dir)
        if not outputs.h5.is_file():
            continue
        if not mouse_threshold_filter(str(outputs.h5), thresh=mouse_threshold):
            print(f"Leaving out {key}: the mean depth of its first frames in {outputs.h5} is not above "
                  f"mouse_threshold ({mouse_threshold}), so it is taken to hold no mouse. Set a lower "
                  "mouse_threshold in config.yaml's extract section to include it.")
            continue
        with h5py.File(outputs.h5, "r") as f:
            uuid_value = f["metadata/uuid"][()]
        found.append((key, outputs.folder, uuid_value.decode() if isinstance(uuid_value, bytes) else str(uuid_value)))
    return found


def find_arena_candidates(bground_im, config_data, output_dir=None):
    """
    Find a recording's candidate arena masks, in the order ``bg_roi_index`` counts them.

    The arena tool and extraction both call this, so a recording's saved parameters give
    the arena the tool showed.

    Args:
    bground_im (numpy.ndarray): the recording's background.
    config_data (dict): the recording's parameters.
    output_dir (str): where a plane-fit background is saved, as ``plane_bground.tiff``, when
        ``use_plane_bground`` is set.

    Returns:
    rois (list): candidate arena masks.
    bground_im (numpy.ndarray): the background extraction subtracts, which is the plane
        fit when ``use_plane_bground`` is set.
    """
    strel_dilate = select_strel(
        config_data["bg_roi_shape"], tuple(config_data["bg_roi_dilate"])
    )
    strel_erode = select_strel(
        config_data["bg_roi_shape"], tuple(config_data["bg_roi_erode"])
    )

    rois, plane = get_roi(
        bground_im,
        **config_data,
        strel_dilate=strel_dilate,
        strel_erode=strel_erode,
        get_all_data=False,
    )

    if config_data["use_plane_bground"]:
        print("Using plane fit for background...")
        bground_im = set_bground_to_plane_fit(bground_im, plane, output_dir)

    # Keep the best-ranked ROIs and order them top to bottom
    if config_data["bg_sort_roi_by_position"]:
        rois = rois[: config_data["bg_sort_roi_by_position_max_rois"]]
        rois = [
            rois[i] for i in np.argsort([np.nonzero(roi)[0].mean() for roi in rois])
        ]

    return rois, bground_im


def get_roi_wrapper(input_file, config_data, output_dir=None):
    """
    Compute ROI given depth file.

    Args:
    input_file (str): path to depth file.
    config_data (dict): dictionary of ROI extraction parameters.
    output_dir (str): path to desired directory to save results in.

    Returns:
    roi (numpy.ndarray): ROI image to plot in GUI
    bground_im (numpy.ndarray): Background image to plot in GUI
    first_frame (numpy.ndarray): First frame image to plot in GUI
    """

    if output_dir is None:
        output_dir = join(dirname(input_file), "proc")
    elif exists(output_dir):
        pass
    elif dirname(output_dir) == "" or dirname(output_dir) not in input_file:
        output_dir = join(dirname(input_file), output_dir)

    os.makedirs(output_dir, exist_ok=True)
    config_data["output_dir"] = output_dir

    if config_data.get("finfo") is None:
        # when depth video is .avi, frame_size/dim is read directly from the video
        config_data["finfo"] = get_movie_info(input_file, **config_data)

    # checks camera type to set appropriate bg_roi_weights
    config_data = detect_and_set_camera_parameters(config_data, input_file)

    print("Getting background...")
    bground_im = get_bground_im_file(input_file, **config_data)
    write_image(join(output_dir, "bground.tiff"), bground_im, scale=True)

    # readjust depth range
    if not config_data.get("manual_set_depth_range", False):
        # search for depth values between the max distance and the halfway point to the camera
        print(
            "Automatically setting depth range. To manually set range values,"
            ' set "manual_set_depth_range" to True.\n For CLI users: use the --manual-set-depth-range flag.'
        )
        print(
            "To manually set a correct --bg-roi-depth-range value, "
            "set the min and max range values to +/-50mm of the actual camera height."
        )

        cX, cY = get_bucket_center(
            bground_im, bground_im.max(), threshold=int(np.median(bground_im) / 2)
        )
        adjusted_bg_depth_range = _floor_depth(bground_im, cX, cY)
        config_data["bg_roi_depth_range"] = [
            int(adjusted_bg_depth_range - 50),
            int(adjusted_bg_depth_range + 50),
        ]

    # pass in config_data['finfo']['dims'] for frame size otherwise frame size is hard coded to 512x424
    first_frame = load_movie_data(
        input_file, 0, frame_size=config_data["finfo"]["dims"], **config_data
    )  # there is a tar object flag that must be set!!
    write_image(
        join(output_dir, "first_frame.tiff"),
        first_frame,
        scale=True,
        scale_factor=config_data["bg_roi_depth_range"],
    )

    print("Getting roi...")
    rois, bground_im = find_arena_candidates(bground_im, config_data, output_dir)

    if type(config_data["bg_roi_index"]) == int:
        config_data["bg_roi_index"] = [config_data["bg_roi_index"]]

    bg_roi_index = [
        idx for idx in config_data["bg_roi_index"] if idx in range(len(rois))
    ]
    roi = rois[bg_roi_index[0]]

    for idx in bg_roi_index:
        roi_filename = f"roi_{idx:02d}.tiff"
        write_image(join(output_dir, roi_filename), rois[idx], scale=True)

    return roi, bground_im, first_frame


class NoFlipClassifierWarning(Warning):
    """
    A recording is extracted without a flip classifier. A subclass of Warning
    rather than UserWarning, so the notebook's filter_warnings does not hide it.
    """


def warn_without_flip_classifier(input_file, config_data):
    """
    Warn, once per recording, that a recording is extracted without a flip
    classifier, so the mouse's heading is known only up to half a turn.

    Args:
    input_file (str): path to the recording being extracted.
    config_data (dict): extraction configuration; its 'flip_classifier' is checked.
    """
    if config_data.get("flip_classifier"):
        return
    warnings.warn(
        f"Extracting {input_file} without a flip classifier (flip_classifier is "
        f"{config_data.get('flip_classifier')!r}). The mouse's heading is then "
        "resolved only up to half a turn for the whole recording: extraction keeps "
        "it continuous from frame to frame, but nothing decides which end of the "
        "mouse is its head, so every frame of the recording may face the wrong way. "
        "To orient the mouse, get a flip classifier that matches your camera and "
        "mice, either a pre-trained one with `moseq2-extract download-flip-file "
        "config.yaml` (which writes its path into config.yaml) or one trained in the "
        "Flip-Classifier-Training-Notebook, and pass it as flip_classifier in the "
        "config file or with --flip-classifier.",
        NoFlipClassifierWarning,
        stacklevel=2,
    )


#: The cleaning settings a trained flip classifier is checked against.
CLEANING_FIELDS = tuple(f.name for f in fields(CleaningOptions))


def _shown(value):
    """A setting's value as config.yaml writes it."""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_shown(v) for v in value) + "]"
    return str(value)


def _cleaning(values):
    """``values``' cleaning settings, as ``CleaningOptions`` reads them, with lists for tuples."""
    opts = CleaningOptions.resolve(config={k: v for k, v in values.items() if k in CLEANING_FIELDS})
    return {k: list(v) if isinstance(v, tuple) else v for k, v in opts.as_dict().items()}


def _check_trained_cleaning(flip_classifier, trained, config_data):
    """Stop when extraction cleans frames otherwise than the classifier's training did (``trained``, a list)."""
    now = _cleaning(config_data)
    differences = []
    for stored in trained:
        then = _cleaning(stored)
        differ = [name for name in CLEANING_FIELDS if then[name] != now[name]]
        if not differ:
            return
        differences.append((differ, then))
    differ, then = min(differences, key=lambda item: len(item[0]))
    listed = "; ".join(f"{name} {_shown(then[name])} in training, {_shown(now[name])} in this extraction"
                       for name in differ)
    raise ValueError(
        f"The flip classifier {flip_classifier} was trained on frames cleaned with other settings than this "
        f"extraction uses: {listed}. A classifier sees frames as they were cleaned in training, so it would "
        f"orient these mice less reliably. Set {', '.join(differ)} back to the training values in config.yaml's "
        "extract section (or in the recording's arena-tool settings), or train the classifier again in the "
        "Flip-Classifier-Training notebook with the current settings."
    )


def validate_flip_classifier(config_data):
    """
    Check the flip classifier before extraction: it exists, it can be read, and it was trained for this crop size
    and, when it says so, these cleaning settings.

    The crop size is checked against the classifier's feature count (the published classifiers) or the crop size
    it was trained on (``moseq2_extract.extract.flip_classifier.training_settings``, carried by a classifier the
    Flip-Classifier-Training notebook trained, with its cleaning settings). Each problem raises before any work,
    naming the classifier and both values.

    Args:
    config_data (dict): extraction configuration containing 'flip_classifier', 'crop_size' and the cleaning settings
    """
    flip_classifier = config_data.get("flip_classifier")
    if not flip_classifier:
        return

    if not exists(flip_classifier):
        raise FileNotFoundError(
            f"The flip classifier {flip_classifier} (flip_classifier in config.yaml's extract section, or "
            "--flip-classifier) does not exist. Correct the path; download a classifier with `moseq2-extract "
            "download-flip-file config.yaml`, which writes its path into config.yaml; or set flip_classifier to "
            "null to extract without one."
        )

    crop_size = tuple(config_data.get("crop_size", (80, 80)))

    try:
        clf = load_flip_classifier(flip_classifier)
    except Exception as error:
        raise ValueError(
            f"The flip classifier {flip_classifier} could not be read ({type(error).__name__}: {error}). Check "
            "that flip_classifier names a classifier file: one downloaded with `moseq2-extract download-flip-file "
            "config.yaml` (an .npz), or one saved by the Flip-Classifier-Training notebook."
        ) from None

    trained = training_settings(clf) or {}
    n_features = getattr(clf, "n_features_", None)
    if n_features is None and trained.get("crop_size") is not None:
        trained_crop = tuple(trained["crop_size"])
        if trained_crop != crop_size:
            raise ValueError(
                f"The flip classifier {flip_classifier} was trained on frames cropped to {trained_crop}, but "
                f"crop_size is {crop_size}. Set crop_size to {trained_crop} in config.yaml (in the arena tool, its "
                f"crop size box), or train the classifier again on extractions made with crop_size {crop_size}."
            )
    elif n_features is not None and crop_size[0] * crop_size[1] != n_features:
        actual_features = crop_size[0] * crop_size[1]
        expected_crop = int(math.sqrt(n_features))
        raise ValueError(
            f"\nFlip classifier / crop size mismatch!\n\n"
            f"  Your crop_size is {crop_size} ({actual_features} pixels per frame),\n"
            f"  but the flip classifier at\n"
            f"    '{flip_classifier}'\n"
            f"  expects {n_features} pixels (i.e. crop_size ({expected_crop}, {expected_crop})).\n\n"
            f"  This typically happens when the flip classifier and camera type don't match.\n"
            f"  The pre-trained flip classifiers are camera-specific:\n"
            f"    - Kinect v2 (K2) classifiers expect crop_size (80, 80)\n"
            f"    - Azure / Orbbec classifiers expect crop_size (120, 120)\n"
            f"  If you generated your config with one camera type but downloaded\n"
            f"  the flip classifier for a different camera, they won't be compatible.\n\n"
            f"  To fix this, either:\n"
            f"  1. Change crop_size in your config.yaml (in the arena tool, its crop size box)\n"
            f"     to ({expected_crop}, {expected_crop}) to match your current flip classifier.\n"
            f"  2. Download a flip classifier that matches your current crop_size by\n"
            f"     re-running the download step with the correct selection:\n"
            f"       CLI:      moseq2-extract download-flip-file config.yaml\n"
            f"       Notebook: change the 'selection' parameter in the download cell\n"
            f"                 (selection=1 for K2, selection=3 for Azure/Orbbec)"
        )

    if trained.get("cleaning"):
        _check_trained_cleaning(flip_classifier, trained["cleaning"], config_data)


def floor_depth_and_background(bground_im, roi, config_data, output_dir):
    """
    Measure the arena floor's depth and build the background frames are subtracted from.

    Extraction and the arena tool's preview both call this before ``extract_chunk``, so
    the preview subtracts what extraction subtracts.

    Args:
    bground_im (numpy.ndarray): the recording's background.
    roi (numpy.ndarray): the arena mask.
    config_data (dict): the recording's settings; ``true_depth`` is set in it.
    output_dir (str): the recording's output folder, where a graduated background is written.

    Returns:
    bground_im (numpy.ndarray): the background, with graduated walls when
        ``graduate_walls`` is set and ``dilate_iterations`` is more than 1.
    """
    # Debugging option: DTD has no effect on extraction results unless dilate iterations > 1
    if config_data.get("detected_true_depth", "auto") == "auto":
        config_data["true_depth"] = np.median(bground_im[roi > 0])
    else:
        config_data["true_depth"] = int(config_data["detected_true_depth"])

    print("Detected true depth:", config_data["true_depth"])

    if config_data.get("dilate_iterations", 0) > 1 and config_data.get(
        "graduate_walls"
    ):
        print("Dilating background and graduating walls")
        bground_im = graduate_dilated_wall_area(
            bground_im,
            config_data,
            select_strel(config_data["bg_roi_shape"], tuple(config_data["bg_roi_dilate"])),
            output_dir,
        )
    return bground_im


def extract_wrapper(input_file, output_dir, config_data, num_frames=None, skip=False, uuid=None, record=None):
    """
    Extract depth videos.

    The session's files go into ``output_dir`` under fixed names (``results.h5``, ``results.mp4`` and
    the images), whichever arena mask is used, so extracting again replaces them. The ``.h5`` holds the
    extraction's record (``moseq2_common.records``) as its ``/provenance`` attribute: the options after
    detection (the camera, the depth range, the floor depth), the recording and the files read beside it.
    The ``.h5`` and the movie are written under temporary names and renamed when the extraction is
    complete, so a recording is extracted when its ``results.h5`` exists, and an extraction stopped
    partway leaves the previous one as it was.

    Args:
    input_file (str): path to depth file
    output_dir (str): the folder the session's files go in, relative to the recording's folder; made when missing.
    config_data (dict): dictionary containing extraction parameters.
    num_frames (int): number of frames to extract.
    skip (bool): indicates whether to skip file if already extracted
    uuid (str): the session's uuid, when it has one already; None draws a new one.
    record (moseq2_common.records.Record): the extraction step's record, started by the step; None starts
        one here with ``config_data`` as its options.

    Returns:
    output_dir (str): path to directory containing extraction, or None if the file was skipped as already extracted
    """
    print("Processing:", input_file)

    # for a tarball, input_file becomes the opened archive further down
    recording = input_file
    outputs = outputs_in(join(dirname(abspath(recording)), output_dir))
    output_dir = str(outputs.folder)

    roi_index = config_data["bg_roi_index"]
    if isinstance(roi_index, list):
        roi_index = roi_index[0]
    results_filename = str(outputs.h5)

    # A recording already extracted is skipped before anything about it is
    # checked: the checks below guard an extraction that is about to run.
    if skip and exists(results_filename):
        print("Skipping...")
        return

    validate_flip_classifier(config_data)
    warn_without_flip_classifier(input_file, config_data)

    # get the basic metadata

    # ensure 'get_cmd' and 'run_cmd' are not in config_data or get_bground_im_file will fail
    config_data = {
        k: v
        for k, v in config_data.items()
        if k not in ("get_cmd", "run_cmd", "extensions")
    }

    uuid = str(uuid) if uuid else str(uuid4())
    if record is None:
        record = start_record("extract", config_data, [])

    # save input directory path
    in_dirname = dirname(input_file)

    # If input file is compressed (tarFile), returns decompressed file path and tar bool indicator.
    # Also gets loads respective metadata dictionary and timestamp array.
    acquisition_metadata, config_data["timestamps"], config_data["tar"], timestamp_source = (
        handle_extract_metadata(input_file, in_dirname, with_source=True)
    )

    # updating input_file reference to open tar file object if input file ends with [.tar/.tar.gz]
    if config_data["tar"] is not None:
        input_file = config_data["tar"]

    config_data["finfo"] = get_movie_info(input_file, **config_data)

    nframes_from_video = config_data["finfo"]["nframes"]
    n_timestamps = len(config_data["timestamps"]) if config_data["timestamps"] is not None else 0

    if nframes_from_video is None:
        # An .mkv stores a duration rather than a frame count, and an .avi can
        # lack one. The timestamps then give the count: an .mkv's own frame
        # times are the video's frames, a timestamp file's lines are not.
        if timestamp_source == f"the frame timestamps inside {recording}":
            print(
                f"{recording} records no frame count of its own; extracting the "
                f"{n_timestamps} frames counted from {timestamp_source}."
            )
        else:
            warnings.warn(
                f"{recording} records no frame count of its own, so extraction "
                f"takes the {n_timestamps} timestamps in {timestamp_source} as its "
                f"number of frames. That count cannot be checked against the "
                f"video. If {timestamp_source} is not this recording's own "
                f"timestamp file, or the extraction stops early or comes out "
                f"shorter than the recording, check that the file belongs to "
                f"{recording}."
            )
        config_data["finfo"]["nframes"] = n_timestamps
    elif n_timestamps > 0 and nframes_from_video != n_timestamps:
        raise ValueError(
            f"Frame count mismatch for {recording}: the depth video has "
            f"{nframes_from_video} frames but {n_timestamps} timestamps were read "
            f"from {timestamp_source}. Each frame needs one timestamp, so the video "
            f"or the timestamps are incomplete, or they do not belong together. "
            f"Check that {timestamp_source} is this recording's own timestamp file "
            f"and that neither was cut short when copied, then extract the "
            f"recording again."
        )

    config_data["fps"] = recording_frame_rate(recording, config_data["timestamps"], config_data["finfo"])

    describe_recording(record, recording, input_file, config_data, timestamp_source)

    # Getting number of frames to extract
    if num_frames is None:
        nframes = int(config_data["finfo"]["nframes"])
    elif num_frames > config_data["finfo"]["nframes"]:
        warnings.warn(
            "Requested more frames than video includes, extracting whole recording..."
        )
        nframes = int(config_data["finfo"]["nframes"])
    elif isinstance(num_frames, int):
        nframes = num_frames

    config_data = check_filter_sizes(config_data)

    # Compute total number of frames to include from an initial starting point.
    total_frames, first_frame_idx, last_frame_idx = get_frame_range_indices(
        *config_data["frame_trim"], nframes
    )

    scalars_attrs = scalar_attributes()
    scalars = list(scalars_attrs)

    # Get frame chunks to extract
    frame_batches = gen_batch_sequence(
        last_frame_idx,
        config_data["chunk_size"],
        config_data["chunk_overlap"],
        offset=first_frame_idx,
    )

    if not exists(output_dir):
        os.makedirs(output_dir)

    config_data["bg_roi_index"] = roi_index

    # Compute ROIs
    roi, bground_im, first_frame = get_roi_wrapper(
        input_file, config_data, output_dir=output_dir
    )

    bground_im = floor_depth_and_background(bground_im, roi, config_data, output_dir)

    extraction_data = {
        "bground_im": bground_im,
        "roi": roi,
        "first_frame": first_frame,
        "first_frame_idx": first_frame_idx,
        "last_frame_idx": last_frame_idx,
        "nframes": total_frames,
        "frame_batches": frame_batches,
    }

    record.ran_with({"bg_roi_depth_range": config_data["bg_roi_depth_range"],
                     "camera_type": camera_for(config_data, input_file) or config_data.get("camera_type"),
                     "true_depth": float(config_data["true_depth"]),
                     **{key: config_data[key] for key in ("bg_roi_weights", "pixel_format", "movie_dtype")
                        if key in config_data}})

    # farm out the batches and write the h5 file and the preview movie, renamed to their final names
    # when complete
    movie = outputs.folder / f".{outputs.movie.stem}.partial{outputs.movie.suffix}"
    with atomic_output(results_filename) as partial_h5, atomic_output(outputs.movie, movie) as partial_movie:
        with h5py.File(partial_h5, "w") as f:
            # Write scalars, roi, acquisition metadata, etc. to h5 file
            create_extract_h5(
                **extraction_data,
                h5_file=f,
                acquisition_metadata=acquisition_metadata,
                config_data=config_data,
                uuid=uuid,
                scalars_attrs=scalars_attrs,
            )

            # Write crop-rotated results to h5 file and write video preview mp4 file
            process_extract_batches(
                **extraction_data,
                h5_file=f,
                input_file=input_file,
                config_data=config_data,
                scalars=scalars,
                output_mov_path=str(partial_movie),
            )
            write_misc_metadata(f, dirname(recording))
            write_h5_record(f, record.finish({
                **{name: f[name] for name in ("frames", "frames_mask", "timestamps") if name in f},
                "files": {"h5": results_filename, "movie": str(outputs.movie),
                          "images": sorted(str(p) for p in outputs.folder.glob("*.tiff"))},
            }))
    # the arena masks find-roi wrote here, with its records beside them, are now the extraction's
    for stale in outputs.folder.glob("roi_*.yaml"):
        stale.unlink()

    print()

    # Compress the depth file to avi format; compresses original raw file by ~8x.
    try:
        if config_data["tar"] is None:
            if input_file.endswith("dat") and config_data["compress"]:
                convert_raw_to_avi_function(
                    input_file,
                    chunk_size=config_data["compress_chunk_size"],
                    fps=config_data["fps"],
                    delete=False,  # to be changed when we're ready!
                    threads=config_data["compress_threads"],
                )
    except AttributeError as e:
        print("Error converting raw video to avi format, continuing anyway...")
        print(e)

    return output_dir


def describe_recording(record, recording, input_file, config_data, timestamp_source):
    """Add the recording and the files read beside it to the extraction's ``record``, with what describes
    the recording's data, and note a missing ``metadata.json``."""
    finfo = config_data.get("finfo") or {}
    facts = {"path": recording,
             "frame_count": finfo.get("nframes"),
             "resolution": list(finfo["dims"]) if finfo.get("dims") is not None else None,
             "frame_rate": config_data.get("fps", finfo.get("fps")),
             "camera": camera_for(config_data, input_file)}
    inputs = [facts]
    if isinstance(timestamp_source, str) and exists(timestamp_source):
        timestamps = config_data.get("timestamps")
        inputs.append({"path": timestamp_source,
                       "timestamp_count": len(timestamps) if timestamps is not None else 0})
    folder = dirname(abspath(recording))
    metadata_file = join(folder, "metadata.json")
    if exists(metadata_file):
        inputs.append(metadata_file)
    elif config_data.get("tar") is None:
        record.note("metadata_json", f"{folder} has no metadata.json: SessionName is the folder's name "
                                     "and SubjectName is empty")
    if config_data.get("flip_classifier"):
        inputs.append(config_data["flip_classifier"])
    record.add_inputs(inputs)


@filter_warnings
def flip_file_wrapper(config_file, output_dir, selected_flip=None):
    """
    Download and save flip classifiers.

    Args:
    config_file (str): path to config file
    output_dir (str): path to directory to save classifier in.
    selected_flip (int or str): int: index of desired flip classifier; str: path to flip file

    Returns:
    output_filename (str): path to the downloaded classifier.
    """

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
    flip_files = {
        "large mice with fibers (K2)": "https://moseq-data.s3.amazonaws.com/flip-classifiers/flip_classifier_k2_largemicewithfiber.npz",
        "adult male c57s (K2)": "https://moseq-data.s3.amazonaws.com/flip-classifiers/flip_classifier_k2_c57_10to13weeks.npz",
        "mice with Inscopix cables (K2)": "https://moseq-data.s3.amazonaws.com/flip-classifiers/flip_classifier_k2_inscopix.npz",
        "adult male c57s (Azure)": "https://moseq-data.s3.amazonaws.com/flip-classifiers/flip_classifier_azure_temp.npz",
    }

    key_list = list(flip_files)

    if selected_flip is None:
        for idx, (k, v) in enumerate(flip_files.items()):
            print(f"[{idx}] {k} ---> {v}")
    else:
        if selected_flip < 0:
            raise IndexError("list index out of range")
        selected_flip = key_list[selected_flip]

    # prompt for user selection if not already inputted
    while selected_flip is None:
        try:
            choice = int(input("Enter a selection "))
            if choice < 0:
                raise IndexError
            selected_flip = key_list[choice]
        except (ValueError, IndexError):
            print("Please enter a valid number listed above")
            continue

    if not exists(output_dir):
        os.makedirs(output_dir)

    selection = flip_files[selected_flip]

    output_filename = join(output_dir, basename(selection))

    urllib.request.urlretrieve(selection, output_filename)
    print("Successfully downloaded flip file to", output_filename)

    # Update the config file with the latest path to the flip classifier
    try:
        config_data = read_yaml(config_file)
        config_data.setdefault("extract", {})["flip_classifier"] = output_filename

        write_config(config_data, config_file)
    except Exception as e:
        print("Could not update configuration file flip classifier path")
        print("Unexpected error:", e)

    return output_filename


def _copy_frames(
    input_file,
    output_file,
    frame_batches,
    fps,
    delete,
    threads,
    mapping,
    avi_encode=True,
):
    """
    Copy the frames of an input depth recording into an output file, batch by batch,
    check that the output reads back identical to the input, and optionally delete the
    input.

    Args:
    input_file (str): Path to depth file to read frames from
    output_file (str): Path to output file
    frame_batches (list): Frame index batches to copy, in order
    fps (int): frame rate.
    delete (bool): Delete the original depth file if True.
    threads (int): Number of threads used to encode video.
    mapping (str or int): Indicate which video stream to from the inputted file
    avi_encode (bool): Encode the frames as video if True, else append them as raw
        uint16 bytes

    Returns:
    """

    video_pipe = None

    for batch in tqdm(frame_batches, desc="Encoding batches"):
        frames = load_movie_data(input_file, batch, mapping=mapping)
        if avi_encode:
            video_pipe = write_frames(
                output_file,
                frames,
                pipe=video_pipe,
                close_pipe=False,
                threads=threads,
                fps=fps,
            )
        else:
            with open(output_file, "ab") as f:
                f.write(frames.astype("uint16").tobytes())

    if video_pipe:
        video_pipe.communicate()

    for batch in tqdm(frame_batches, desc="Checking data integrity"):
        raw_frames = load_movie_data(input_file, batch, mapping=mapping)
        encoded_frames = load_movie_data(output_file, batch, mapping=mapping)

        if not np.array_equal(raw_frames, encoded_frames):
            raise RuntimeError(
                f"Raw frames and encoded frames not equal from {batch[0]} to {batch[-1]}"
            )

    print("Encoding successful")

    if delete:
        print("Deleting", input_file)
        os.remove(input_file)


def convert_raw_to_avi_wrapper(
    input_file, output_file, chunk_size, fps, delete, threads, mapping
):
    """
    compress a raw depth file into an avi file (with depth values) that is 8x smaller.

    Args:
    input_file (str): Path to depth file to convert
    output_file (str): Path to output avi file
    chunk_size (int): Size of frame chunks to iteratively process
    fps (int): frame rate.
    delete (bool): Delete the original depth file if True.
    threads (int): Number of threads used to encode video.
    mapping (str or int): Indicate which video stream to from the inputted file

    Returns:
    """

    if output_file is None:
        base_filename = splitext(basename(input_file))[0]
        output_file = join(dirname(input_file), f"{base_filename}.avi")

    vid_info = get_movie_info(input_file, mapping=mapping)
    frame_batches = gen_batch_sequence(vid_info["nframes"], chunk_size, 0)

    _copy_frames(input_file, output_file, frame_batches, fps, delete, threads, mapping)


def copy_slice_wrapper(
    input_file, output_file, copy_slice, chunk_size, fps, delete, threads, mapping
):
    """
    Copy a segment of an input depth recording into a new video file.

    Args:
    input_file (str): Path to depth file to read segment from
    output_file (str): Path to outputted video file with copied slice.
    copy_slice (2-tuple): Frame range to copy from input file.
    chunk_size (int): Size of frame chunks to iteratively process
    fps (int): Frames per second.
    delete (bool): Delete the original depth file if True.
    threads (int): Number of threads used to encode video.
    mapping (str or int): Indicate which video stream to from the inputted file

    Returns:
    """

    if output_file is None:
        base_filename = splitext(basename(input_file))[0]
        avi_encode = True
        output_file = join(dirname(input_file), f"{base_filename}.avi")
    else:
        output_filename, ext = splitext(basename(output_file))
        if ext == ".avi":
            avi_encode = True
        else:
            avi_encode = False

    vid_info = get_movie_info(input_file)
    copy_slice = (copy_slice[0], np.minimum(copy_slice[1], vid_info["nframes"]))
    nframes = copy_slice[1] - copy_slice[0]
    offset = copy_slice[0]

    frame_batches = gen_batch_sequence(nframes, chunk_size, 0, offset)

    if exists(output_file):
        overwrite = input(
            "Press ENTER to overwrite your previous extraction, else to end the process."
        )
        if overwrite != "":
            sys.exit(0)

    _copy_frames(
        input_file,
        output_file,
        frame_batches,
        fps,
        delete,
        threads,
        mapping,
        avi_encode=avi_encode,
    )
