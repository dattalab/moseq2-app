"""Trimming frames off the start or end of a recording.

``--frame-trim A B`` extracts a recording without its first ``A`` and last
``B`` frames. The output holds only the kept frames, so frame ``A`` of the
recording is frame 0 of the extraction, and every per-frame dataset and the
timestamps line up on that numbering (moseq2-extract#157).

These run the real extraction on a copy of the suite's 600-frame
``saline_example_0`` recording, with its flip classifier, in chunks of 100
frames. Extraction resolves a mouse's heading within each chunk, starting from
the chunk's first frame, so two extractions that cut the recording at
different places can crop the same frame rotated half a turn apart. With
chunks of 100 and trims of 100, the trimmed extraction's chunks are the
untrimmed one's, and every kept frame has to come out identical.
"""

from __future__ import annotations

import shutil

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
from moseq2_extract.util import load_timestamps

RECORDING = "saline_example_0"
FLIP_CLASSIFIER = "flip_classifier_k2_c57_10to13weeks.npz"
NFRAMES = 600
CHUNK_SIZE = 100


@pytest.fixture(scope="module")
def recording(raw_sessions, tmp_path_factory):
    """A writable copy of the recording, so extraction writes beside it."""
    root = tmp_path_factory.mktemp("frame_trim")
    shutil.copytree(raw_sessions / RECORDING, root / RECORDING)
    shutil.copy(raw_sessions / FLIP_CLASSIFIER, root / FLIP_CLASSIFIER)
    return root / RECORDING


def extract(recording, start, end):
    """Extract with the given trim and return what the results file holds."""
    output_dir = f"proc_{start}_{end}"
    result = CliRunner().invoke(
        extract_cli.cli,
        [
            "extract",
            str(recording / "depth.avi"),
            "--frame-trim", str(start), str(end),
            "--chunk-size", str(CHUNK_SIZE),
            "--flip-classifier", str(recording.parent / FLIP_CLASSIFIER),
            "--output-dir", output_dir,
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    with h5py.File(recording / output_dir / "results_00.h5", "r") as f:
        return {
            "frames": f["frames"][()],
            "frames_mask": f["frames_mask"][()],
            "flips": f["metadata/extraction/flips"][()],
            "timestamps": f["timestamps"][()],
            "scalars": {k: f["scalars"][k][()] for k in f["scalars"]},
        }


@pytest.fixture(scope="module")
def untrimmed(recording):
    return extract(recording, 0, 0)


@pytest.fixture(scope="module", params=[(100, 0), (100, 100)], ids=lambda t: f"trim{t[0]}-{t[1]}")
def trimmed(request, recording):
    start, end = request.param
    return start, end, extract(recording, start, end)


def test_a_trimmed_extraction_holds_only_the_kept_frames(recording, untrimmed, trimmed):
    start, end, trimmed = trimmed
    kept = NFRAMES - start - end

    assert len(trimmed["frames"]) == kept
    assert len(trimmed["frames_mask"]) == kept
    assert len(trimmed["flips"]) == kept
    for name, values in trimmed["scalars"].items():
        assert len(values) == kept, name

    recorded = load_timestamps(str(recording / "depth_ts.txt"), col=0)
    assert len(untrimmed["timestamps"]) == NFRAMES
    np.testing.assert_array_equal(trimmed["timestamps"], recorded[start : NFRAMES - end])
    np.testing.assert_array_equal(trimmed["timestamps"], untrimmed["timestamps"][start : NFRAMES - end])


def test_each_kept_frame_is_the_same_frame_of_the_recording(untrimmed, trimmed):
    """Frame ``i`` of the trimmed extraction is frame ``start + i`` of the untrimmed one."""
    start, end, trimmed = trimmed
    kept = slice(start, NFRAMES - end)
    np.testing.assert_array_equal(trimmed["frames"], untrimmed["frames"][kept])
    np.testing.assert_array_equal(trimmed["frames_mask"], untrimmed["frames_mask"][kept])
    np.testing.assert_array_equal(trimmed["flips"], untrimmed["flips"][kept])
    for name, values in trimmed["scalars"].items():
        np.testing.assert_array_equal(values, untrimmed["scalars"][name][kept], err_msg=name)
