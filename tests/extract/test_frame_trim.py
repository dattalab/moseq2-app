"""Trimming frames off the start or end of a recording.

``--frame-trim A B`` extracts a recording without its first ``A`` and last
``B`` frames. The output holds only the kept frames, so frame ``A`` of the
recording is frame 0 of the extraction, and every per-frame dataset and the
timestamps line up on that numbering (moseq2-extract#157).

These run the real extraction on a copy of the suite's 600-frame
``saline_example_0`` recording, with its flip classifier, in chunks of 100
frames. Extraction resolves the mouse's heading only up to half a turn, by
keeping it continuous from the first extracted frame on, so a trimmed
extraction, which starts at a later frame, can crop every frame rotated half a
turn from the untrimmed one. What places each frame and does not depend on the
heading (its centroid, area, length, width and 2D velocity) has to come out
identical, and the trimmed extraction's heading has to differ from the
untrimmed one's by the same amount on every frame.
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
    with h5py.File(recording / output_dir / "results.h5", "r") as f:
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


# per-frame scalars computed before the crop is rotated, so they do not depend
# on which way the mouse is taken to face
HEADING_FREE = [
    "area_mm", "area_px", "centroid_x_mm", "centroid_x_px", "centroid_y_mm",
    "centroid_y_px", "length_mm", "length_px", "width_mm", "width_px",
    "velocity_2d_mm", "velocity_2d_px", "velocity_theta",
]


def test_each_kept_frame_is_the_same_frame_of_the_recording(untrimmed, trimmed):
    """Frame ``i`` of the trimmed extraction is frame ``start + i`` of the untrimmed one."""
    start, end, trimmed = trimmed
    kept = slice(start, NFRAMES - end)
    for name in HEADING_FREE:
        np.testing.assert_array_equal(
            trimmed["scalars"][name], untrimmed["scalars"][name][kept], err_msg=name
        )


def test_the_trimmed_heading_is_turned_as_a_whole(untrimmed, trimmed):
    """Before flip correction, the trimmed extraction's heading differs from the
    untrimmed one's by one fixed amount, a whole number of half turns, on every
    frame: no chunk boundary turns part of it around."""
    start, end, trimmed = trimmed
    kept = slice(start, NFRAMES - end)

    def raw(result, frames=slice(None)):
        return result["scalars"]["angle"][frames] - np.pi * result["flips"][frames]

    turn = np.exp(1j * (raw(trimmed) - raw(untrimmed, kept)))
    turn = turn[~np.isnan(turn)]
    assert np.allclose(turn.imag, 0, atol=1e-6)
    assert np.all(np.sign(turn.real) == np.sign(turn.real[0]))
