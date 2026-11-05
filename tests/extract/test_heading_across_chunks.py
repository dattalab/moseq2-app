"""The mouse's heading is resolved across the whole recording, not per chunk.

Extraction finds the mouse's body axis in each frame, which says nothing about
which end is the head: the angle is known only up to half a turn. It is made
continuous from frame to frame, so a mouse facing east stays facing east, and
without a flip classifier that continuity is all that decides the heading.
Extraction works through a recording in chunks (``--chunk-size``), and the
heading is carried from the last frame of one chunk to the first frame of the
next, so where the recording is cut does not change which way any frame faces
(moseq2-extract#82). With ``--chunk-overlap``, the frames a chunk shares with
the previous one are extracted again, and the heading is still carried from the
frame before the new chunk starts.

These run the real extraction on a copy of the suite's 600-frame
``saline_example_0`` recording, once in a single chunk and once in chunks of
100 frames, and compare the heading of every frame: the angle written to
``scalars/angle``, which is the angle each frame was rotated by.
"""

from __future__ import annotations

import shutil
import warnings

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli

RECORDING = "saline_example_0"
FLIP_CLASSIFIER = "flip_classifier_k2_c57_10to13weeks.npz"
NFRAMES = 600


@pytest.fixture(scope="module")
def recording(raw_sessions, tmp_path_factory):
    """A writable copy of the recording, so extraction writes beside it."""
    root = tmp_path_factory.mktemp("heading")
    shutil.copytree(raw_sessions / RECORDING, root / RECORDING)
    shutil.copy(raw_sessions / FLIP_CLASSIFIER, root / FLIP_CLASSIFIER)
    return root / RECORDING


def extract(recording, output_dir, chunk_size, overlap=0, classifier=False):
    """Extract and return the results file's frames, flips and angles, and the
    warnings the extraction gave."""
    args = [
        "extract",
        str(recording / "depth.avi"),
        "--chunk-size", str(chunk_size),
        "--chunk-overlap", str(overlap),
        "--output-dir", output_dir,
    ]
    if classifier:
        args += ["--flip-classifier", str(recording.parent / FLIP_CLASSIFIER)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = CliRunner().invoke(extract_cli.cli, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    with h5py.File(recording / output_dir / "results.h5", "r") as f:
        return {
            "frames": f["frames"][()],
            "angle": f["scalars/angle"][()],
            "flips": f["metadata/extraction/flips"][()] if classifier else None,
            "warnings": [w for w in caught if "flip classifier" in str(w.message)],
        }


def half_turns_apart(a, b):
    """Frames whose headings differ by more than a quarter turn."""
    d = np.angle(np.exp(1j * (a - b)))
    both = ~np.isnan(a) & ~np.isnan(b)
    return np.flatnonzero(both & (np.abs(d) > np.pi / 2))


@pytest.fixture(scope="module")
def whole(recording):
    return extract(recording, "proc_whole", NFRAMES)


@pytest.fixture(scope="module", params=[0, 20], ids=["no-overlap", "overlap20"])
def chunked(request, recording):
    return extract(recording, f"proc_100_{request.param}", 100, overlap=request.param)


def test_heading_does_not_depend_on_the_chunk_size(whole, chunked):
    assert len(whole["angle"]) == len(chunked["angle"]) == NFRAMES
    flipped = half_turns_apart(whole["angle"], chunked["angle"])
    assert flipped.size == 0, (
        f"{flipped.size} of {NFRAMES} frames face the other way when extracted "
        f"in chunks of 100, starting at frame {flipped[:1]}"
    )


def test_extracting_without_a_flip_classifier_warns_once(recording, whole):
    assert len(whole["warnings"]) == 1
    message = str(whole["warnings"][0].message)
    assert str(recording / "depth.avi") in message
    assert "half a turn" in message
    assert "--flip-classifier" in message


@pytest.fixture(scope="module")
def classified(recording):
    return (
        extract(recording, "proc_whole_flip", NFRAMES, classifier=True),
        extract(recording, "proc_100_flip", 100, classifier=True),
    )


def test_with_a_flip_classifier_the_final_heading_agrees(classified):
    whole, chunked = classified
    assert whole["warnings"] == chunked["warnings"] == []
    flipped = half_turns_apart(whole["angle"], chunked["angle"])
    assert flipped.size == 0, f"{flipped.size} frames face the other way"
