"""Cell 7: check the extractions before anything is built on them.

    from moseq2_app.main import validate_extractions

    validate_extractions(progress_paths['base_dir'])

The first of two review cells. It reads every extraction in the project,
measures four things that go wrong often enough to be worth catching early,
and prints a report. Nothing is returned and nothing is written; the output is
the whole product.

Four of the flags are percentages of a recording's frames and are raised at
five percent: ``corrupted`` for frames whose scalars could not be measured,
``missing`` for frames with no mouse, ``stationary`` for frames where it did
not move, and ``size_anomaly`` for frames where it measured too small.
``dropped_frames`` is a fifth, taken from the timestamps rather than the
scalars. ``scalar_anomaly`` is the odd one out: a recording is flagged when
its mean scalars fall outside the inter-quartile range of the others, so it
is relative, and with four recordings the quartiles are narrow enough that it
moves between them as the rest of the report changes. The tests here pin what
the report says about the absolute flags and only the shape of what it says
about that one.

The report is written for a terminal, so every line carries ANSI colour
codes. ``report`` strips them.
"""

from __future__ import annotations

import contextlib
import io
import re
import shutil

import h5py
import numpy as np
import pytest

from moseq2_app.main import validate_extractions

from test_05_arena_mask import RECORDINGS
from test_06_extract import proc

#: Frames out of 600 to make unmeasurable: comfortably under and over the
#: five percent at which a flag is raised.
UNDER, OVER = 12, 60

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def report(project):
    """Run the cell and return what it printed, without the colour codes."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert validate_extractions(str(project)) is None
    return ANSI.sub("", printed.getvalue())


def corrupt(project, recording, frames):
    """Make the first ``frames`` frames of one recording unmeasurable.

    Setting the scalars to NaN is what a frame the extraction could not
    measure looks like on disk, which is what ``corrupted`` counts.
    """
    with h5py.File(proc(project, recording) / "results_00.h5", "r+") as f:
        for scalar in f["scalars"]:
            values = f[f"scalars/{scalar}"][:]
            values[:frames] = np.nan
            f[f"scalars/{scalar}"][:] = values


@pytest.fixture
def reviewable(tmp_path, extracted):
    """A copy of the extracted project, for tests that damage a recording."""
    root = tmp_path / "project"
    shutil.copytree(extracted, root)
    return root


def test_the_report_counts_every_recording(extracted):
    """1. The headline: how many of the project's recordings want a look."""
    printed = report(extracted)
    flagged = re.search(r"(\d+)/(\d+) were flagged with warning\(s\)\.", printed)

    assert flagged, f"no count in the report:\n{printed}"
    assert int(flagged.group(2)) == len(RECORDINGS)
    assert int(flagged.group(1)) <= len(RECORDINGS)
    assert "may need visually inspection for outliers" in printed


def test_a_recording_whose_frames_cannot_be_measured_is_flagged(reviewable):
    """2. The flag that catches an extraction that silently went wrong.

    A tenth of this recording's frames have no scalars, which is what the
    cell exists to notice before a model is fit on them.
    """
    corrupt(reviewable, RECORDINGS[0], OVER)
    printed = report(reviewable)

    assert f"Session: {RECORDINGS[0]};" in printed
    assert "corrupted flag raised" in section(printed, RECORDINGS[0])


def test_a_few_unmeasurable_frames_are_not_worth_flagging(reviewable):
    """3. The other side of the five percent threshold.

    Every recording in this fixture already loses its opening frames to the
    background, so a flag that fired on any missing frame at all would fire on
    all four and say nothing.
    """
    corrupt(reviewable, RECORDINGS[0], UNDER)
    printed = report(reviewable)

    assert "corrupted flag raised" not in printed


def test_a_flagged_recording_is_named_with_its_subject(reviewable):
    """4. What a user needs to find the recording the report is talking about."""
    corrupt(reviewable, RECORDINGS[0], OVER)
    printed = report(reviewable)

    assert re.search(
        rf"Session: {RECORDINGS[0]}; Subject: \S+ flags:", printed
    ), f"the flagged recording is not named:\n{printed}"


def test_an_unflagged_recording_is_not_listed(reviewable):
    """5. The report is a list of what to look at, not of everything."""
    corrupt(reviewable, RECORDINGS[0], OVER)
    printed = report(reviewable)

    listed = set(re.findall(r"Session: (\S+);", printed))
    assert RECORDINGS[0] in listed
    assert listed <= set(RECORDINGS)


@pytest.mark.xfail(
    reason="a raised flag holds the percentage as a nonzero float, which is "
    "truthy, so the branch that prints it is only reachable by 0.0 and the "
    "number is never shown"
)
def test_the_percentage_behind_a_flag_is_printed(reviewable):
    """6. How bad a flag is decides whether a user acts on it.

    The cell measures the percentage and formats a line for it. What a user
    sees is `corrupted flag raised` and nothing else, so a recording that lost
    six percent of its frames reads exactly like one that lost sixty.
    """
    corrupt(reviewable, RECORDINGS[0], OVER)
    printed = report(reviewable)

    assert re.search(
        r"corrupted flag raised: \d+\.\d\d%", section(printed, RECORDINGS[0])
    ), f"no percentage behind the flag:\n{printed}"


def section(printed, recording):
    """The part of the report belonging to one recording."""
    blocks = printed.split("Session: ")
    for block in blocks[1:]:
        if block.startswith(f"{recording};"):
            return block
    return ""
