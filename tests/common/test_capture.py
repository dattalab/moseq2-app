"""Everything a running step emits reaches its interface, and nothing else.

``moseq2_common.capture``: a step run under ``in_background`` (as a notebook GUI
runs one) has its prints, log records and library progress bars kept in its own
details and its warnings sent to its reporter, while a print from the cell that
runs meanwhile reaches that cell's output as before. Output below Python, on the
process's file descriptors, is out of the proxies' reach, so nothing on the
extraction path starts a subprocess that writes there.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
import warnings

from moseq2_common import capture, report
from reporting import CollectingReporter
from extract_projects import dat_recording


def test_a_background_steps_output_stays_out_of_other_cells(capsys):
    reporter = CollectingReporter()
    started, cell_printed = threading.Event(), threading.Event()
    finished = []

    def step():
        from tqdm import tqdm

        started.set()
        cell_printed.wait(10)
        print("printed by the step")
        warnings.warn("the step's warning")
        for _ in tqdm(range(3), desc="a library's bar"):
            pass
        report.current().status("the step's status")
        # last: pytest's live logging swaps the test's capture streams back in while it writes a record
        logging.getLogger("moseq2.extract").warning("logged by the step")
        return "done"

    thread = capture.in_background(step, reporter=reporter, on_finish=finished.append)
    started.wait(10)
    print("printed by the cell")
    cell_printed.set()
    thread.join(30)

    out, err = capsys.readouterr()
    assert out == "printed by the cell\n"
    assert err == ""
    (outcome,) = finished
    assert outcome.value == "done" and outcome.error is None
    assert "printed by the step" in outcome.details
    assert "logged by the step" in outcome.details
    assert "a library's bar" in outcome.details
    assert reporter.warnings == ["the step's warning"]
    assert reporter.statuses == ["the step's status"]


def test_an_error_in_the_background_comes_back_in_the_outcome():
    finished = []

    def step():
        print("before the error")
        raise KeyError("the bug")

    capture.in_background(step, reporter=CollectingReporter(), on_finish=finished.append).join(30)

    (outcome,) = finished
    assert isinstance(outcome.error, KeyError)
    assert outcome.details == "before the error\n"


def test_outside_a_run_everything_passes_through(capsys):
    capture.install()
    print("a cell's own line")
    print("a cell's own error", file=sys.stderr)

    assert capsys.readouterr() == ("a cell's own line\n", "a cell's own error\n")


def test_subprocesses_reach_neither_stream(tmp_path, raw_sessions):
    from moseq2_extract import steps
    from moseq2_extract.io.video import get_movie_info
    from moseq2_extract.options import ExtractOptions

    recording = dat_recording(tmp_path / "session", raw_sessions)
    avi = raw_sessions / "saline_example_0" / "depth.avi"
    reporter, details = CollectingReporter(), io.StringIO()
    written = tmp_path / "descriptors.txt"

    capture.install()
    saved = os.dup(1), os.dup(2)
    sys.stdout.flush()
    sys.stderr.flush()
    with open(written, "w") as target:
        os.dup2(target.fileno(), 1)
        os.dup2(target.fileno(), 2)
        try:
            with report.reporting(reporter), capture.captured(reporter, details):
                get_movie_info(str(avi))
                steps.extract(str(recording), None, ExtractOptions(compress=True, num_frames=None))
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])

    assert written.read_text() == ""
    assert (recording.parent / "depth.avi").is_file()
    assert (recording.parent / "proc" / "results.h5").is_file()
