"""A step reports what it is doing through a reporter; without one, the default prints on stderr.

``moseq2_common.report``: status lines, warnings and progress bars go to the
reporter current in the thread (``reporting``); outside any, a
``TerminalReporter`` on ``sys.stderr``, so a script's own standard output stays
its own. ``checkpoint`` stops a step when its reporter says the person asked to.
"""

from __future__ import annotations

import io
import threading

import pytest

from moseq2_common import report
from moseq2_common.errors import StepStopped
from reporting import CollectingReporter


def test_default_reporter_prints_lines_and_bars_on_stderr(capsys):
    reporter = report.current()
    reporter.status("Processing: depth.avi")
    reporter.warning("the timestamps are short")
    with reporter.progress("Processing batches", total=3) as bar:
        for _ in range(3):
            bar.advance()

    out, err = capsys.readouterr()
    assert out == ""
    assert "Processing: depth.avi\n" in err
    assert "Warning: the timestamps are short\n" in err
    assert "Processing batches" in err and "3/3" in err


def test_quiet_keeps_warnings_only():
    stream = io.StringIO()
    reporter = report.TerminalReporter(stream, quiet=True)
    reporter.status("Processing: depth.avi")
    reporter.warning("the timestamps are short")
    with reporter.progress("Processing batches", total=2) as bar:
        bar.advance(2)

    assert stream.getvalue() == "Warning: the timestamps are short\n"


def test_reporting_is_per_thread():
    first, second = CollectingReporter(), CollectingReporter()
    ready = threading.Barrier(2)

    def step(reporter, name):
        with report.reporting(reporter):
            ready.wait()
            for i in range(50):
                report.current().status(f"{name} {i}")

    threads = [threading.Thread(target=step, args=(first, "first")),
               threading.Thread(target=step, args=(second, "second"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert first.statuses == [f"first {i}" for i in range(50)]
    assert second.statuses == [f"second {i}" for i in range(50)]
    assert report.current() is report.default_reporter()


def test_reporting_none_keeps_the_callers_reporter():
    outer = CollectingReporter()
    with report.reporting(outer), report.reporting(None) as inner:
        report.current().status("from the inner step")

    assert inner is outer
    assert outer.statuses == ["from the inner step"]


def test_checkpoint_stops_when_asked():
    reporter = CollectingReporter()
    with report.reporting(reporter):
        report.checkpoint()
        reporter.request_stop()
        with pytest.raises(StepStopped):
            report.checkpoint()
