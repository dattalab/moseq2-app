"""A batch of extractions ends by saying which recordings failed.

``run_local_extract``, which the notebook's extraction cell and
``batch-extract`` run locally, extracts one recording at a time and catches
each recording's error so the rest still run. With many recordings an error
in the middle scrolls past, so the batch ends with a summary: how many of the
recordings it tried were extracted, and each one that failed with its error.
A recording skipped because it was already extracted counts as neither.

The notebook prints that summary in place of "Extractions Complete." when
anything failed, and ``batch-extract`` exits with a non-zero status.

The extraction itself is replaced by a stand-in that fails for one recording.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
import moseq2_extract.steps as extract_steps
import moseq2_extract.gui as extract_gui
from moseq2_extract.helpers.extract import run_local_extract

SESSIONS = ("session_a", "session_b", "session_c", "session_d")
FAILS = "session_b"
SKIPPED = "session_d"


@pytest.fixture
def project(tmp_path):
    """Four recordings and a local config file."""
    for name in SESSIONS:
        session = tmp_path / name
        session.mkdir()
        (session / "depth.avi").touch()
        (session / "metadata.json").write_text("{}")
    (tmp_path / "config.yaml").write_text("extract:\n  cluster_type: local\n")
    return tmp_path


def stand_in(skipped=(SKIPPED,), fails=(FAILS,)):
    """An extraction that fails for some recordings and skips others.

    ``extract_wrapper`` returns the output directory when it extracts a
    recording and None when it skips one already extracted.
    """

    def extract_wrapper(input_file, output_dir, config_data, num_frames=None, skip=False):
        if any(name in input_file for name in fails):
            raise ValueError(f"the depth video in {input_file} is broken\nsecond line")
        if any(name in input_file for name in skipped):
            return None
        return "proc"

    return extract_wrapper


def recordings(root):
    return [str(root / name / "depth.avi") for name in SESSIONS]


def test_the_batch_ends_with_a_summary_of_its_failures(project, monkeypatch, capsys):
    monkeypatch.setattr(extract_steps, "extract_wrapper", stand_in())

    failures = run_local_extract(recordings(project), str(project / "config.yaml"))

    out = capsys.readouterr().out
    summary = out[out.index("Extracted 2 of 3 recordings"):]
    failed = str(project / FAILS / "depth.avi")
    assert summary.startswith(
        "Extracted 2 of 3 recordings; 1 failed (1 already extracted and skipped).\n"
        "Failed:\n"
        f"  {failed}\n"
        f"    ValueError: the depth video in {failed} is broken\n"
        "    second line\n"
    )
    assert "Fix the problem each error names" in summary
    # batch-extract leaves finished extractions out by default; --skip-checks puts them back
    assert "unless --skip-checks" in summary
    assert "or --skip-completed (batch-extract) is set" not in summary
    assert [path for path, _ in failures] == [failed]


def test_a_batch_without_failures_says_how_many_were_extracted(project, monkeypatch, capsys):
    monkeypatch.setattr(extract_steps, "extract_wrapper", stand_in(fails=()))

    failures = run_local_extract(recordings(project), str(project / "config.yaml"))

    assert failures == []
    out = capsys.readouterr().out
    assert "Extracted 3 of 3 recordings (1 already extracted and skipped).\n" in out
    assert "Failed" not in out


def test_the_notebook_reports_failures_instead_of_complete(project, monkeypatch, capsys):
    monkeypatch.setattr(extract_steps, "extract_wrapper", stand_in())

    extract_gui.extract_found_sessions(str(project), str(project / "config.yaml"), [".avi"])

    out = capsys.readouterr().out
    assert "Extracted 2 of 3 recordings; 1 failed" in out
    assert "Extractions Complete." not in out


def test_the_notebook_says_complete_when_nothing_failed(project, monkeypatch, capsys):
    monkeypatch.setattr(extract_steps, "extract_wrapper", stand_in(fails=()))

    extract_gui.extract_found_sessions(str(project), str(project / "config.yaml"), [".avi"])

    assert "Extractions Complete." in capsys.readouterr().out


@pytest.mark.parametrize("fails, status", [((FAILS,), 1), ((), 0)], ids=["one-fails", "none-fail"])
def test_batch_extract_exits_non_zero_when_a_recording_failed(project, monkeypatch, fails, status):
    monkeypatch.setattr(extract_steps, "extract_wrapper", stand_in(fails=fails))

    result = CliRunner().invoke(
        extract_cli.cli,
        ["batch-extract", str(project), "--config-file", str(project / "config.yaml"),
         "--extensions", ".avi"],
    )

    assert result.exit_code == status, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
