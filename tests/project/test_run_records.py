"""Every output carries one record of the run that produced it.

A record names the step, the front end and, for a command, what was typed; the
start and end times; the versions of the moseq2 packages and of the libraries
results depend on; the options that ran, after every detection; each input,
with its paths, its fingerprint (size and a hash of its content) and, for a
file the pipeline wrote, the run that wrote it; and the outputs' array shapes. An ``.h5`` output holds it as its ``/provenance`` attribute, any other
output in a ``.yaml`` beside it. A fact that cannot be gathered reads
``unknown: <reason>`` and never stops the run, and nothing in a record names the
user or the machine beyond what its paths contain.

No check reads a file's modification time. A file the pipeline wrote is the run
its record names (its start and end times); a raw recording, which nothing
rewrites, is its size and the hash of its first and last MB. A project copied
with new file times is the same project.

The extraction record replaces the session's ``results.yaml`` and the ``.h5``'s
``metadata/extraction/parameters``: an extraction is complete when its
``results.h5`` exists, and an extraction stopped partway leaves the session's
previous ``.h5``, its record and ``moseq2-project.yaml`` as they were.

``print-record`` on each of the four programs prints an output's record, found
from the output itself, a session's folder or its key, for pasting into an issue.

1.5.0 wrote the configured settings (not the detected ones) into the ``.h5`` and
a ``results_00.yaml`` beside it, which it rewrote with ``complete: false`` when
an extraction started.
"""

from __future__ import annotations

import getpass
import io
import os
import shutil
import socket
from contextlib import contextmanager
from pathlib import Path

import h5py
import pytest
from click.testing import CliRunner

from moseq2_common.yaml import read_yaml

RECORDING = "saline_example_0"
KEY = f"{RECORDING}/depth.avi"


@contextmanager
def working_in(folder):
    before = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(before)


def invoke(cli, *args, cwd):
    with working_in(cwd):
        return CliRunner().invoke(cli, [str(arg) for arg in args], catch_exceptions=False)


def ok(cli, *args, cwd):
    """Invoke ``cli`` with ``args`` from ``cwd``, as a person in that folder types it; it must succeed."""
    result = invoke(cli, *args, cwd=cwd)
    assert result.exit_code == 0, result.output
    return result


def extract_cli():
    from moseq2_extract.cli import cli

    return cli


def extracted_project(root, raw_sessions):
    """A project holding one recording, extracted (30 frames) and aggregated on the command line."""
    root.mkdir(parents=True)
    shutil.copytree(raw_sessions / RECORDING, root / RECORDING)
    ok(extract_cli(), "generate-config", "--output-file", root / "config.yaml", cwd=root)
    ok(extract_cli(), "extract", root / KEY, "--config-file", root / "config.yaml", "--num-frames", 30, cwd=root)
    ok(extract_cli(), "aggregate-results", cwd=root)
    return root


def session_h5(root):
    from moseq2_common.bookkeeping import session_index
    from moseq2_common.project import Project

    (h5,) = [f["path"][0] for f in session_index(Project.at(root))["files"] if f["key"] == KEY]
    return Path(h5)


@pytest.fixture(scope="module")
def project(tmp_path_factory, raw_sessions):
    return extracted_project(tmp_path_factory.mktemp("records") / "project", raw_sessions)


# ------------------------------------------------------------ the extraction record


def test_the_extraction_record_holds_the_detected_depth_range(project):
    """The record holds the options extraction ran with after its detections: the depth range found from
    the background (not config.yaml's), the camera told from the recording and the floor depth measured.
    The session's folder holds no results.yaml and the ``.h5`` no ``metadata/extraction/parameters``."""
    from moseq2_common.records import read_record

    h5 = session_h5(project)
    configured = read_yaml(str(project / "config.yaml"))["extract"]
    record = read_record(h5)

    options = record["options"]
    assert options["manual_set_depth_range"] is False
    assert list(options["bg_roi_depth_range"]) != list(configured["bg_roi_depth_range"])
    low, high = options["bg_roi_depth_range"]
    assert low < options["true_depth"] < high
    assert options["camera_type"] == "kinect"
    assert record["step"] == "extract"
    assert record["front_end"] == "cli"
    with h5py.File(h5, "r") as f:
        assert "metadata/extraction/parameters" not in f
        assert record["outputs"]["frames"]["shape"] == list(f["frames"].shape)
    assert not (h5.parent / "results.yaml").exists()
    recording = [i for i in record["inputs"] if i["path"].endswith(KEY)]
    assert recording and recording[0]["relative_path"] == KEY
    assert recording[0]["frame_count"] > 0


def test_an_interrupted_step_leaves_the_previous_output_and_bookkeeping(tmp_path, raw_sessions, monkeypatch):
    """An extraction stopped partway through writing its frames (Ctrl-C) leaves the session's previous
    ``.h5`` with its record, its preview movie and moseq2-project.yaml byte for byte, and nothing under the
    ``.h5``'s final name. (1.5.0's record was ``results.yaml``, rewritten when an extraction started.)"""
    import moseq2_extract.helpers.wrappers as wrappers

    root = extracted_project(tmp_path / "project", raw_sessions)
    h5 = session_h5(root)
    kept = [h5, h5.with_suffix(".mp4"), h5.with_suffix(".yaml"), root / "moseq2-project.yaml"]
    kept = {path: path.read_bytes() for path in kept if path.exists()}

    real = wrappers.process_extract_batches

    def stopped(*args, **kwargs):
        kwargs["frame_batches"] = kwargs["frame_batches"][:1]
        real(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(wrappers, "process_extract_batches", stopped)
    result = invoke(extract_cli(), "extract", root / KEY, "--config-file", root / "config.yaml",
                    "--num-frames", 30, cwd=root)
    assert result.exit_code != 0

    for path, data in kept.items():
        assert path.read_bytes() == data, f"{path} changed"
    assert [p.name for p in h5.parent.glob("*.h5")] == ["results.h5"]


def test_no_record_field_names_the_user_or_the_machine(project):
    """Outside the paths it records, nothing in a record holds the user's name, the machine's name or
    the home folder."""
    from moseq2_common.records import read_record

    record = read_record(session_h5(project))
    root = str(project.parent)
    found = []

    def strings(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value

    for text in strings(record):
        text = text.replace(root, "<project>")
        if text.startswith("/") or "<project>" in text:
            continue
        found.append(text)
    words = [getpass.getuser(), socket.gethostname(), str(Path.home())]
    leaks = [(word, text) for text in found for word in words if word and word in text]
    assert not leaks, leaks


# ------------------------------------------------------------ print-record


def _clis():
    from moseq2_extract.cli import cli as extract
    from moseq2_model.cli import cli as model
    from moseq2_pca.cli import cli as pca
    from moseq2_viz.cli import cli as viz

    return {"moseq2-extract": extract, "moseq2-pca": pca, "moseq2-model": model, "moseq2-viz": viz}


@pytest.mark.parametrize("given", ["h5", "folder", "key"])
def test_print_record_prints_a_sessions_record(project, given):
    """``print-record`` takes the session's ``.h5``, its folder in aggregate_results or its key, and prints
    one line naming the file, then the record as stored; every program has the command."""
    from moseq2_common.records import read_record

    h5 = session_h5(project)
    argument = {"h5": h5, "folder": h5.parent, "key": KEY}[given]
    expected = read_record(h5)
    for name, cli in _clis().items():
        result = ok(cli, "print-record", argument, cwd=project)
        first, _, rest = result.output.partition("\n")
        assert first == f"# record of {h5}", (name, first)
        assert read_yaml(io.StringIO(rest)) == expected, name


def test_print_record_names_what_it_could_not_find(project):
    result = invoke(extract_cli(), "print-record", "no/such/session.avi", cwd=project)
    assert result.exit_code != 0
    assert "no/such/session.avi" in result.output


# ------------------------------------------------------------ soft facts


def test_a_fact_that_cannot_be_gathered_is_unknown(tmp_path, monkeypatch):
    """Run from a checkout with no ``git`` to ask, the record's commit reads ``unknown: <reason>`` and the
    record is still written."""
    from moseq2_common import records
    from moseq2_extract.options import ExtractOptions

    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    record = records.start_record("extract", ExtractOptions(), [], project=None)
    output = tmp_path / "out.png"
    output.write_bytes(b"png")
    written = records.write_sidecar(output, record.finish({}))
    data = read_yaml(str(written))
    assert str(data["git"]).startswith("unknown: "), data["git"]
    assert data["versions"]["moseq2-extract"]


# ------------------------------------------------------------ what identifies a file


def test_a_fingerprint_holds_no_file_time(tmp_path):
    """Two copies of a file with different modification times have the same fingerprint."""
    from moseq2_common.records import fingerprint

    first = tmp_path / "depth.avi"
    first.write_bytes(os.urandom(3 * 1024 ** 2))
    second = tmp_path / "copy.avi"
    shutil.copyfile(first, second)
    os.utime(second, (1e9, 1e9))
    assert fingerprint(first) == fingerprint(second)
    assert set(fingerprint(first)) == {"size", "sha256", "hashed"}


def test_a_written_input_is_named_by_its_run(tmp_path):
    """An input the pipeline wrote is recorded with the run its own record names, and is unchanged while it
    carries that record, whatever its file time; written again, it is another run."""
    from moseq2_common.records import run_of, start_record, unchanged, write_record

    model = tmp_path / "model.p"
    model.write_bytes(b"a model")
    write_record(model, start_record("learn-model", {}, []).finish({}))
    fitted = run_of(model)
    assert set(fitted) == {"step", "started", "ended"} and fitted["step"] == "learn-model"

    entry = start_record("make-df", {}, [model]).finish({})["inputs"][0]
    assert entry["run"] == fitted
    os.utime(model, (1e9, 1e9))
    assert unchanged(entry, model)

    write_record(model, start_record("learn-model", {}, []).finish({}))
    assert run_of(model) != fitted
    assert not unchanged(entry, model)


def test_a_yaml_output_keeps_its_record_beside_it(tmp_path):
    """A ``.yaml`` output's record is ``<stem>.record.yaml``, and ``read_record`` finds it there."""
    from moseq2_common.records import read_record, sidecar_path, start_record, write_record

    labels = tmp_path / "syll_info.yaml"
    labels.write_text("0: {label: rear}\n")
    assert sidecar_path(labels) == tmp_path / "syll_info.record.yaml"
    write_record(labels, start_record("save-syllable-labels", {}, []).finish({}))
    assert read_record(labels)["step"] == "save-syllable-labels"
    assert read_yaml(str(labels)) == {0: {"label": "rear"}}
