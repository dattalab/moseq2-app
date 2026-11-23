"""``moseq2 extract`` in a project, ``moseq2 standalone extract`` on recordings named, and ``moseq2 run-job``.

``moseq2 extract`` lists every recording it plans as a session before it
extracts any, saves the settings typed before the run, and on a cluster writes
one job per recording asking for 4 CPUs and running ``moseq2 run-job`` through
the Python that wrote it. ``moseq2 standalone extract`` reads and writes no
project file, takes recordings on the command line or in a list file, and
writes each one's outputs into ``proc/`` beside it.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path

from click.testing import CliRunner

from moseq2_app.cli import cli
from moseq2_common.bookkeeping import load
from moseq2_common.records import read_record
from moseq2_common.yaml import read_yaml, write_config
from extract_projects import dat_recording, small_project
from layers import trap_project

TWO = ("saline_example_0", "saline_example_1")


def _run(args, cwd):
    before = os.getcwd()
    os.chdir(cwd)
    try:
        return CliRunner().invoke(cli, [str(a) for a in args], prog_name="moseq2")
    finally:
        os.chdir(before)


def _computing(project, **values):
    data = read_yaml(str(project.config_file))
    data["computing"]["extract"].update(values)
    write_config(data, str(project.config_file))


def test_moseq2_extract_lists_every_session_then_extracts(tmp_path, raw_sessions, monkeypatch):
    project = small_project(tmp_path / "project", raw_sessions, TWO)
    seen = []

    def extract_recording(project_, input_file, opts, **kwargs):
        listed = load(project_).sessions
        seen.append((Path(input_file).parent.name, sorted(listed),
                     sorted(Path(project.folder).rglob("results.h5"))))

    monkeypatch.setattr("moseq2_extract.project.extract_recording", extract_recording)
    result = _run(["extract"], project.folder)

    assert result.exit_code == 0, result.output
    keys = [f"{name}/depth.avi" for name in TWO]
    assert seen == [(TWO[0], keys, []), (TWO[1], keys, [])]
    assert "Extracted 2 of 2 recordings." in result.stderr


def test_typed_settings_are_saved_before_the_run(tmp_path, raw_sessions, monkeypatch):
    project = small_project(tmp_path / "project", raw_sessions, TWO)
    at_run = []
    monkeypatch.setattr("moseq2_extract.project.extract_recording",
                        lambda project_, input_file, opts, **kwargs: at_run.append((input_file, opts)))

    assert _run(["extract", "--crop-size", "100", "100"], project.folder).exit_code == 0
    assert read_yaml(str(project.config_file))["extract"]["crop_size"] == [100, 100]
    assert {opts.crop_size for _, opts in at_run} == {(100, 100)}

    at_run.clear()
    result = _run(["extract", "--spatial-filter-size", "5", "saline_example_0/depth.avi"], project.folder)
    assert result.exit_code == 0, result.output
    assert read_yaml(str(project.session_config_file)) == {"saline_example_0/depth.avi": {"spatial_filter_size": [5]}}
    assert read_yaml(str(project.config_file))["extract"]["spatial_filter_size"] == [3]
    assert [(Path(f).parent.name, opts.spatial_filter_size) for f, opts in at_run] == [("saline_example_0", (5,))]


def test_standalone_extract_takes_a_list_and_reads_no_project_file(tmp_path, raw_sessions):
    folder = tmp_path / "project"
    for name in TWO:
        shutil.copytree(Path(raw_sessions) / name, folder / name)
    trap = trap_project(folder, entries={f"{name}/depth.avi": {"noise_tolerance": 99} for name in TWO},
                        sessions={f"{name}/depth.avi": f"elsewhere/{name}" for name in TWO})

    result = _run(["standalone", "extract", *(f"{name}/depth.avi" for name in TWO), "--num-frames", "30"], folder)

    assert result.exit_code == 0, result.output
    assert trap.unchanged()
    for name in TWO:
        h5 = folder / name / "proc" / "results.h5"
        record = read_record(h5)
        assert record["project"] is None
        assert record["options"]["noise_tolerance"] == 30
    assert not (folder / "aggregate_results").exists() and not (folder / "elsewhere").exists()


def test_standalone_files_from_reads_one_path_per_line(tmp_path, monkeypatch):
    for name in ("a", "b", "c"):
        (tmp_path / "data" / name).mkdir(parents=True)
        (tmp_path / "data" / name / "depth.avi").write_bytes(b"x")
    listed = tmp_path / "lists" / "recordings.txt"
    listed.parent.mkdir()
    listed.write_text(f"# this week's\n../data/a/depth.avi\n\n{tmp_path / 'data' / 'b' / 'depth.avi'}  # absolute\n")
    called = []
    monkeypatch.setattr("moseq2_extract.steps.extract",
                        lambda recording, output_dir, opts, **kwargs: called.append(Path(recording).resolve()))

    result = _run(["standalone", "extract", "data/c/depth.avi", "--files-from", listed], tmp_path)

    assert result.exit_code == 0, result.output
    assert called == [(tmp_path / "data" / name / "depth.avi").resolve() for name in ("c", "a", "b")]

    listed.write_text("../data/a/depth.avi\n../data/gone/depth.avi\n")
    result = _run(["standalone", "extract", "--files-from", listed], tmp_path)
    assert result.exit_code == 1
    assert result.stderr == (f"Error: {listed} names ../data/gone/depth.avi on line 2, which does not exist.\n"
                             "To fix it: correct or delete that line.\n")


def test_standalone_output_dir_with_several_recordings_is_refused(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "depth.avi").write_bytes(b"x")

    result = _run(["standalone", "extract", "a/depth.avi", "b/depth.avi", "--output-dir", "out"], tmp_path)

    assert result.exit_code == 1
    assert result.stderr == ("Error: `--output-dir` names one recording's output folder, and 2 recordings were "
                             "given.\nTo fix it: leave out --output-dir to write each recording's outputs into "
                             "proc/ beside it, or extract one recording per command.\n")


def _jobs(tmp_path, raw_sessions):
    project = small_project(tmp_path / "project", raw_sessions, TWO, num_frames=30)
    _computing(project, cluster_type="slurm")
    result = _run(["extract"], project.folder)
    assert result.exit_code == 0, result.output
    return project, result


def test_cluster_jobs_ask_for_4_cpus_and_run_with_4_threads(tmp_path, raw_sessions):
    project, result = _jobs(tmp_path, raw_sessions)

    lines = (Path(project.folder) / "extract_out.sh").read_text().splitlines()
    assert len(lines) == 2
    for name, line in zip(TWO, lines):
        words = shlex.split(line)
        assert words[:9] == ["sbatch", "-c", "4", "--mem=16GB", "-p", "short", "-t", "1:00:00", "--wrap"]
        job_file = Path(project.folder) / "aggregate_results" / name / "extract-job.yaml"
        assert words[9] == shlex.join([sys.executable, "-m", "moseq2_app", "run-job", str(job_file)]) + "; "
        job = read_yaml(str(job_file))
        assert "threads" not in job["extract"] and "compress_threads" not in job["extract"]
        assert job["job"] == {"project": str(Path(project.folder).absolute()), "session": f"{name}/depth.avi",
                              "uuid": load(project).sessions[f"{name}/depth.avi"].uuid,
                              "recording": f"{name}/depth.avi", "cpus": 4}
        assert job["extract"]["flip_classifier"] == str(Path(project.folder) / "flip_classifier_k2_c57_10to13weeks.npz")
    assert "Listing extract commands..." in result.stderr
    assert not list(Path(project.folder).rglob("results.h5"))


def test_a_job_runs_its_recording_from_its_settings_file(tmp_path, raw_sessions):
    project, _ = _jobs(tmp_path, raw_sessions)
    job_file = Path(project.folder) / "aggregate_results" / TWO[0] / "extract-job.yaml"

    result = _run(["run-job", job_file], tmp_path)

    assert result.exit_code == 0, result.output
    h5 = Path(project.folder) / "aggregate_results" / TWO[0] / "results.h5"
    record = read_record(h5)
    assert (record["options"]["threads"], record["options"]["compress_threads"]) == (4, 4)
    assert record["project"] == str(Path(project.folder).resolve())
    import h5py

    with h5py.File(h5, "r") as f:
        uuid = f["metadata/uuid"][()]
    assert (uuid.decode() if isinstance(uuid, bytes) else uuid) == load(project).sessions[f"{TWO[0]}/depth.avi"].uuid


def test_compressing_after_extraction_needs_no_old_command(tmp_path, raw_sessions, monkeypatch):
    project = small_project(tmp_path / "project", raw_sessions, (), num_frames=None, extract={"compress": True})
    recording = dat_recording(Path(project.folder) / "session_1", raw_sessions, n_frames=60)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))

    result = _run(["extract"], project.folder)

    assert result.exit_code == 0, result.output
    assert (recording.parent / "depth.avi").is_file()
    assert (Path(project.folder) / "aggregate_results" / "session_1" / "results.h5").is_file()
