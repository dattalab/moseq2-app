"""The command line and the notebook extract the same recordings from a project.

``moseq2-extract batch-extract`` and the notebook's extraction cell
(``extract_found_sessions``) run one step, ``moseq2_extract.steps.batch_extract``,
which finds the project's recordings and leaves out the ones already extracted.
Both search every recording type the pipeline reads (``.dat``, ``.mkv``,
``.avi``, ``.tar.gz``) unless told otherwise, and both leave out a recording
whose ``results_00.yaml`` records a finished extraction: the command line
unless ``--skip-checks`` is given, the notebook while ``skip_extracted`` is
True, as its cell sets it.

Each test builds a project of empty recording files and stops the step before
it extracts anything: the list it would have extracted is what it found.

When every recording is already extracted, both say so in one line; when only
some are, they name how many were left out.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner
from moseq2_common.yaml import write_yaml

from moseq2_extract.cli import batch_extract
from moseq2_extract.gui import extract_found_sessions, generate_config_command

#: One session folder per recording type.
RECORDINGS = {
    "dat_session": "depth.dat",
    "mkv_session": "depth.mkv",
    "avi_session": "depth.avi",
    "tar_session": "session_20260101.tar.gz",
}


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project holding one recording of each type and the config.yaml generate-config writes."""
    root = tmp_path / "project"
    for folder, name in RECORDINGS.items():
        (root / folder).mkdir(parents=True)
        (root / folder / name).touch()
        (root / folder / "metadata.json").write_text("{}")
    monkeypatch.chdir(root)
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    return root


@pytest.fixture
def extracted(monkeypatch):
    """The recordings the step hands to the local extraction, in place of extracting them."""
    chosen = []
    monkeypatch.setattr(
        "moseq2_extract.steps.run_local_extract",
        lambda to_extract, config_file, skip_extracted=False, **kw: chosen.extend(to_extract) or [],
    )
    return chosen


def mark_extracted(folder):
    """What a finished extraction leaves behind, as far as skipping reads it."""
    (folder / "proc").mkdir()
    write_yaml({"complete": True, "uuid": "x"}, str(folder / "proc" / "results_00.yaml"))


def every_recording(root):
    return sorted(str(root / folder / name) for folder, name in RECORDINGS.items())


def run_cli(root, *flags):
    result = CliRunner().invoke(batch_extract, [str(root), "--config-file", str(root / "config.yaml"), *flags])
    assert result.exit_code == 0, result.output
    return result


def test_batch_extract_finds_every_recording_type_by_default(project, extracted):
    run_cli(project)

    assert sorted(extracted) == every_recording(project)


@pytest.mark.parametrize(
    "skip_extracted, cli_flags",
    [(True, ()), (False, ("--skip-checks",))],
    ids=["finished-left-out", "finished-included"],
)
def test_both_front_ends_list_the_same_recordings(project, extracted, skip_extracted, cli_flags):
    mark_extracted(project / "avi_session")
    finished = str(project / "avi_session" / "depth.avi")
    expected = [r for r in every_recording(project) if not (skip_extracted and r == finished)]

    run_cli(project, *cli_flags)
    by_cli = sorted(extracted)
    extracted.clear()
    extract_found_sessions(str(project), str(project / "config.yaml"), extract_all=True,
                           skip_extracted=skip_extracted)
    by_notebook = sorted(extracted)

    assert by_cli == expected
    assert by_notebook == expected


def test_a_rerun_with_every_recording_extracted_says_so_in_one_line(project, extracted, capsys):
    for folder in RECORDINGS:
        mark_extracted(project / folder)
    line = (f"All {len(RECORDINGS)} recordings under {project} are already extracted, so there is nothing "
            "to extract. To extract them again, set skip_extracted to False (in the notebook) or add "
            "--skip-checks (moseq2-extract batch-extract).\n")

    by_cli = run_cli(project).output
    capsys.readouterr()
    extract_found_sessions(str(project), str(project / "config.yaml"), extract_all=True, skip_extracted=True)
    by_notebook = capsys.readouterr().out

    assert extracted == []
    assert by_cli == line
    assert by_notebook == line


def test_a_rerun_with_some_recordings_extracted_extracts_the_rest(project, extracted):
    mark_extracted(project / "avi_session")

    output = run_cli(project).output

    assert sorted(extracted) == [r for r in every_recording(project) if not r.endswith("depth.avi")]
    assert "nothing to extract" not in output, output
    assert ("1 recording(s) already extracted were left out, each with a finished extraction recorded "
            f"beside it (such as {project / 'avi_session' / 'proc' / 'results_00.yaml'}). To include them, "
            "set skip_extracted to False (in the notebook) or add --skip-checks "
            "(moseq2-extract batch-extract).") in output, output
