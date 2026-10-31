"""Cell 3: generate the configuration every later step reads.

    config_filepath = join(progress_paths['base_dir'], 'config.yaml')
    camera_type = 'k2'
    generate_config_command(config_filepath, camera_type=camera_type)
    progress_paths = update_progress(progress_filepath, 'config_file', config_filepath)

``generate_config_command`` does not write a template. It harvests the
defaults off moseq2-extract's own click ``extract`` command and dumps them to
yaml, so the contents of a user's config are a function of what click reports
a default to be. That is why a click release once broke this cell outright on
a fresh install: the defaults themselves were fine, and what changed was how
they presented.

The full-config baseline below is therefore the alarm for that whole class of
problem. A dependency that changes any default turns it red on purpose.

This cell is the one place the notebook asks for the camera. ``'azure'``
(an Azure Kinect or an Orbbec Femto) writes ``camera_type: azure`` and that
camera's depth range, filters and crop, the same as ``moseq2-extract
generate-config --camera-type azure``; ``'k2'`` leaves ``camera_type: auto``,
so extraction tells the camera from the file. Later cells read the key and
never set it.
"""

from __future__ import annotations

import re

import pytest

from moseq2_app.gui.progress import restore_progress_vars
from moseq2_extract.gui import generate_config_command
from moseq2_extract.util import read_yaml

from cells import code_cell

EXTRACT = "MoSeq2-Extract-Modeling-Notebook.ipynb"

#: The only keys `camera_type='azure'` changes.
AZURE_OVERRIDES = {
    "bg_roi_depth_range": [550, 650],
    "spatial_filter_size": [5],
    "tail_filter_size": [15, 15],
    "crop_size": [120, 120],
    "camera_type": "azure",
}

#: What a config value is allowed to be once written and read back.
SCALARS = (type(None), bool, int, float, str)


def plain(value):
    """Strip ruamel's container subclasses so comparisons are over plain types."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def generate(path, camera_type="k2"):
    generate_config_command(str(path), camera_type=camera_type)
    return plain(read_yaml(str(path)))


def test_generated_config_is_unchanged(tmp_path, data_regression):
    """1. Every default, pinned.

    This is the cell's real test. The values come from a dependency's
    introspected defaults, so a change here is a change to what every user's
    pipeline runs with, whether or not anyone meant it.
    """
    data_regression.check(generate(tmp_path / "config.yaml", camera_type="k2"))


def test_azure_changes_exactly_its_parameters(tmp_path):
    """2. The azure branch's whole contract."""
    k2 = generate(tmp_path / "k2.yaml", camera_type="k2")
    azure = generate(tmp_path / "azure.yaml", camera_type="azure")

    assert set(k2) == set(azure), "azure added or dropped a key"
    differing = {key for key in k2 if k2[key] != azure[key]}
    assert differing == set(AZURE_OVERRIDES), (
        f"azure differs from k2 in {sorted(differing)}, expected "
        f"{sorted(AZURE_OVERRIDES)}"
    )
    for key, expected in AZURE_OVERRIDES.items():
        assert azure[key] == expected


def test_the_camera_chosen_here_is_the_one_the_config_records(tmp_path):
    """Later cells read ``camera_type``, so the choice made here has to land in it."""
    assert generate(tmp_path / "k2.yaml", camera_type="k2")["camera_type"] == "auto"
    assert generate(tmp_path / "azure.yaml", camera_type="azure")["camera_type"] == "azure"


@pytest.mark.parametrize("cell", [5, 13], ids=["arena", "pca"])
def test_later_cells_do_not_reset_the_camera(cell):
    """A later cell that assigned ``camera_type`` would undo the choice made here."""
    assert not re.search(r"config_data\['camera_type'\]\s*=(?!=)", code_cell(EXTRACT, cell))


def test_every_value_is_a_plain_yaml_type(tmp_path):
    """3. Nothing in the config is a library object that leaked through.

    The click regression that broke this cell put a sentinel where a boolean
    belonged. A sentinel survives being written and comes back as something
    no downstream step can use, so the failure surfaced far from its cause.
    Checking the types here names it at the source.
    """
    config = generate(tmp_path / "config.yaml")
    bad = {}
    for key, value in config.items():
        if isinstance(value, list):
            if not all(isinstance(item, SCALARS) for item in value):
                bad[key] = value
        elif not isinstance(value, SCALARS):
            bad[key] = value
    assert not bad, f"config values that are not plain yaml types: {bad}"


def test_config_lands_where_the_notebook_puts_it(project, progress_file, configured):
    """4. The file is written beside the recordings and recorded in progress.yaml."""
    assert configured.exists()
    assert configured.parent == project
    assert configured.name == "config.yaml"

    recorded = restore_progress_vars(str(progress_file), init=False, overwrite=False)
    assert recorded["config_file"] == str(configured)


def test_regenerating_over_an_existing_config_asks_first(tmp_path, monkeypatch):
    """5a. Declining the prompt keeps the file exactly as it was."""
    path = tmp_path / "config.yaml"
    generate_config_command(str(path), camera_type="k2")
    path.write_text("edited_by_hand: true\n")

    monkeypatch.setattr("builtins.input", lambda *_: "n")
    message = generate_config_command(str(path), camera_type="k2")

    assert message == "Configuration file has been retained"
    assert path.read_text() == "edited_by_hand: true\n"


def test_accepting_the_prompt_regenerates(tmp_path, monkeypatch):
    """5b. Accepting it replaces the file with fresh defaults."""
    path = tmp_path / "config.yaml"
    generate_config_command(str(path), camera_type="k2")
    path.write_text("edited_by_hand: true\n")

    monkeypatch.setattr("builtins.input", lambda *_: "y")
    message = generate_config_command(str(path), camera_type="k2")

    assert message == "Configuration file has been successfully generated."
    rewritten = plain(read_yaml(str(path)))
    assert "edited_by_hand" not in rewritten
    assert rewritten["bg_roi_depth_range"] == [650, 750]


def test_first_generation_reports_success(tmp_path):
    """The message the cell prints when there was nothing to overwrite."""
    path = tmp_path / "config.yaml"
    assert generate_config_command(str(path), camera_type="k2") == (
        "Configuration file has been successfully generated."
    )
