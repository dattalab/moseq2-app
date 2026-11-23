"""Generating the configuration every later step reads.

    generate_config_command(str(project.config_file), camera_type='auto')

``generate_config_command`` does not write a template. It writes every
setting each stage's options classes declare, at its declared default, into
that stage's section (``extract``, ``pca``, ``model``, ``crowd_movies``), with
``camera_type`` at the top, so the contents of a user's config are a function
of those declarations.

The full-config baseline below is therefore the alarm for that whole class of
problem. A change to any default turns it red on purpose.

Generating the config is the one place the camera is chosen. ``'azure'``
(an Azure Kinect or an Orbbec Femto) writes ``camera_type: azure`` and that
camera's depth range, filters and crop in ``extract`` and its PCA filters in
``pca``, the same as ``moseq2-extract generate-config --camera-type azure``;
``'kinect'`` (or ``'k2'``) writes ``camera_type: auto``, so extraction tells the
camera from the file, and ``'auto'`` takes ``'azure'`` when every recording is
an ``.mkv``. Later steps read the key, and no notebook cell sets it.
"""

from __future__ import annotations

import json
import re

import pytest

from moseq2_extract.gui import generate_config_command
from moseq2_common.yaml import read_yaml

from cells import NOTEBOOKS

#: The only settings `camera_type='azure'` changes, by (section, key); None for the top level.
AZURE_OVERRIDES = {
    ("extract", "bg_roi_depth_range"): [550, 650],
    ("extract", "spatial_filter_size"): [5],
    ("extract", "tail_filter_size"): [15, 15],
    ("extract", "crop_size"): [120, 120],
    ("pca", "gaussfilter_space"): [2.25, 1.5],
    ("pca", "tailfilter_size"): [13, 13],
    (None, "camera_type"): "azure",
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


def generate(path, camera_type="auto"):
    generate_config_command(str(path), camera_type=camera_type)
    return plain(read_yaml(str(path)))


def settings(config, within=None):
    """Every setting by (section, key), the top-level ones under section None.

    A block inside a section (``computing``'s ``extract``) gives its settings
    as (section, block, key).
    """
    out = {}
    for key, value in config.items():
        if isinstance(value, dict):
            out.update(settings(value, (key,) if within is None else (*within, key)))
        else:
            out[(None, key) if within is None else (*within, key)] = value
    return out


def test_generated_config_is_unchanged(tmp_path, data_regression):
    """1. Every default, pinned.

    This is the cell's real test. The values come from a dependency's
    introspected defaults, so a change here is a change to what every user's
    pipeline runs with, whether or not anyone meant it.
    """
    data_regression.check(generate(tmp_path / "config.yaml", camera_type="auto"))


def test_azure_changes_exactly_its_parameters(tmp_path):
    """2. The azure branch's whole contract."""
    k2 = settings(generate(tmp_path / "k2.yaml", camera_type="k2"))
    azure = settings(generate(tmp_path / "azure.yaml", camera_type="azure"))

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


@pytest.mark.parametrize("notebook", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda p: p.name)
def test_no_notebook_cell_resets_the_camera(notebook):
    """A cell that assigned ``camera_type`` would undo the choice made when the config was generated."""
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    for number, cell in enumerate((c for c in cells if c["cell_type"] == "code"), start=1):
        assert not re.search(r"\['camera_type'\]\s*=(?!=)", "".join(cell["source"])), f"code cell {number}"


def test_every_value_is_a_plain_yaml_type(tmp_path):
    """3. Nothing in the config is a library object that leaked through.

    The click regression that broke this cell put a sentinel where a boolean
    belonged. A sentinel survives being written and comes back as something
    no downstream step can use, so the failure surfaced far from its cause.
    Checking the types here names it at the source.
    """
    config = settings(generate(tmp_path / "config.yaml"))
    bad = {}
    for key, value in config.items():
        if isinstance(value, list):
            if not all(isinstance(item, SCALARS) for item in value):
                bad[key] = value
        elif not isinstance(value, SCALARS):
            bad[key] = value
    assert not bad, f"config values that are not plain yaml types: {bad}"


def test_config_lands_where_the_notebook_puts_it(project, started, configured):
    """4. The file is written in the project folder, where every later cell finds it."""
    assert configured.exists()
    assert configured.parent == project
    assert configured.name == "config.yaml"

    assert started.config_file == configured


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
    path.write_text("extract:\n  edited_by_hand: true\n")

    monkeypatch.setattr("builtins.input", lambda *_: "y")
    message = generate_config_command(str(path), camera_type="k2")

    assert message == "Configuration file has been successfully generated."
    rewritten = plain(read_yaml(str(path)))
    assert "edited_by_hand" not in rewritten["extract"]
    assert rewritten["extract"]["bg_roi_depth_range"] == [650, 750]


def test_first_generation_reports_success(tmp_path):
    """The message the cell prints when there was nothing to overwrite."""
    path = tmp_path / "config.yaml"
    assert generate_config_command(str(path), camera_type="k2") == (
        "Configuration file has been successfully generated."
    )
