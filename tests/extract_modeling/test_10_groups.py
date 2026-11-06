"""Cell 10: say which recordings belong to which condition.

    from moseq2_app.main import interactive_group_setting

    interactive_group_setting(str(project.folder))

Cell 9 left every recording in group ``default``. Until that changes, every
group comparison in the analysis notebook compares the experiment against
itself, so this cell is the one that makes the rest of the project mean
something.

It shows an editable table of the project's sessions: session, subject, group,
exclude, uuid and filename, with only the group and exclude columns editable.
Ticking a session's exclude saves it in ``config.yaml``'s ``exclude``, which every
step after extraction leaves out. A user selects the rows
for one condition, types its name, presses "Set Group Name" to write it into the
selected rows, and presses "Save Groups" to save each session's group in
``config.yaml``'s ``groups``, keyed by its recording. The first button changes
the table and nothing else; only the second touches disk, so a user can
reorganise freely and commit once.

Everything here runs in process. The table is a qgridnext grid, which keeps
its data as a DataFrame and exposes the selection, so the tests select rows,
set values and click exactly as a user does.
"""

from __future__ import annotations

import shutil

import ipywidgets as widgets
import pytest

from moseq2_app.main import interactive_group_setting
from moseq2_common.yaml import read_yaml

from test_03_config import plain
from test_05_arena_mask import RECORDINGS
from test_09_aggregate import entries

#: The conditions the fixture recordings come from, by name prefix.
CONDITIONS = {"saline": "saline", "amphetamine": "amph"}


def open_table(project):
    """Open the cell's widget on a project folder."""
    return interactive_group_setting(str(project))


def select(widget, prefix):
    """Select every row whose recording name starts with ``prefix``."""
    table = widget.qgrid_widget.get_changed_df()
    widget.qgrid_widget.change_selection(
        [i for i, name in zip(table.index, table["SessionName"]) if name.startswith(prefix)]
    )


def assign(widget, prefix, group):
    """Do what a user does: select the rows, name the group, press Set."""
    select(widget, prefix)
    widget.group_input.value = group
    widget.save_button.click()


@pytest.fixture
def scratch_project(tmp_path, project, aggregated, bookkeeping_as_aggregated):
    """The project as cell 9 left it, for tests that save changes: its sessions, and a config.yaml without groups."""
    from moseq2_common.options import read_config
    from moseq2_common.yaml import write_config

    root = tmp_path / "project"
    shutil.copytree(project / "aggregate_results", root / "aggregate_results")
    shutil.copy2(bookkeeping_as_aggregated, root / "moseq2-project.yaml")
    config = read_config(project / "config.yaml")
    config.pop("groups", None)
    write_config(config, str(root / "config.yaml"))
    return root


def test_the_cell_displays_the_controls_and_the_table(scratch_project, displayed):
    """1. What a user is given."""
    widget = open_table(scratch_project)

    controls = displayed.one(widgets.HBox)
    assert [type(child).__name__ for child in controls.children] == [
        "Text",
        "Button",
        "Button",
    ]
    assert controls.children[1].description == "Set Group Name"
    assert controls.children[2].description == "Save Groups"
    assert widget.qgrid_widget in displayed.objects


def test_the_table_shows_every_recording(scratch_project):
    """2. One row per recording, with what a user needs to tell them apart."""
    widget = open_table(scratch_project)
    table = widget.qgrid_widget.get_changed_df()

    assert list(table.columns) == [
        "SessionName",
        "SubjectName",
        "group",
        "exclude",
        "uuid",
        "filename",
    ]
    assert sorted(table["SessionName"]) == RECORDINGS
    assert set(table["group"]) == {"default"}
    assert not table["exclude"].any()


def test_only_the_group_and_exclude_can_be_edited(scratch_project):
    """3. The table is for labelling, not for correcting the metadata."""
    widget = open_table(scratch_project)

    assert widget.col_opts["editable"] is False
    assert widget.col_defs == {
        "group": {"editable": True, "toolTip": "editable"},
        "exclude": {"editable": True, "toolTip": "leave the session out of every step after extraction"},
    }


def test_ticking_exclude_saves_the_session_in_exclude(scratch_project):
    """3b. A session ticked in the exclude column is saved in config.yaml's exclude, and shown ticked again."""
    widget = open_table(scratch_project)
    table = widget.qgrid_widget.get_changed_df()
    row = next(i for i, name in zip(table.index, table["SessionName"]) if name == "saline_example_1")
    widget.qgrid_widget.edit_cell(row, "exclude", True)
    widget.save_groups_button.click()

    assert read_yaml(str(scratch_project / "config.yaml"))["exclude"] == ["saline_example_1/depth.avi"]
    reopened = open_table(scratch_project).qgrid_widget.get_changed_df()
    assert dict(zip(reopened["SessionName"], reopened["exclude"]))["saline_example_1"]


def test_setting_a_name_changes_only_the_selected_rows(scratch_project):
    """4. The point of selecting first."""
    widget = open_table(scratch_project)
    assign(widget, "saline", "saline")

    table = widget.qgrid_widget.get_changed_df()
    groups = dict(zip(table["SessionName"], table["group"]))
    assert {name: groups[name] for name in groups if name.startswith("saline")} == {
        "saline_example_0": "saline",
        "saline_example_1": "saline",
    }
    assert all(
        groups[name] == "default" for name in groups if name.startswith("amphetamine")
    )


def test_setting_a_name_does_not_touch_config_yaml(scratch_project):
    """5. A user can reorganise the table freely and commit once.

    Only "Save Groups" writes, so an accidental selection costs nothing until
    it is saved.
    """
    before = (scratch_project / "config.yaml").read_bytes()
    widget = open_table(scratch_project)
    assign(widget, "saline", "saline")

    assert (scratch_project / "config.yaml").read_bytes() == before


def test_saving_writes_every_group_to_config_yaml(scratch_project):
    """6. The cell's output, and what the rest of the project reads."""
    widget = open_table(scratch_project)
    for prefix, group in CONDITIONS.items():
        assign(widget, prefix, group)
    widget.save_groups_button.click()

    assert read_yaml(str(scratch_project / "config.yaml"))["groups"] == {
        "saline_example_0/depth.avi": "saline",
        "saline_example_1/depth.avi": "saline",
        "amphetamine_example_0/depth.avi": "amph",
        "amphetamine_example_1/depth.avi": "amph",
    }
    saved = entries(scratch_project / "moseq2-project.yaml")
    assert {name: entry["group"] for name, entry in saved.items()} == {
        "saline_example_0": "saline",
        "saline_example_1": "saline",
        "amphetamine_example_0": "amph",
        "amphetamine_example_1": "amph",
    }


def test_saving_changes_nothing_but_the_groups(scratch_project):
    """7. Only the groups move: config.yaml's settings and the bookkeeping file stay as they were."""
    before_config = plain(read_yaml(str(scratch_project / "config.yaml")))
    before_bookkeeping = (scratch_project / "moseq2-project.yaml").read_bytes()
    widget = open_table(scratch_project)
    assign(widget, "saline", "saline")
    widget.save_groups_button.click()
    after_config = plain(read_yaml(str(scratch_project / "config.yaml")))

    assert {k: v for k, v in after_config.items() if k != "groups"} == before_config
    assert (scratch_project / "moseq2-project.yaml").read_bytes() == before_bookkeeping


def test_the_save_button_tracks_unsaved_changes(scratch_project):
    """8. The only feedback a user gets about whether the groups are saved.

    The button passes through three states: plain while nothing has changed,
    highlighted once the table holds edits that are not on disk, and ticked
    once they are.
    """
    widget = open_table(scratch_project)
    assert (widget.save_groups_button.button_style, widget.save_groups_button.icon) == ("", "")

    assign(widget, "saline", "saline")
    assert (widget.save_groups_button.button_style, widget.save_groups_button.icon) == (
        "info",
        "none",
    )

    widget.save_groups_button.click()
    assert (widget.save_groups_button.button_style, widget.save_groups_button.icon) == (
        "success",
        "check",
    )


def test_the_chain_labels_both_conditions(grouped):
    """9. The state the rest of the notebook runs on."""
    assert {entry["group"] for entry in entries(grouped / "moseq2-project.yaml").values()} == {"saline", "amph"}
