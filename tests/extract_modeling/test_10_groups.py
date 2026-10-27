"""Cell 10: say which recordings belong to which condition.

    from moseq2_app.main import interactive_group_setting

    interactive_group_setting(progress_paths['index_file'])

Cell 9 left every recording in group ``default``. Until that changes, every
group comparison in the analysis notebook compares the experiment against
itself, so this cell is the one that makes the rest of the project mean
something.

It shows an editable table of the index: session, subject, group, uuid and
filename, with only the group column editable. A user selects the rows for
one condition, types its name, presses "Set Group Name" to write it into the
selected rows, and presses "Update Index File" to save the table back. The
first button changes the table and nothing else; only the second touches
disk, so a user can reorganise freely and commit once.

Everything here runs in process. The table is a qgridnext grid, which keeps
its data as a DataFrame and exposes the selection, so the tests select rows,
set values and click exactly as a user does.
"""

from __future__ import annotations

import shutil

import ipywidgets as widgets
import pytest

from moseq2_app.main import interactive_group_setting
from moseq2_extract.util import read_yaml

from test_03_config import plain
from test_05_arena_mask import RECORDINGS
from test_09_aggregate import entries

#: The conditions the fixture recordings come from, by name prefix.
CONDITIONS = {"saline": "saline", "amphetamine": "amph"}


def open_table(index):
    """Open the cell's widget on an index file."""
    return interactive_group_setting(str(index))


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
def scratch_index(tmp_path, aggregated, project):
    """A copy of the project's index, for tests that save changes to it."""
    index = tmp_path / "moseq2-index.yaml"
    shutil.copy2(aggregated, index)
    return index


def test_the_cell_displays_the_controls_and_the_table(aggregated, displayed):
    """1. What a user is given."""
    widget = open_table(aggregated)

    controls = displayed.one(widgets.HBox)
    assert [type(child).__name__ for child in controls.children] == [
        "Text",
        "Button",
        "Button",
    ]
    assert controls.children[1].description == "Set Group Name"
    assert controls.children[2].description == "Update Index File"
    assert widget.qgrid_widget in displayed.objects


def test_the_table_shows_every_recording(aggregated):
    """2. One row per recording, with what a user needs to tell them apart."""
    widget = open_table(aggregated)
    table = widget.qgrid_widget.get_changed_df()

    assert list(table.columns) == [
        "SessionName",
        "SubjectName",
        "group",
        "uuid",
        "filename",
    ]
    assert sorted(table["SessionName"]) == RECORDINGS
    assert set(table["group"]) == {"default"}


def test_only_the_group_can_be_edited(aggregated):
    """3. The table is for labelling, not for correcting the metadata.

    A user who could edit a uuid here would break the link between the index
    and the extraction it describes.
    """
    widget = open_table(aggregated)

    assert widget.col_opts["editable"] is False
    assert widget.col_defs == {"group": {"editable": True, "toolTip": "editable"}}


def test_setting_a_name_changes_only_the_selected_rows(scratch_index):
    """4. The point of selecting first."""
    widget = open_table(scratch_index)
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


def test_setting_a_name_does_not_touch_the_index(scratch_index):
    """5. A user can reorganise the table freely and commit once.

    Only "Update Index File" writes, so an accidental selection costs
    nothing until it is saved.
    """
    before = scratch_index.read_text()
    widget = open_table(scratch_index)
    assign(widget, "saline", "saline")

    assert scratch_index.read_text() == before


def test_saving_writes_every_group_to_the_index(scratch_index):
    """6. The cell's output, and what the rest of the project reads."""
    widget = open_table(scratch_index)
    for prefix, group in CONDITIONS.items():
        assign(widget, prefix, group)
    widget.update_index_button.click()

    saved = entries(scratch_index)
    assert {name: entry["group"] for name, entry in saved.items()} == {
        "saline_example_0": "saline",
        "saline_example_1": "saline",
        "amphetamine_example_0": "amph",
        "amphetamine_example_1": "amph",
    }


def test_saving_keeps_the_rest_of_the_index_intact(scratch_index):
    """7. Only the groups move.

    The index is rebuilt from a DataFrame rather than edited in place, so
    everything the later cells read has to survive the round trip.
    """
    before = plain(read_yaml(str(scratch_index)))
    widget = open_table(scratch_index)
    assign(widget, "saline", "saline")
    widget.update_index_button.click()
    after = plain(read_yaml(str(scratch_index)))

    assert sorted(after) == sorted(before)
    assert after["pca_path"] == before["pca_path"]
    before_by_uuid = {entry["uuid"]: entry for entry in before["files"]}
    for entry in after["files"]:
        original = before_by_uuid[entry["uuid"]]
        assert entry["path"] == original["path"]
        assert entry["metadata"] == original["metadata"]


def test_the_save_button_tracks_unsaved_changes(scratch_index):
    """8. The only feedback a user gets about whether the file is current.

    The button passes through three states: plain while nothing has changed,
    highlighted once the table holds edits that are not on disk, and ticked
    once they are. It is the only thing distinguishing a saved table from an
    edited one, since the table itself looks the same either way.
    """
    widget = open_table(scratch_index)
    assert (widget.update_index_button.button_style, widget.update_index_button.icon) == ("", "")

    assign(widget, "saline", "saline")
    assert (widget.update_index_button.button_style, widget.update_index_button.icon) == (
        "info",
        "none",
    )

    widget.update_index_button.click()
    assert (widget.update_index_button.button_style, widget.update_index_button.icon) == (
        "success",
        "check",
    )


@pytest.mark.xfail(
    reason="update_clicked calls self.df.drop('filename', axis=1) and discards "
    "the result, so the column it meant to remove is written out with the rest"
)
def test_saving_does_not_add_a_key_to_the_index(scratch_index):
    """9. The index should hold what it held, with the groups changed.

    Saving adds a ``filename`` key to every entry, which cell 9 did not write
    and nothing reads. It is harmless and idempotent, but it means the file on
    disk is no longer the shape the aggregation produced.
    """
    before = {key for entry in plain(read_yaml(str(scratch_index)))["files"] for key in entry}
    widget = open_table(scratch_index)
    assign(widget, "saline", "saline")
    widget.update_index_button.click()
    after = {key for entry in plain(read_yaml(str(scratch_index)))["files"] for key in entry}

    assert after == before


def test_the_chain_labels_both_conditions(grouped):
    """10. The state the rest of the notebook runs on."""
    assert {entry["group"] for entry in entries(grouped).values()} == {"saline", "amph"}
