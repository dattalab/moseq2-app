"""Cell 12: find out which recording a point on a plot came from.

    from moseq2_app.util import uuid_lookup

    target_uuid = '40'

    uuid_lookup(target_uuid, viewer.sorted_index['files'])

A companion to cell 11 rather than a step of its own. The violins there label
their points by uuid, so a user who sees an outlier has a uuid and nothing
else. This prints the recording behind it: session, subject, start time, and
the two files it was built from.

The uuid is matched as a substring, which is why the notebook's example is
two characters. A user reads enough of a uuid off a hover label to be
unambiguous and types that. A fragment matching several recordings prints
all of them, and one matching none prints nothing at all, which is the only
signal that a lookup failed.

It is handed ``viewer.sorted_index['files']`` from the previous cell, so this
is also where the viewer that cell 11 returned gets used.
"""

from __future__ import annotations

import pytest

from moseq2_app.util import uuid_lookup

from tests.extract_modeling.test_05_arena_mask import RECORDINGS

#: The fields the lookup prints, in the order it prints them.
REPORTED = ("SessionName", "SubjectName", "StartTime", "h5Path", "yamlPath")


@pytest.fixture
def files(scalar_viewer):
    """The uuid-keyed index the previous cell hands over."""
    return scalar_viewer.sorted_index["files"]


def look_up(capsys, target, files):
    """Run the cell and return what it printed."""
    assert uuid_lookup(target, files) is None
    return capsys.readouterr().out


def test_a_full_uuid_finds_its_recording(capsys, files):
    """1. The ordinary case."""
    uuid, entry = next(iter(files.items()))
    printed = look_up(capsys, uuid, files)

    assert f"UUID: {uuid}" in printed
    assert entry["metadata"]["SessionName"] in printed
    assert entry["metadata"]["SubjectName"] in printed


def test_a_fragment_of_a_uuid_is_enough(capsys, files):
    """2. Why the notebook's example is two characters.

    A user reading a uuid off a hover label types as much of it as they can
    be bothered to, and the lookup matches on the fragment.
    """
    uuid = next(iter(files))
    printed = look_up(capsys, uuid[:8], files)

    assert f"UUID: {uuid}" in printed


def test_the_recording_is_reported_with_the_files_behind_it(capsys, files):
    """3. What the lookup is for: getting from a uuid back to the data."""
    uuid, entry = next(iter(files.items()))
    printed = look_up(capsys, uuid, files)

    for field in REPORTED:
        assert field in printed
    for path in entry["path"]:
        assert path in printed


def test_a_fragment_shared_by_several_recordings_reports_all_of_them(capsys, files):
    """4. An ambiguous fragment is not an error.

    Every uuid is a hex string, so a single character usually matches more
    than one. The cell prints each match rather than choosing.
    """
    shared = next(
        character
        for character in "0123456789abcdef"
        if sum(character in uuid for uuid in files) > 1
    )
    printed = look_up(capsys, shared, files)

    matching = [uuid for uuid in files if shared in uuid]
    assert printed.count("UUID:") == len(matching)
    for uuid in matching:
        assert uuid in printed


def test_an_empty_target_reports_every_recording(capsys, files):
    """5. Every string contains the empty string."""
    printed = look_up(capsys, "", files)
    assert printed.count("UUID:") == len(RECORDINGS)


def test_a_uuid_that_matches_nothing_says_nothing(capsys, files):
    """6. Current behaviour, captured.

    A typo prints an empty cell, which is indistinguishable from the cell not
    having run. There is no "no such recording" to tell a user they mistyped.
    """
    assert look_up(capsys, "zzzzzzzz", files) == ""


def test_looking_up_does_not_change_the_index(capsys, files):
    """7. The paths it prints are added to a copy.

    The lookup folds the two file paths into the metadata so it can print
    them in one block. It deep-copies first, so the viewer the next lookup
    reads is the one cell 11 built.
    """
    uuid = next(iter(files))
    before = {key: dict(entry["metadata"]) for key, entry in files.items()}

    look_up(capsys, uuid, files)

    assert {key: dict(entry["metadata"]) for key, entry in files.items()} == before
    assert "h5Path" not in files[uuid]["metadata"]
