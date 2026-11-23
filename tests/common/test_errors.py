"""Every error a step expects derives from one base, so an interface can tell it from a bug.

``moseq2_common.errors.ExpectedError`` carries the message and, optionally, the
fix and the help page. The step classes that report a problem with a person's
data, settings or project join it and keep the bases they had, so a caller
catching ``ValueError`` (or a test asking ``pytest.raises(ValueError)``) still
catches them.
"""

from __future__ import annotations

import importlib

import pytest

from moseq2_common.errors import ExpectedError, StepRunning, StepStopped, details_for_bug

#: Each class R3 names, where it lives, and the bases it had before joining.
EXPECTED = [
    ("moseq2_common.records", "StepInputError", ()),
    ("moseq2_common.records", "InputError", (ValueError,)),
    ("moseq2_common.bookkeeping", "BookkeepingError", (ValueError,)),
    ("moseq2_pca.project", "NoSessionsError", (ValueError,)),
    ("moseq2_pca.util", "FrameCountError", (ValueError,)),
    ("moseq2_pca.helpers.data", "TrainedSettingsError", (ValueError,)),
    ("moseq2_extract.helpers.wrappers", "FlipClassifierError", (ValueError,)),
    ("moseq2_extract.helpers.wrappers", "FlipClassifierMissingError", (FileNotFoundError,)),
    ("moseq2_common.project", "OldProjectError", ()),
    ("moseq2_common.project", "OutdatedNotebookError", (ImportError,)),
    ("moseq2_common.options", "ConfigFileError", (ValueError,)),
    ("moseq2_pca.util", "TrainingMemoryError", (RuntimeError,)),
    ("moseq2_common.errors", "StepStopped", ()),
    ("moseq2_common.errors", "StepRunning", ()),
]


@pytest.mark.parametrize("module, name, bases", EXPECTED, ids=[name for _, name, _ in EXPECTED])
def test_every_expected_error_derives_from_the_base(module, name, bases):
    cls = getattr(importlib.import_module(module), name)
    error = cls("the message")

    assert isinstance(error, ExpectedError)
    for base in bases:
        assert isinstance(error, base)
    assert str(error) == "the message"


def test_an_expected_error_carries_its_fix_and_help_page():
    error = ExpectedError("It went wrong.", fix="do this.", help_page="Extracting")

    assert (error.message, error.fix, error.help_page) == ("It went wrong.", "do this.", "Extracting")
    assert str(error) == "It went wrong."
    assert issubclass(StepStopped, ExpectedError) and issubclass(StepRunning, ExpectedError)


def test_read_exclude_of_a_bad_list_is_a_config_file_error(tmp_path):
    from moseq2_common.options import ConfigFileError
    from moseq2_common.project import Project, read_exclude, read_groups

    project = Project.at(tmp_path)
    project.config_file.write_text("extract: {}\nexclude: {a: 1}\ngroups: [x]\n")

    with pytest.raises(ConfigFileError, match="`exclude` holds"):
        read_exclude(project)
    with pytest.raises(ConfigFileError, match="`groups` holds"):
        read_groups(project)


def test_details_for_bug_hold_the_traceback_then_what_was_captured():
    try:
        raise KeyError("missing")
    except KeyError as error:
        details = details_for_bug(error, "printed while it ran\n")

    assert details.index("Traceback") < details.index("KeyError") < details.index("printed while it ran")
