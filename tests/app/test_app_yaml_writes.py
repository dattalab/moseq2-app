"""moseq2-app's YAML writes go through moseq2-common.

``update_config`` saves the project's ``config.yaml`` after every notebook
step that changes it. A numpy value set in the config made that save raise
``RepresenterError`` after the file had been opened for writing, so the
user's config was left empty. It now saves the plain value, and a value that
truly cannot be saved leaves the file as it was.

``progress.yaml`` and ``syll_info.yaml`` were written by their own
round-trip ``YAML()`` instances with their own indentation. They are written
in the shared format now; files in the old layout read back the same.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ruamel.yaml import YAML

from moseq2_app.gui.progress import generate_intital_progressfile, update_progress
from moseq2_app.util import update_config
from moseq2_app.viz.controller import SyllableLabeler
from moseq2_common.yaml import UnwritableValueError, dump_yaml, read_yaml, write_yaml

#: A long path with spaces, as on macOS or OneDrive. The shared layout folds a
#: long plain value across lines; it must read back unchanged.
SPACED = (
    "/Users/Jane Doe/MoSeq projects/2026 cohort/models/model_0/crowd_movies/"
    "syllable_sorted-id-00_(usage)_original-id-03.mp4"
)


def old_layout(path, data, mapping):
    """Write ``data`` as progress.py (mapping=2) and controller.py (mapping=3) did."""
    yml = YAML()
    yml.indent(mapping=mapping, offset=2)
    with open(path, "w") as f:
        yml.dump(data, f)


def test_update_config_writes_numpy_values_as_plain_values(tmp_path):
    path = tmp_path / "config.yaml"
    write_yaml({"fps": 30}, path)
    with update_config(str(path)) as config:
        config["kappa"] = np.int64(212414)
        config["weights"] = np.array([1.0, 0.1])
    saved = read_yaml(path)
    assert saved == {"fps": 30, "kappa": 212414, "weights": [1.0, 0.1]}
    assert type(saved["kappa"]) is int


def test_update_config_leaves_the_file_alone_when_a_value_cannot_be_saved(tmp_path):
    path = tmp_path / "config.yaml"
    write_yaml({"fps": 30}, path)
    before = path.read_bytes()
    with pytest.raises(UnwritableValueError) as err:
        with update_config(str(path)) as config:
            config["bad"] = 1j
    assert path.read_bytes() == before
    assert str(path) in str(err.value) and "['bad']" in str(err.value)


def test_progress_file_is_rewritten_in_the_shared_format(tmp_path):
    path = tmp_path / "progress.yaml"
    old = {"base_dir": str(tmp_path), "config_file": "", "plot_path": SPACED, "snapshot": "x"}
    old_layout(path, old, mapping=2)
    update_progress(str(path), "config_file", SPACED + ".yaml")
    saved = read_yaml(path)
    assert path.read_text() == dump_yaml(saved)
    assert {k: v for k, v in saved.items() if k != "snapshot"} == {
        "base_dir": str(tmp_path),
        "config_file": SPACED + ".yaml",
        "plot_path": SPACED,
    }
    assert saved["snapshot"] != "x"


def test_new_progress_file_is_written_in_the_shared_format(tmp_path):
    path = tmp_path / "progress.yaml"
    generate_intital_progressfile(str(path))
    assert path.read_text() == dump_yaml(read_yaml(path))


def test_syllable_labels_are_written_in_the_shared_format(tmp_path):
    path = tmp_path / "syll_info.yaml"
    labeller = SimpleNamespace(
        syll_info={
            0: {"label": "rear", "desc": "saut é", "crowd_movie_path": SPACED, "group_info": {"a": 1}},
            1: {"label": "", "desc": "", "crowd_movie_path": "", "group_info": {}},
        },
        max_sylls=2,
        unshown_syll_info={7: {"label": "groom", "desc": "", "crowd_movie_path": ""}},
        save_path=str(path),
    )
    SyllableLabeler.write_syll_info(labeller)
    expected = {
        0: {"label": "rear", "desc": "saut é", "crowd_movie_path": SPACED},
        1: {"label": "", "desc": "", "crowd_movie_path": ""},
        7: {"label": "groom", "desc": "", "crowd_movie_path": ""},
    }
    assert path.read_text() == dump_yaml(expected)
    assert read_yaml(path) == expected


@pytest.mark.parametrize("mapping", [2, 3], ids=["progress", "syll_info"])
def test_files_in_the_old_layout_read_the_same(tmp_path, mapping):
    data = {0: {"label": "saut é", "crowd_movie_path": SPACED}, "base_dir": SPACED}
    old_layout(tmp_path / "old.yaml", data, mapping)
    assert read_yaml(tmp_path / "old.yaml") == data
