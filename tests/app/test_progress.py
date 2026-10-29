"""Building ``progress.yaml`` from what is already in a project.

The notebooks' first cell calls ``restore_progress_vars(..., init=True)``.
When there is no progress file yet, or it is asked to overwrite one, that
scans the project with ``find_progress`` and records the paths it finds.
``extract_modeling/test_02_progress.py`` walks the cell; the tests here hold
the scan to what it reports.
"""

from __future__ import annotations

import os

from moseq2_app.gui.progress import find_progress


def test_the_latest_model_is_the_one_chosen(tmp_path):
    """With several models in the project, the scan says it takes the latest."""
    old, new = tmp_path / "old.p", tmp_path / "new.p"
    old.write_bytes(b"model")
    new.write_bytes(b"model")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    progress = dict(
        base_dir=str(tmp_path),
        config_file=str(tmp_path / "missing.yaml"),
        pca_dirname=str(tmp_path),
    )
    assert find_progress(progress)["model_path"] == str(new)
