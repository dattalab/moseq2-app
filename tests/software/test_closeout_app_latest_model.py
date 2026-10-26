"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd
import h5py


def test_progress_selects_the_newest_model_path(tmp_path):
    import os
    from moseq2_app.gui.progress import find_progress
    old, new = tmp_path / "old.p", tmp_path / "new.p"
    old.write_bytes(b"synthetic placeholder")
    new.write_bytes(b"synthetic placeholder")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    progress = dict(base_dir=str(tmp_path), config_file=str(tmp_path / "missing.yaml"),
                    pca_dirname=str(tmp_path))
    assert find_progress(progress)["model_path"] == str(new)
