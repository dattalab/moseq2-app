"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd
import h5py


def test_index_save_does_not_persist_display_filename(monkeypatch):
    from moseq2_app.gui import widgets
    captured = {}
    frame = pd.DataFrame({"uuid": ["synthetic"], "path": [["synthetic.h5", "synthetic.yaml"]],
                          "group": ["default"], "filename": ["synthetic.h5"],
                          "SessionName": ["synthetic"], "metadata": [{"SessionName": "synthetic"}]})
    subject = SimpleNamespace(
        index_dict={"files": [{"metadata": {"SessionName": "synthetic"}}], "pca_path": "synthetic-pca.h5"},
        df=frame, qgrid_widget=SimpleNamespace(get_changed_df=lambda: frame.copy()),
        index_filepath="synthetic-unused.yaml", update_index_button=SimpleNamespace())
    monkeypatch.setattr(widgets, "_write_yaml", lambda value, path: captured.update(value))
    widgets.GroupSettingWidgets.update_clicked(subject)
    assert "filename" not in captured["files"][0]
