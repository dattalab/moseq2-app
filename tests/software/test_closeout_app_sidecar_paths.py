"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd
import h5py


def test_sidecar_derivation_preserves_mp4_in_parent_path(monkeypatch):
    from moseq2_app.roi import validation
    expected_h5 = "/synthetic/mp4_archive/session.h5"
    expected_yaml = "/synthetic/mp4_archive/session.yaml"
    calls = []
    monkeypatch.setattr(validation, "exists", lambda path: path == expected_yaml)
    monkeypatch.setattr(validation, "read_yaml", lambda path: {
        "metadata": {"SessionName": "synthetic", "SubjectName": "synthetic"}, "uuid": "synthetic"})
    def scalars(filename, **kw):
        calls.append(filename)
        return {"area_px": np.ones(2)}
    monkeypatch.setattr(validation, "h5_to_dict", scalars)
    validation.get_scalar_df({"synthetic": "/synthetic/mp4_archive/session.mp4"})
    assert calls == [expected_h5]
