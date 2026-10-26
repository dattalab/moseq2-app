"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd
import h5py


def test_qc_prints_the_value_of_a_percentage_flag(monkeypatch, capsys):
    from moseq2_app.roi import validation
    flags = {"synthetic": {"metadata": {"SessionName": "synthetic", "SubjectName": "synthetic"},
                           "dropped_frames": .125}}
    monkeypatch.setattr(validation, "run_validation_tests", lambda *args: flags)
    validation.print_validation_results(pd.DataFrame(), flags)
    assert "12.50%" in capsys.readouterr().out
