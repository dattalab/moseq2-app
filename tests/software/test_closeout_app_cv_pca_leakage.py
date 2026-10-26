"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd
import h5py


def test_flip_pca_is_inside_cross_validation_pipeline(tmp_path, monkeypatch):
    from moseq2_app.flip import train
    path = tmp_path / "synthetic.npz"
    np.savez(path, frames=np.zeros((12, 8, 8)), flipped=np.tile([False, True], 6))
    class NoPrefitPCA:
        def __init__(self, **kwargs):
            pass
        def fit(self, *args):
            raise AssertionError("PCA fitted before cross-validation")
    class FakePipeline:
        def __init__(self, steps):
            self.steps = steps
        def fit(self, *args):
            return self
    monkeypatch.setattr(train, "PCA", NoPrefitPCA)
    monkeypatch.setattr(train, "make_pipeline", lambda *steps: FakePipeline(steps))
    def inspect_cv(pipeline, *args, **kwargs):
        assert any(isinstance(step, NoPrefitPCA) for step in pipeline.steps)
        return np.ones(4)
    monkeypatch.setattr(train, "cross_val_score", inspect_cv)
    assert isinstance(train.train_classifier(str(path), n_components=2), FakePipeline)
