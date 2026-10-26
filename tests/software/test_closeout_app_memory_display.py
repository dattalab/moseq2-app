"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd
import h5py





def test_memory_report_uses_actual_array_bytes(tmp_path, monkeypatch, capsys):
    import pickle
    from moseq2_app.flip import train
    path, index = tmp_path / 'synthetic.h5', tmp_path / 'synthetic.p'
    with h5py.File(path, 'w') as f:
        f['frames'] = np.zeros((500, 80, 80), dtype=np.uint8)
    with index.open('wb') as f:
        pickle.dump(({'synthetic': str(path)}, {'synthetic': [(False, slice(None))]}), f)
    monkeypatch.setattr(train, 'clean_frames', lambda values, *a, **kw: values)
    recorded = {}
    monkeypatch.setattr(np, 'savez', lambda path, **arrays: recorded.update(arrays))
    train.create_training_dataset(str(index))
    message = capsys.readouterr().out
    expected = recorded['frames'].nbytes / 1e9
    assert f'memory usage: {expected:0.2f} GB' in message
