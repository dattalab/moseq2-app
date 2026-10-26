"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd
import h5py





@pytest.mark.parametrize('returncode', [0, 1])
def test_preview_writer_failure_is_not_reported_as_completion(tmp_path, monkeypatch, returncode):
    from moseq2_app.flip import widget
    path = tmp_path / 'synthetic.h5'
    with h5py.File(path, 'w') as f:
        f['frames'] = np.zeros((5, 8, 8), dtype=np.uint8)
        f['scalars/angle'] = np.zeros(5)
    class Pipe:
        def communicate(self):
            self.returncode = returncode
    monkeypatch.setattr(widget, 'get_flips', lambda values, **kw: np.zeros(len(values), dtype=bool))
    monkeypatch.setattr(widget, 'write_frames_preview', lambda *a, **kw: Pipe())
    subject = SimpleNamespace(sessions={'synthetic': path})
    def run():
        widget.FlipClassifierWidget.apply_flip_classifier(subject, 'unused', write_movie=True,
                                                           smoothing=1, verbose=False)
    if returncode:
        with pytest.raises(RuntimeError, match='exit code 1'):
            run()
    else:
        run()
