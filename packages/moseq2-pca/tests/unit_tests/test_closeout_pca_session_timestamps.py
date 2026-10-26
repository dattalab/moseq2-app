"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest





def test_missing_timestamp_fallback_uses_each_sessions_frame_count(tmp_path, monkeypatch):
    import h5py
    from moseq2_pca.pca import util
    files = []
    for name, length in [('first', 5), ('second', 9)]:
        path = tmp_path / (name + '.h5')
        with h5py.File(path, 'w') as f:
            f.create_dataset('frames', data=np.ones((length, 2, 2)))
        files.append(str(path))
    monkeypatch.setattr(util, 'read_yaml', lambda path: {'uuid': path})
    monkeypatch.setattr(util, 'clean_frames', lambda x, **kw: x)
    monkeypatch.setattr(util, 'as_completed', lambda futures, with_results: [(x, x.result) for x in futures])
    class Client:
        def scheduler_info(self):
            return {'workers': {'synthetic': {}}}
        def compute(self, values):
            return [SimpleNamespace(key=str(i), result=x.compute()) for i, x in enumerate(values)]
    destination = str(tmp_path / 'scores')
    util.apply_pca_dask(np.ones((1, 4)), files, ['first', 'second'], False,
                        {'gaussfilter_time': 0, 'medfilter_time': [0]}, destination, 20, {}, False, Client())
    with h5py.File(destination + '.h5') as f:
        for name, length in [('first', 5), ('second', 9)]:
            assert len(f['scores/' + name]) == length
            assert len(f['scores_idx/' + name]) == length
