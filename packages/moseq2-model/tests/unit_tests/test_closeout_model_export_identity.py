"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest





@pytest.mark.parametrize('excluded', ['heldout', 'n/a'])
def test_exported_keys_follow_actual_backend_label_order(tmp_path, monkeypatch, excluded):
    from moseq2_model.helpers import wrappers as w
    data = OrderedDict((key, np.zeros((8, 2))) for key in ['excluded', 'second', 'first'])
    metadata = {'groups': dict.fromkeys(data, 'synthetic'), 'uuids': list(data)}
    monkeypatch.setattr(w, 'load_pcs', lambda **kw: (data, metadata))
    monkeypatch.setattr(w, 'process_indexfile', lambda *a: (None, metadata))
    train_keys = ['second', 'first'] if excluded == 'heldout' else list(data)
    hold_keys = ['excluded'] if excluded == 'heldout' else []
    monkeypatch.setattr(w, 'prepare_model_metadata', lambda *a: (data, {}, train_keys, hold_keys, None))
    # This stand-in represents the adapter's existing post-filter session order.
    backend = SimpleNamespace(_session_order=['second', 'first'])
    monkeypatch.setattr(w, 'get_current_model', lambda *a: (backend, 0))
    monkeypatch.setattr(w, 'get_session_groupings', lambda *a: ([], []))
    label_arrays = [np.full(8, 2), np.full(8, 7)]
    monkeypatch.setattr(w, 'train_model', lambda **kw: (backend, [], label_arrays, [], [], False))
    monkeypatch.setattr(w, 'get_loglikelihoods', lambda *a: [])
    monkeypatch.setattr(w, 'get_parameters_from_model', lambda *a: {})
    saved = {}
    monkeypatch.setattr(w, 'save_dict', lambda filename, obj_to_save: saved.update(obj_to_save))
    config = dict(npcs=2, load_groups=False, default_group='synthetic', hold_out=bool(hold_keys),
                  percent_split=0, use_checkpoint=False, num_iter=0, ncpus=1, progressbar=False,
                  separate_trans=False, check_every=1, verbose=False, e_step=False, save_model=False)
    w.learn_model_wrapper('synthetic-unused.h5', str(tmp_path / 'not-written.p'), config)
    assert saved['keys'] == ['second', 'first']
    assert len(saved['keys']) == len(saved['labels'])
    np.testing.assert_array_equal(dict(zip(saved['keys'], saved['labels']))['first'], np.full(8, 7))
