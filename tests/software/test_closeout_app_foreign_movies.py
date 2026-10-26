"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd
import h5py





def test_movie_path_inventory_ignores_foreign_mp4(tmp_path, monkeypatch):
    from moseq2_app.viz import controller
    import inspect
    cls = next(obj for obj in vars(controller).values() if inspect.isclass(obj) and 'get_crowd_movie_paths' in obj.__dict__)
    directory = tmp_path / 'movies'
    directory.mkdir()
    (directory / 'notes.mp4').write_bytes(b'not a movie; filename-only fixture')
    valid = directory / 'syllable_sorted-id-1000.(usage)_original-id-1200.mp4'
    valid.write_bytes(b'filename-only fixture')
    subject = SimpleNamespace(max_sylls=1, syll_info={1000: {'crowd_movie_path': ''}}, save_path=str(tmp_path / 'info.yaml'))
    cls.get_crowd_movie_paths(subject, 'unused-index', 'unused-model', {}, str(directory) + '/')
    assert subject.syll_info[1000]['crowd_movie_path'] == str(valid)
    assert subject.syll_info[1000]['original_id'] == 1200
