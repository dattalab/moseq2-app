import os
import math
import joblib
import shutil
import unittest
import numpy as np
import pandas as pd
from operator import add
from copy import deepcopy
from functools import reduce
from unittest import TestCase
from cytoolz import keyfilter, groupby
from moseq2_viz.model.trans_graph import get_transitions
from moseq2_viz.scalars.util import scalars_to_dataframe
from moseq2_viz.util import parse_index, get_index_hits, load_changepoint_distribution, load_timestamps, read_yaml
from moseq2_viz.model.util import (relabel_by_usage, h5_to_dict, retrieve_pcs_from_slices,
    get_best_fit, get_syllable_statistics, parse_model_results, merge_models, get_mouse_syllable_slices,
    syllable_slices_from_dict, get_syllable_slices, labels_to_changepoints,
    _gen_to_arr, normalize_pcs, _whiten_all, simulate_ar_trajectory, whiten_pcs,
    make_separate_crowd_movies, get_normalized_syllable_usages, get_Xy_values, compute_behavioral_statistics)

def make_sequence(lbls, durs):
    arr = [[x] * y for x, y in zip(lbls, durs)]
    return np.array(reduce(add, arr))

class TestModelUtils(TestCase):

    def test_get_Xy_values(self):

        index_path = 'data/test_index.yaml'
        model_path = 'data/test_model.p'

        _, sorted_index = parse_index(index_path)

        # compute session scalar data
        scalar_df = scalars_to_dataframe(sorted_index, model_path=model_path)
        mean_df = compute_behavioral_statistics(scalar_df, count='usage',
                                                groupby=['group', 'uuid'],
                                                usage_normalization=True)

        X, y, mapping, rev_mapping = get_Xy_values(mean_df,
                                                   mean_df.group.unique(),
                                                   stat='usage')

        assert list(mapping.keys()) == list(rev_mapping.values())
        assert list(mapping.values()) == list(rev_mapping.keys())

        assert len(X) == len(y)

        assert X.ndim == 2

    def test_index_hits(self):

        test_index = 'data/test_index.yaml'
        index = parse_index(test_index)[0]
        metadata = [f['metadata'] for f in index['files']]

        config_data = {
            'lowercase': True,
            'negative': False,
        }

        key = 'SessionName'
        v = '012'

        hits = get_index_hits(config_data, metadata, key, v)

        assert len(hits) == 2

    def test_load_changepoint_distribution(self):

        cp_path = 'data/_pca/changepoints.h5'

        cps = load_changepoint_distribution(cp_path)

        assert cps.shape == (100,)

    def test_load_timestamps(self):

        ts_path = 'data/depth_ts.txt'

        ts_contents = """350709011.4036 0
350709044.3977 0
350709077.4811 0
350709111.4189 0
350709144.4593 0
350709177.5818 0
350709211.5186 0
350709244.3859 0
350709277.4546 0
350709411.3864 0
350709577.5677 0
350709611.3888 0"""

        with open(ts_path, 'w') as f:
            f.write(ts_contents)

        ts = load_timestamps(ts_path)

        assert len(ts) == 12

        os.remove(ts_path)

    def test_get_transitions(self):
        true_labels = [1, 2, 4, 1, 5]
        durs = [3, 4, 2, 6, 7]
        arr = make_sequence(true_labels, durs)

        trans, locs = get_transitions(arr)

        assert true_labels[1:] == list(trans), 'syllable labels do not match with the transitions'
        assert list(np.diff(locs)) == durs[1:-1], 'syllable locations do not match their durations'

    def test_get_syllable_usages(self):

        test_model = 'data/test_model.p'

        model = parse_model_results(test_model)

        mean_usages = get_normalized_syllable_usages(model, count='usage')
        assert len(mean_usages) == 100
        assert math.isclose(sum(mean_usages), 1.0)

        mean_usages = get_normalized_syllable_usages(model, max_syllable=40, count='usage')
        assert len(mean_usages) == 40
        assert math.isclose(sum(mean_usages), 1.0)

    def test_get_mouse_syllable_slices(self):
        syllable = 2
        labels = [1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2]
        slices = get_mouse_syllable_slices(syllable, labels)

        assert isinstance(slices, list)
        assert slices != []
        assert list(slices) == [slice(3, 6, None), slice(9, 12, None)]

    def test_syllable_slices_from_dict(self):
        model_fit = 'data/test_model.p'
        index_file = 'data/test_index.yaml'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/_pca/pca_scores.h5'

        model_data = parse_model_results(model_fit)
        lbl_dict = {}
        labels, _ = relabel_by_usage(model_data['labels'])
        for k,v in zip(model_data['keys'], labels):
            lbl_dict[k] = v

        ret = syllable_slices_from_dict(2, lbl_dict, index_data)

        assert isinstance(ret, dict)
        assert len(ret.keys()) == 2


    def test_get_syllable_slices(self):
        model_fit = 'data/mock_model.p'
        index_file = 'data/test_index_crowd.yaml'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'

        model_data = parse_model_results(model_fit)
        labels = model_data['labels']

        labels, _ = relabel_by_usage(labels)
        label_uuids = model_data['keys']

        lbl_dict = {}
        for k, v in zip(model_data['keys'], model_data['labels']):
            lbl_dict[k] = v

        ret = syllable_slices_from_dict(2, lbl_dict, index_data)
        syllable_slices = get_syllable_slices(2, labels, label_uuids, index_data)

        # The slices are ragged, so this is an object array. numpy built one
        # implicitly until 1.24 and needs the dtype stated from 2.0.
        assert np.asarray(syllable_slices, dtype=object).shape == (2*len(list(ret.values())[0]), 3)
        assert len(syllable_slices) == 2*len(list(ret.values())[0])
        assert len(list(ret.values())[0]) == len(list(ret.values())[1])

    def test_get_syllable_statistics(self):
        # For now this just tests if there are any function-related errors
        model_fit = 'data/mock_model.p'
        index_file = 'data/test_index_crowd.yaml'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'

        model_data = parse_model_results(model_fit)
        labels = model_data['labels']

        labels, _ = relabel_by_usage(labels)
        max_syllable = 40
        usages, durations = get_syllable_statistics(labels, max_syllable=max_syllable)
        assert len(usages) == len(durations) == max_syllable

        Fusages, Fdurations = get_syllable_statistics(labels, max_syllable=max_syllable, count='frames')
        assert len(Fusages) == len(Fdurations) == max_syllable

        assert usages.values() != Fusages.values()

        Uusages, Udurations = get_syllable_statistics(labels, max_syllable=max_syllable, count='fraasdfsdgmes')
        assert len(Uusages) == len(Udurations) == max_syllable

        assert list(usages.values()) == list(Uusages.values())

    def test_labels_to_changepoints(self):

        labels = np.asarray([[-5, -5, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 5, 5, 5]])
        cps = labels_to_changepoints(labels, fs=30.)

        # number of changepoints == number of unique labels in list -1
        # number of unique labels is decremented by 1 to account for nlag prefixes
        assert len(cps) == len(list(set(labels[0])))-2

    def test_parse_model_results(self):
        model_fit = 'data/mock_model.p'

        model_dict = parse_model_results(model_fit)
        assert 'keys' in model_dict.keys()
        assert 'labels' in model_dict.keys()
        assert 'train_list' in model_dict.keys()

        model_dict2 = parse_model_results(model_fit, sort_labels_by_usage=True)
        assert 'keys' in model_dict2.keys()
        assert 'labels' in model_dict2.keys()
        assert 'train_list' in model_dict2.keys()

        model_dict3 = parse_model_results(model_fit, sort_labels_by_usage=True, map_uuid_to_keys=True)
        assert 'keys' in model_dict3.keys()
        assert 'labels' in model_dict3.keys()
        assert 'train_list' in model_dict3.keys()

        labels, _ = relabel_by_usage(model_dict['labels'])
        np.testing.assert_array_equal(model_dict2['labels'], labels)
        np.testing.assert_array_equal(model_dict2['labels'], list(model_dict3['labels'].values()))

    def test_relabel_by_usage(self):
        labels = dict(
                      session1=np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 5, 5, 5]),
                      session2=np.array([4, 4, 4, 4, 4, 1, 1, 1, 2, 2, 2, 5, 5, 5, 5])
                      )
        rel, ordering = relabel_by_usage(labels)

        actual_labels = []
        for arr in rel.values():
            subarr = []
            for i in arr:
                actual = ordering[i]
                subarr.append(actual)
            actual_labels.append(subarr)

        np.testing.assert_array_equal(actual_labels, list(labels.values()))

    def test_normalize_pcs(self):
        index_file = 'data/test_index.yaml'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'

        pca_scores = h5_to_dict(index_data['pca_path'], 'scores')
        norm = normalize_pcs(pca_scores)

        print(pca_scores, norm)
        assert pca_scores.keys() == norm.keys()
        assert pca_scores.values() != norm.values()

        norm_scores = deepcopy(pca_scores)
        all_values = np.concatenate(list(norm_scores.values()), axis=0)
        mu = np.nanmean(all_values, axis=0)
        sig = np.nanstd(all_values, axis=0)
        for k, v in norm_scores.items():
            norm_scores[k] = (v - mu) / sig

        assert norm_scores.keys() == norm.keys()
        assert np.all(np.not_equal(list(norm_scores.values()), list(norm.values())))

        norm2 = normalize_pcs(pca_scores, 'zscore')
        norm_scores = deepcopy(pca_scores)
        all_values = np.concatenate(list(norm_scores.values()), axis=0)
        mu = np.nanmean(all_values, axis=0)
        for k, v in norm_scores.items():
            norm_scores[k] = v - mu

        assert norm_scores.keys() == norm2.keys()
        assert np.all(np.not_equal(list(norm_scores.values()), list(norm2.values())))

        norm3 = normalize_pcs(pca_scores, 'ind-zscore')
        norm_scores = deepcopy(pca_scores)
        for k, v in norm_scores.items():
            norm_scores[k] = (v - np.nanmean(v)) / np.nanstd(v)

        assert norm_scores.keys() == norm3.keys()
        assert np.all(np.not_equal(list(norm_scores.values()), list(norm3.values())))

        norm4 = normalize_pcs(pca_scores, 'demean')
        norm_scores = deepcopy(pca_scores)
        for k, v in norm_scores.items():
            norm_scores[k] = (v - np.nanmean(v)) / np.nanstd(v)

        assert norm_scores.keys() == norm4.keys()
        assert np.all(np.not_equal(list(norm_scores.values()), list(norm4.values())))

    def test_gen_to_arr(self):
        syllable = 2
        labels = [1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2]
        slices = get_mouse_syllable_slices(syllable, labels)
        out = _gen_to_arr(slices)
        assert type(out) == type(np.array([]))

    def test_whiten_all(self):
        index_file = 'data/test_index.yaml'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'

        pca_scores = h5_to_dict(index_data['pca_path'], 'scores')
        whitened_test = _whiten_all(pca_scores)
        assert pca_scores.values() != whitened_test.values()

        whitened_test2 = _whiten_all(pca_scores, center=False)

        assert pca_scores.values() != whitened_test.values() != whitened_test2.values()

    def test_whiten_pcs(self):

        index_file = 'data/test_index.yaml'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'

        pca_scores = h5_to_dict(index_data['pca_path'], 'scores')

        whitened_test = whiten_pcs(pca_scores, 'e')

        assert pca_scores.values() != whitened_test.values()

    def test_retrieve_pcs_from_slices(self):

        index_file = 'data/test_index.yaml'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'

        pca_scores = h5_to_dict(index_data['pca_path'], 'scores')

        pca_scores = normalize_pcs(pca_scores, method='zscore')

        # [(match_idx[i], match_idx[j] + 1), label_uuid, h5]
        slices = [[(23, 32), '5c72bf30-9596-4d4d-ae38-db9a7a28e912', 'path'],
                  [(35, 45), '5c72bf30-9596-4d4d-ae38-db9a7a28e912', 'path'],
                  [(50, 60), '5c72bf30-9596-4d4d-ae38-db9a7a28e912', 'path'],
                  [(100, 130), '5c72bf30-9596-4d4d-ae38-db9a7a28e912', 'path']]

        syllable_matrix = retrieve_pcs_from_slices(slices, pca_scores, max_dur = 30)

        assert syllable_matrix.shape == (100, 30, 10)

    def test_simulate_ar_trajectory(self):
        model_path = 'data/mock_model.p'

        model_data = parse_model_results(model_path)
        ar_mats = np.array(model_data['model_parameters']['ar_mat'])

        sim_mats = simulate_ar_trajectory(ar_mats)
        assert sim_mats.shape == (100, 100)
        assert ar_mats.all() != sim_mats.all()

    def test_get_best_fit(self):
        model_path_1 = 'data/mock_model.p'
        model_path_2 = 'data/test_model.p'
        cp_file = 'data/_pca/changepoints.h5'
        model_data1 = parse_model_results(model_path_1)
        model_data1['changepoints'] = labels_to_changepoints(model_data1['labels'])

        model_data2 = parse_model_results(model_path_2)
        model_data2['labels'] = np.random.random(size=(2, 50))
        model_data2['changepoints'] = labels_to_changepoints(model_data2['labels'])

        model_results = {
                            'model1': model_data1,
                            'model2': model_data2
                         }

        best_model, _ = get_best_fit(cp_file, model_results)
        assert best_model['best model - duration (median match)'] == 'model1'

    def test_make_separate_crowd_movies(self):

        index_file = 'data/test_index_crowd.yaml'
        model_path = 'data/mock_model.p'
        output_dir = 'data/test_movies/'
        config_file = 'data/config.yaml'

        config_data = read_yaml(config_file)

        _, sorted_index = parse_index(index_file)

        model_data = parse_model_results(model_path)

        # Get list of syllable labels for all sessions
        labels = model_data['labels']

        # Relabel syllable labels by usage sorting and save the ordering for crowd-movie file naming
        if config_data.get('sort', True):
            labels, ordering = relabel_by_usage(labels, count=config_data.get('count', 'usage'))
        else:
            ordering = list(range(config_data['max_syllable']))

        # get train list uuids if available, otherwise default to 'keys'
        label_uuids = model_data.get('train_list', model_data['keys'])
        label_dict = dict(zip(label_uuids, labels))

        # Get uuids found in both the labels and the index
        uuid_set = set(label_uuids) & set(sorted_index['files'])
        # Make sure the files exist
        uuid_set = [uuid for uuid in uuid_set if os.path.exists(sorted_index['files'][uuid]['path'][0])]
        # filter to only include existing UUIDs
        sorted_index['files'] = keyfilter(lambda k: k in uuid_set, sorted_index['files'])
        label_dict = keyfilter(lambda k: k in uuid_set, label_dict)

        labels = list(label_dict.values())
        label_uuids = list(label_dict)

        group_keys = groupby(lambda k: sorted_index['files'][k]['group'], label_uuids)

        sorted_index['files'] = {k: v for k, v in sorted_index['files'].items() if k in uuid_set}
        sorted_index['pca_path'] = 'data/test_scores.h5'

        # Setting config_data parameters
        config_data['max_syllable'] = 5
        ordering = range(config_data['max_syllable'])

        config_data['crowd_syllables'] = ordering
        config_data['progress_bar'] = False
        config_data['max_examples'] = 1
        config_data['scale'] = 1
        config_data['legacy_jitter_fix'] = False

        path_dict = make_separate_crowd_movies(config_data, sorted_index, group_keys, label_dict, output_dir, ordering)

        assert len(path_dict['Group1']) == 5
        for value in path_dict['Group1']:
            assert os.path.exists(value)

        shutil.rmtree(output_dir)

if __name__ == '__main__':
    unittest.main()