import numpy as np
from scipy import stats
from copy import deepcopy
from unittest import TestCase
from moseq2_model.util import load_pcs
from moseq2_model.train.models import ARHMM
from moseq2_model.helpers.data import prepare_model_metadata, get_training_data_splits
from moseq2_model.train.util import (
    train_model,
    get_labels_from_model,
    whiten_all,
    whiten_each,
    run_e_step,
    zscore_all,
    zscore_each,
    get_crosslikes,
)
from moseq2_model.train.backend import (
    CorrectedFastARHMM,
    CorrectedFastARHMMSeparateTrans,
)
from moseq2_model._yaml import read_yaml


def get_model(separate_trans=False, robust=False, groups=[]):
    input_file = "data/_pca/pca_scores.h5"
    config_file = "data/config.yaml"

    with open(config_file, "r") as f:
        config_data = read_yaml(f)

    config_data["separate_trans"] = separate_trans
    config_data["robust"] = robust
    nkeys = 5
    groups = ["key1", "key2", "key3", "key4", "key5"]

    data_dict, data_metadata = load_pcs(
        filename=input_file, var_name="scores", npcs=10, load_groups=True
    )
    data_metadata["groups"] = {k: g for k, g in zip(data_dict, groups)}

    data_dict, model_parameters, _, _, whitening_parameters = prepare_model_metadata(
        data_dict, data_metadata, config_data
    )

    arhmm = ARHMM(data_dict=data_dict, **model_parameters)

    return arhmm, data_dict


class TestTrainUtils(TestCase):

    def test_train_model(self):
        config_file = "data/config.yaml"
        with open(config_file, "r") as f:
            config_data = read_yaml(f)
            config_data["percent_split"] = 1

        model, data_dict = get_model()

        X, whitening_parameters = whiten_all(data_dict)
        training_data, validation_data = get_training_data_splits(
            config_data["percent_split"] / 100, X
        )

        model, lls, labels, iter_lls, iter_holls, _ = train_model(
            model, num_iter=5, train_data=training_data, val_data=validation_data
        )
        assert isinstance(model, CorrectedFastARHMM)
        assert isinstance(lls, float)
        assert len(labels) == 2
        assert len(labels[0]) == 906
        assert len(iter_lls) == 0
        assert len(iter_holls) == 0

        model, lls, labels, iter_lls, iter_holls, _ = train_model(
            model,
            num_iter=5,
            train_data=training_data,
            val_data=validation_data,
            verbose=True,
        )
        assert isinstance(model, CorrectedFastARHMM)
        assert isinstance(lls, float)
        assert len(labels) == 2
        assert len(labels[0]) == 906
        assert len(iter_lls) == 2
        assert len(iter_holls) == 2

        model, data_dict = get_model(separate_trans=True, groups=["default", "Group1"])

        X, whitening_parameters = whiten_all(data_dict)
        training_data, validation_data = get_training_data_splits(
            config_data["percent_split"] / 100, X
        )

        model, lls, labels, iter_lls, iter_holls, _ = train_model(
            model,
            num_iter=5,
            train_data=training_data,
            val_data=validation_data,
            separate_trans=True,
            groups=["default", "Group1"],
            check_every=1,
            verbose=True,
        )
        assert isinstance(model, CorrectedFastARHMMSeparateTrans)
        assert isinstance(lls, float)
        assert len(labels) == 2
        assert len(labels[0]) == 906
        assert len(iter_lls) == 5
        assert len(iter_holls) == 5

    def test_get_labels_from_model(self):

        config_file = "data/config.yaml"

        with open(config_file, "r") as f:
            config_data = read_yaml(f)

        model, data_dict = get_model()

        X, whitening_parameters = whiten_all(data_dict)
        training_data, validation_data = get_training_data_splits(
            config_data["percent_split"] / 100, X
        )

        model, lls, labels, iter_lls, iter_holls, _ = train_model(
            model, num_iter=5, train_data=training_data, val_data=validation_data
        )

        labels = get_labels_from_model(model)
        print(labels)
        assert len(labels[0]) == 906

    def test_whiten_all(self):

        _, data_dict = get_model()

        whitened_a, whitening_parameters = whiten_all(data_dict)
        whitened_e, whitening_parameters = whiten_each(data_dict)
        assert data_dict.values() != whitened_a.values()
        assert whitened_a.values() != whitened_e.values()

    def test_whiten_each(self):
        _, data_dict = get_model()

        whitened_a, whitening_parameters = whiten_all(data_dict, center=False)
        whitened_e, whitening_parameters = whiten_each(data_dict, center=False)
        assert data_dict.values() != whitened_a.values()
        assert whitened_a.values() != whitened_e.values()

    def test_run_estep(self):
        model, _ = get_model()
        ex_states = run_e_step(model)
        assert len(ex_states) == 2
        assert len(ex_states[0]) == 903  # 908 - 3 nlag frames

    def test_zscore_each(self):

        _, data_dict = get_model()
        test_result = zscore_each(deepcopy(data_dict))

        for k, v in data_dict.items():
            valid_truth_scores = v[~np.isnan(v).any(axis=1), :10]
            truth = stats.zscore(valid_truth_scores)

            valid_test_scores = test_result[k][
                ~np.isnan(test_result[k]).any(axis=1), :10
            ]
            assert np.allclose(truth, valid_test_scores)

    def test_zscore_all(self):
        npcs = 10
        _, data_dict = get_model()
        test_result = zscore_all(deepcopy(data_dict), npcs=npcs)

        for k, v in test_result.items():
            valid_truth_scores = data_dict[k][~np.isnan(data_dict[k]).any(axis=1), :10]
            truth = stats.zscore(valid_truth_scores)

            valid_test_scores = v[~np.isnan(v).any(axis=1), :10]
            assert not np.allclose(truth, valid_test_scores)

        valid_scores = np.concatenate(
            [x[~np.isnan(x).any(axis=1), :npcs] for x in data_dict.values()]
        )

        all_zscored = stats.zscore(valid_scores)
        test_valid_scores = np.concatenate(
            [x[~np.isnan(x).any(axis=1), :npcs] for x in test_result.values()]
        )
        assert np.allclose(all_zscored, test_valid_scores)

    def test_get_crosslikes(self):

        arhmm, _ = get_model()

        all_CLs, CL = get_crosslikes(arhmm)

        assert len(all_CLs.keys()) == 10000
        assert CL.shape == (100, 100)

        all_CLs2, CL2 = get_crosslikes(arhmm, frame_by_frame=True)

        assert len(all_CLs2.keys()) == 10000
        assert CL2.shape == (100, 100)

        self.assertRaises(AssertionError, np.testing.assert_equal, CL, CL2)
