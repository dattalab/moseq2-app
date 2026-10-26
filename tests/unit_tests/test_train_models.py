import numpy as np
from unittest import TestCase
from moseq2_model.util import load_pcs
from moseq2_model.helpers.data import prepare_model_metadata
from moseq2_model.train.models import ARHMM, _get_empirical_ar_params
from moseq2_model.train.backend import (
    CorrectedFastARHMM,
    CorrectedFastARHMMSeparateTrans,
    CorrectedRobustARHMM,
    CorrectedRobustARHMMSeparateTrans,
)
from moseq2_model._yaml import read_yaml


class TestTrainModels(TestCase):

    def test_get_empirical_ar_params(self):
        input_file = "data/_pca/pca_scores.h5"

        data_dict, data_metadata = load_pcs(
            filename=input_file, var_name="scores", npcs=10, load_groups=True
        )

        data_dim = list(data_dict.values())[0].shape[1]
        S_0_scale = 0.01
        K_0_scale = 10.0
        affine = True
        nlags = 3

        params = {
            "nu_0": data_dim + 2,
            "S_0": S_0_scale * np.eye(data_dim),
            "M_0": np.hstack(
                (
                    np.eye(data_dim),
                    np.zeros((data_dim, data_dim * (nlags - 1))),
                    np.zeros((data_dim, int(affine))),
                )
            ),
            "affine": affine,
            "K_0": K_0_scale * np.eye(data_dim * nlags + affine),
        }

        obs_params = dict(
            nu_0=params["nu_0"],
            S_0=params["S_0"],
            M_0=params["M_0"],
            K_0=params["K_0"],
            affine=params["affine"],
        )

        new_params = _get_empirical_ar_params(list(data_dict.values()), obs_params, nlags)
        assert len(new_params["S_0"]) == 10  # npcs
        assert len(new_params["K_0"]) == len(params["K_0"])
        assert len(new_params["M_0"]) == len(params["M_0"])
        assert new_params["affine"] == params["affine"]

    def test_ARHMM(self):
        input_file = "data/_pca/pca_scores.h5"
        config_file = "data/config.yaml"

        with open(config_file, "r") as f:
            config_data = read_yaml(f)

        data_dict, data_metadata = load_pcs(
            filename=input_file, var_name="scores", npcs=10, load_groups=True
        )

        nkeys = 5
        all_keys = ["key1", "key2", "key3", "key4", "key5"]

        data_dict, model_parameters, train_list, hold_out_list, whitening_parameters = (
            prepare_model_metadata(data_dict, data_metadata, config_data)
        )

        model_parameters["separate_trans"] = False
        model_parameters["robust"] = False
        arhmm = ARHMM(data_dict=data_dict, **model_parameters)
        assert isinstance(arhmm, CorrectedFastARHMM)

        model_parameters["sticky_init"] = True
        arhmm = ARHMM(data_dict=data_dict, **model_parameters)
        assert isinstance(arhmm, CorrectedFastARHMM)
        model_parameters["sticky_init"] = False

        model_parameters["separate_trans"] = True
        model_parameters["robust"] = False
        model_parameters["groups"] = {k: f"group{i}" for i, k in enumerate(data_dict)}
        arhmm = ARHMM(data_dict=data_dict, **model_parameters)
        assert isinstance(arhmm, CorrectedFastARHMMSeparateTrans)

        model_parameters["separate_trans"] = False
        model_parameters["robust"] = True
        arhmm = ARHMM(data_dict=data_dict, **model_parameters)
        assert isinstance(arhmm, CorrectedRobustARHMM)

        model_parameters["separate_trans"] = True
        model_parameters["robust"] = True
        model_parameters["groups"] = {k: f"group{i}" for i, k in enumerate(data_dict)}
        arhmm = ARHMM(data_dict=data_dict, **model_parameters)
        assert isinstance(arhmm, CorrectedRobustARHMMSeparateTrans)
