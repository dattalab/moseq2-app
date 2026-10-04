import os
import sys
import h5py
import numpy as np
from unittest import TestCase
from tests.unit_tests.test_train_utils import get_model
from moseq2_model.train.util import whiten_all, train_model
from moseq2_model.helpers.data import get_training_data_splits
from moseq2_model.util import (
    load_data_from_matlab,
    load_cell_string_from_matlab,
    load_pcs,
    save_dict,
    dict_to_h5,
    h5_to_dict,
    _load_h5_to_dict,
    copy_model,
    get_parameters_from_model,
    count_frames,
    get_parameter_strings,
    create_command_strings,
    get_scan_range_kappas,
)
from moseq2_model._yaml import read_yaml


class TestUtils(TestCase):

    def test_count_frames(self):
        input_data = "data/_pca/pca_scores.h5"
        data_dict, data_metadata = load_pcs(
            input_data, var_name="scores", load_groups=True
        )

        nframes = count_frames(data_dict)

        assert nframes == 1800

    def test_get_parameter_strings(self):

        index = "data/test_index.yaml"
        config_data = {
            "index": index,
            "npcs": 10,
            "num_iter": 100,
            "separate_trans": True,
            "robust": True,
            "e_step": True,
            "hold_out": True,
            "nfolds": 2,
            "max_states": 100,
            "converge": True,
            "tolerance": 1000,
            "cluster_type": "slurm",
            "ncpus": 1,
            "memory": "10GB",
            "partition": "short",
            "wall_time": "01:00:00",
            "prefix": '"',
        }

        parameters, prefix = get_parameter_strings(config_data)

        truth_str = f" --npcs 10 --num-iter 100 -i data/test_index.yaml --separate-trans --robust --e-step --hold-out --nfolds 2 --max-states 100 --ncpus 1 "
        truth_prefix = 'sbatch -c 1 --mem=10GB -p short -t 01:00:00 --wrap ""'

        assert parameters == truth_str
        assert prefix == truth_prefix

    def test_create_command_strings(self):
        input_file = "data/_pca/pca_scores.h5"
        index_file = "data/test_index.yaml"
        output_dir = "data/models/"
        kappas = [10]

        config_data = {
            "index": index_file,
            "npcs": 10,
            "num_iter": 100,
            "separate_trans": True,
            "robust": True,
            "e_step": True,
            "hold_out": True,
            "nfolds": 2,
            "max_states": 100,
            "converge": True,
            "tolerance": 1000,
            "cluster_type": "slurm",
            "ncpus": 1,
            "memory": "10GB",
            "partition": "short",
            "wall_time": "01:00:00",
            "prefix": "",
        }

        command_string = create_command_strings(
            input_file,
            output_dir,
            config_data,
            kappas,
            model_name_format="model-{}-{}.p",
        )
        print(command_string)

        truth_output = (
            "set -e\nsbatch -c 1 --mem=10GB -p short -t 01:00:00 --wrap "
            '"moseq2-model learn-model data/_pca/pca_scores.h5 data/models/model-0-10.p --npcs 10 '
            "--num-iter 100 -i data/test_index.yaml --separate-trans --robust --e-step --hold-out --nfolds 2 "
            '--max-states 100 --ncpus 1 --kappa 10"'
        )
        assert command_string == truth_output

    def test_get_scan_range_kappas(self):

        input_data = "data/_pca/pca_scores.h5"
        data_dict, data_metadata = load_pcs(
            input_data, var_name="scores", load_groups=True
        )

        config_data = {"min_kappa": None, "max_kappa": None, "n_models": 10}

        test_kappas = get_scan_range_kappas(data_dict, config_data)

        # For nframes == 1800
        assert len(test_kappas) == 10
        assert min(test_kappas) == 10
        assert max(test_kappas) == 1e5

        config_data = {"min_kappa": None, "max_kappa": 1e6, "n_models": 10}

        test_kappas = get_scan_range_kappas(data_dict, config_data)
        # For nframes == 1800
        assert len(test_kappas) == 10
        assert min(test_kappas) == 10
        assert max(test_kappas) == 1e6

        config_data = {"min_kappa": 1, "max_kappa": None, "n_models": 10}

        test_kappas = get_scan_range_kappas(data_dict, config_data)
        # For nframes == 1800
        assert len(test_kappas) == 10
        assert min(test_kappas) == 1
        assert max(test_kappas) == 1e5

        config_data = {"min_kappa": 1e3, "max_kappa": 1e5, "n_models": 10}

        test_kappas = get_scan_range_kappas(data_dict, config_data)
        # For nframes == 1800
        assert len(test_kappas) == 10
        assert min(test_kappas) == 1e3
        assert max(test_kappas) == 1e5

        config_data = {
            "scan_scale": "linear",
            "min_kappa": None,
            "max_kappa": None,
            "n_models": 10,
        }

        test_kappas = get_scan_range_kappas(data_dict, config_data)

        # For nframes == 1800
        assert len(test_kappas) == 10
        assert min(test_kappas) == 18
        assert max(test_kappas) == 180000

        config_data = {
            "scan_scale": "linear",
            "min_kappa": 3,
            "max_kappa": None,
            "n_models": 10,
        }

        test_kappas = get_scan_range_kappas(data_dict, config_data)

        # For nframes == 1800
        assert len(test_kappas) == 10
        assert min(test_kappas) == 3
        assert max(test_kappas) == 180000

        config_data = {
            "scan_scale": "linear",
            "min_kappa": None,
            "max_kappa": 60000,
            "n_models": 10,
        }

        test_kappas = get_scan_range_kappas(data_dict, config_data)

        # For nframes == 1800
        assert len(test_kappas) == 10
        assert min(test_kappas) == 18
        assert max(test_kappas) == 60000

        config_data = {
            "scan_scale": "linear",
            "min_kappa": 2,
            "max_kappa": 4,
            "n_models": 10,
        }

        test_kappas = get_scan_range_kappas(data_dict, config_data)

        # For nframes == 1800
        assert len(test_kappas) == 10
        assert min(test_kappas) == 2
        assert max(test_kappas) == 4

    def test_load_pcs(self):

        # first test loading dummy matlab file, then pickle, then h5
        pcs, metadata = load_pcs("data/dummy_matlab.mat", load_groups=True)

        assert len(pcs) == 1
        assert len(metadata["groups"]) == 1
        assert np.all(pcs[0] == 1)
        assert list(metadata["groups"].values())[0] == "test"

        input_data = "data/_pca/pca_scores.h5"
        data_dict, data_metadata = load_pcs(
            input_data, var_name="scores", load_groups=True
        )

        assert list(data_dict.keys()) == data_metadata["uuids"]

        data_dict, data_metadata = load_pcs(
            input_data, var_name="scores", load_groups=True
        )

        assert list(data_dict.keys()) == data_metadata["uuids"]

        outfile = "data/saved_dict.p"
        save_dict(outfile, data_dict)

        data_dict, data_metadata = load_pcs(
            outfile, var_name="scores", load_groups=True
        )

        assert list(data_dict.keys()) == data_metadata["uuids"]
        os.remove(outfile)

    def test_save_dict(self):
        input_data = "data/_pca/pca_scores.h5"
        data_dict, data_metadata = load_pcs(
            input_data, var_name="scores", load_groups=True
        )

        outfile = "data/saved_dict.pkl"
        save_dict(outfile, data_dict)

        assert os.path.exists(outfile)
        os.remove(outfile)

        outfile = "data/saved_dict.h5"
        save_dict(outfile, data_dict)

        assert os.path.exists(outfile)
        os.remove(outfile)

        outfile = "data/saved_dict.z"
        save_dict(outfile, data_dict)

        assert os.path.exists(outfile)
        os.remove(outfile)

        outfile = "data/saved_dict.mat"
        save_dict(outfile, data_dict)

        assert os.path.exists(outfile)
        os.remove(outfile)

    def test_h5_to_dict(self):
        input_data = "data/_pca/pca_scores.h5"
        outdict = h5_to_dict(input_data, "scores")
        assert isinstance(outdict, dict)

    def test_dict_to_h5(self):
        input_data = "data/_pca/pca_scores.h5"
        data_dict, data_metadata = load_pcs(
            input_data, var_name="scores", load_groups=True
        )
        assert isinstance(data_dict, dict)

        outpath = "data/out_scores.h5"

        with h5py.File(outpath, "w") as tmp_h5:
            dict_to_h5(tmp_h5, data_dict)

        assert os.path.exists(outpath)
        os.remove(outpath)

        outpath = "data/out_scores.h5"
        test_dict = {
            "int-test": 1,
            "float": 1.12432,
            "list": [1, 2, 3, 4],
            "np": np.array([[[1], [2], [3]], [[4], [5], [6]]]).astype(object),
            "dict": {"test": 12},
            12: "int key test",
            (1, 2): "tuple key test",
        }

        with h5py.File(outpath, "w") as tmp_h5:
            dict_to_h5(tmp_h5, test_dict)

        assert os.path.exists(outpath)
        os.remove(outpath)

        test_dict = {"tup": (1, 2, 3)}

        tmp_h5 = h5py.File(outpath, "w")
        self.assertRaises(ValueError, dict_to_h5, tmp_h5, test_dict)

    def test_load_h5_to_dict(self):
        input_data = "data/_pca/pca_scores.h5"
        with h5py.File(input_data, "r") as f:
            outdict = _load_h5_to_dict(f, "/")
        assert isinstance(outdict, dict)

    def test_copy_model(self):
        model, data_dict = get_model()
        config_file = "data/config.yaml"

        with open(config_file, "r") as f:
            config_data = read_yaml(f)

        X, whitening_parameters = whiten_all(data_dict)
        training_data, validation_data = get_training_data_splits(
            config_data["percent_split"] / 100, X
        )

        model, lls, labels, iter_lls, iter_holls, _ = train_model(
            model, num_iter=5, train_data=training_data, val_data=validation_data
        )

        cp = copy_model(model)
        assert sys.getsizeof(model) == sys.getsizeof(cp)

    def test_get_parameters_from_model(self):

        def check_params(model, params):
            trans_dist = model.trans_distn
            assert params["kappa"] == trans_dist.kappa
            assert params["gamma"] == trans_dist.gamma
            assert params["max_states"] == trans_dist.N
            assert params["nlags"] == model.nlags
            assert params["ar_mat"] == [obs.A for obs in model.obs_distns]
            assert params["sig"] == [obs.sigma for obs in model.obs_distns]

        model, data_dict = get_model()
        config_file = "data/config.yaml"

        with open(config_file, "r") as f:
            config_data = read_yaml(f)

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
        )

        params = get_parameters_from_model(model)
        check_params(model, params)

        model, lls, labels, iter_lls, iter_holls, _ = train_model(
            model,
            num_iter=5,
            train_data=training_data,
            val_data=validation_data,
            separate_trans=False,
        )

        params = get_parameters_from_model(model)
        check_params(model, params)

    def test_load_matlab_data(self):

        pcs = load_data_from_matlab("data/dummy_matlab.mat", var_name="features")
        keys = list(pcs.keys())

        assert len(keys) == 1
        assert keys[0] == 0
        assert np.all(pcs[0] == 1)

    def test_load_cell_string_from_matlab(self):

        groups = load_cell_string_from_matlab(
            "data/dummy_matlab.mat", var_name="groups"
        )

        assert len(groups) == 1
        assert groups[0] == "test"
