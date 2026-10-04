import os
import sys
from os.path import dirname
from unittest import TestCase
from moseq2_model.util import load_pcs
from moseq2_model.helpers.data import (
    process_indexfile,
    select_data_to_model,
    prepare_model_metadata,
    get_heldout_data_splits,
    get_training_data_splits,
    graph_modeling_loglikelihoods,
)
from moseq2_model._yaml import read_yaml


class TestDataHelpers(TestCase):

    def test_process_indexfile(self):

        input_file = "data/_pca/pca_scores.h5"
        index_path = "data/test_index.yaml"
        config_file = "data/config.yaml"

        data_dict, data_metadata = load_pcs(
            filename=input_file, var_name="scores", npcs=10, load_groups=True
        )

        with open(config_file, "r") as f:
            config_data = read_yaml(f)

        index_data, data_metadata = process_indexfile(
            index_path, config_data, data_metadata
        )
        assert len(index_data["files"]) == len(
            data_metadata["groups"]
        ), "Number of input files != number of uuids in the index file"

    def test_select_data_to_model(self):
        input_file = "data/_pca/pca_scores.h5"
        index_path = "data/test_index.yaml"

        data_dict, data_metadata = load_pcs(
            filename=input_file, var_name="scores", npcs=10, load_groups=True
        )

        with open(index_path, "r") as f:
            index_data = read_yaml(f)

        # test simple CLI case
        data_dict, data_metadata = select_data_to_model(
            index_data, data_dict, data_metadata
        )
        assert len(data_dict) == len(
            data_metadata["groups"]
        ), "Number of groups != number of uuids"

        # test single GUI input
        index_data["files"][1]["group"] = "default"
        stdin = "data/stdin.txt"
        with open(stdin, "w") as f:
            f.write("default")

        sys.stdin = open(stdin)
        t_data_dict, data_metadata = select_data_to_model(
            index_data, data_dict, data_metadata, select_groups=True
        )

        assert (
            len(t_data_dict) == len(data_metadata["groups"]) == 1
        ), "index data was incorrectly parsed"
        assert (
            list(data_metadata["groups"].values())[0] == "default"
        ), "groups were returned incorrectly"

    def test_prepare_model_metadata(self):

        input_file = "data/_pca/pca_scores.h5"
        config_file = "data/config.yaml"

        data_dict, data_metadata = load_pcs(
            filename=input_file, var_name="scores", npcs=10, load_groups=True
        )

        with open(config_file, "r") as f:
            config_data = read_yaml(f)

        index_path = "data/test_index.yaml"

        with open(index_path, "r") as f:
            index_data = read_yaml(f)

        data_dict, data_metadata = select_data_to_model(
            index_data, data_dict, data_metadata
        )

        (
            data_dict1,
            model_parameters,
            train_list,
            hold_out_list,
            whitening_parameters,
        ) = prepare_model_metadata(data_dict, data_metadata, config_data)

        assert (
            data_dict.values() != data_dict1.values()
        ), "Index loaded uuids and training data does not match scores file"
        assert train_list == list(
            data_dict1
        ), "Loaded uuids do not match total number of uuids"
        assert hold_out_list == [], "Some of the data is unintentionally held out"

        config_data["whiten"] = "each"
        config_data["noise_level"] = 1
        (
            data_dict1,
            model_parameters,
            train_list,
            hold_out_list,
            whitening_parameters,
        ) = prepare_model_metadata(data_dict, data_metadata, config_data)

        assert (
            data_dict.values() != data_dict1.values()
        ), "Index loaded uuids and training data does not match scores file"
        assert train_list == list(
            data_dict1
        ), "Loaded uuids do not match total number of uuids"
        assert hold_out_list == [], "Some of the data is unintentionally held out"

        config_data["whiten"] = "none"
        (
            data_dict1,
            model_parameters,
            train_list,
            hold_out_list,
            whitening_parameters,
        ) = prepare_model_metadata(data_dict, data_metadata, config_data)

        assert (
            data_dict.values() != data_dict1.values()
        ), "Index loaded uuids and training data does not match scores file"
        assert train_list == list(
            data_dict1
        ), "Loaded uuids do not match total number of uuids"
        assert hold_out_list == [], "Some of the data is unintentionally held out"

    def test_get_heldout_data_splits(self):
        input_file = "data/_pca/pca_scores.h5"
        config_file = "data/config.yaml"

        data_dict, data_metadata = load_pcs(
            filename=input_file, var_name="scores", npcs=10, load_groups=True
        )

        with open(config_file, "r") as f:
            config_data = read_yaml(f)
            config_data["hold_out"] = True
            config_data["nfolds"] = 2

        index_path = "data/test_index.yaml"

        with open(index_path, "r") as f:
            index_data = read_yaml(f)

        data_dict, data_metadata = select_data_to_model(
            index_data, data_dict, data_metadata
        )

        data_dict, model_parameters, train_list, hold_out_list, whitening_parameters = (
            prepare_model_metadata(data_dict, data_metadata, config_data)
        )

        assert sorted(train_list) != sorted(
            hold_out_list
        ), "Training data is the same as held out data"
        assert len(train_list) == len(
            hold_out_list
        ), "Number of held out sets is incorrect, supposed to be 1"

        train_data, test_data = get_heldout_data_splits(
            data_dict, train_list, hold_out_list
        )

        assert (
            train_data is not None and test_data is not None
        ), "There are missing datasets"

    def test_get_training_data_splits(self):

        input_file = "data/_pca/pca_scores.h5"
        config_file = "data/config.yaml"

        data_dict, data_metadata = load_pcs(
            filename=input_file, var_name="scores", npcs=10, load_groups=True
        )

        with open(config_file, "r") as f:
            config_data = read_yaml(f)
            config_data["hold_out"] = False
            config_data["percent_split"] = 50

        training_data, validation_data = get_training_data_splits(
            config_data["percent_split"] / 100, data_dict
        )
        for k in training_data:
            assert len(list(training_data[k])) == len(
                list(validation_data[k])
            ), "Data split is incorrect"

        assert len(list(training_data.keys())) == len(
            list(validation_data.keys())
        ), "Training data and Val data do not contain same keys"

        val_frames = sum(map(len, validation_data.values()))
        total_frames = sum(map(len, training_data.values())) + val_frames
        percent_out = int(val_frames / total_frames * 100)

        assert (
            percent_out == config_data["percent_split"]
        ), "Config file was not correctly updated"

    def test_graph_modeling_loglikelihoods(self):
        dest_file = "data/_pca/pca_scores.h5"
        config_file = "data/config.yaml"

        iter_lls = [12, 15, 19]
        iter_holls = [2, 5, 9]

        with open(config_file, "r") as f:
            config_data = read_yaml(f)
            config_data["hold_out"] = True
            config_data["verbose"] = True

        img_path = graph_modeling_loglikelihoods(
            config_data, iter_lls, iter_holls, dirname(dest_file)
        )

        assert os.path.exists(img_path), "Something went wrong; graph was not created."
        os.remove(img_path)

        iter_lls = [[12, 15, 19], [15, 16, 18]]
        iter_holls = [[2, 3, 8], [5, 5, 9]]

        with open(config_file, "r") as f:
            config_data = read_yaml(f)
            config_data["hold_out"] = True
            config_data["verbose"] = True

        img_path = graph_modeling_loglikelihoods(
            config_data, iter_lls, iter_holls, dirname(dest_file)
        )

        assert os.path.exists(img_path), "Something went wrong; graph was not created."
        os.remove(img_path)
