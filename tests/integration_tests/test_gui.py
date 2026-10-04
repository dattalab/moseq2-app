import os
import sys
import shutil
import joblib
from unittest import TestCase
from moseq2_model.gui import learn_model_command
from moseq2_model._yaml import read_yaml, write_yaml


class TestGUI(TestCase):

    def test_learn_model(self):

        input_file = "data/_pca/pca_scores.h5"

        model_path = "data/test_model/model.p"
        base_model_path = "data/test_model/"
        config_file = "data/test_config.yaml"
        index = "data/test_index.yaml"
        checkpoint_path = "data/test_model/checkpoints/"

        # test space-separated input
        stdin = "data/stdin.txt"
        with open(stdin, "w") as f:
            f.write("default Group1")

        sys.stdin = open(stdin)

        progress_paths = {
            "scores_path": input_file,
            "model_path": model_path,
            "base_model_path": base_model_path,
            "config_file": config_file,
            "index_file": index,
        }

        # create test config file to update
        shutil.copyfile("data/config.yaml", config_file)

        with open(config_file, "r") as f:
            config_data = read_yaml(f)

        # adding required run parameters to config file
        config_data["hold_out"] = True
        config_data["hold_out_seed"] = -1
        config_data["nfolds"] = 2
        config_data["num_iter"] = 10
        config_data["max_states"] = 100
        config_data["npcs"] = 10
        config_data["kappa"] = None
        config_data["separate_trans"] = True
        config_data["robust"] = True
        config_data["checkpoint_freq"] = 2
        config_data["percent_split"] = 1
        config_data["out_script"] = "train_out.sh"

        with open(config_file, "w") as f:
            write_yaml(config_data, f)

        _ = learn_model_command(progress_paths, verbose=True, get_cmd=True)

        assert os.path.exists(
            model_path
        ), "Trained model file was not created or is in the incorrect location"
        assert os.path.exists(checkpoint_path), "Checkpoints were not created"
        assert len(os.listdir(checkpoint_path)) == 5  # iters: 1, 3, 5, 7, 9

        assert os.path.exists("data/test_model/train_heldout_summary.png")
        os.remove("data/test_model/train_heldout_summary.png")
        os.rename(model_path, "data/original_model.p")
        shutil.rmtree(checkpoint_path)

        # test space-separated input
        with open(stdin, "w") as f:
            f.write("default Group1")

        sys.stdin = open(stdin)

        with open(config_file, "r") as f:
            config_data = read_yaml(f)

        config_data["hold_out_seed"] = 1000
        config_data["ncpus"] = 100

        with open(config_file, "w") as f:
            write_yaml(config_data, f)

        _ = learn_model_command(progress_paths, verbose=True, get_cmd=True)

        assert os.path.exists(
            model_path
        ), "Updated model file was not created or is in the incorrect location"

        assert os.path.exists("data/test_model/train_heldout_summary.png")
        os.remove("data/test_model/train_heldout_summary.png")

        model1 = joblib.load("data/original_model.p")
        model2 = joblib.load(model_path)

        assert model1 != model2

        os.remove("data/original_model.p")
        os.remove(model_path)
        os.remove(stdin)
        os.remove(config_file)

    def test_kappa_scan(self):

        input_file = "data/_pca/pca_scores.h5"
        model_path = "data/models/model.p"
        base_model_path = "data/models/"
        config_file = "data/test_config.yaml"
        index = "data/test_index.yaml"
        out_script = "train_out.sh"

        # test space-separated input
        stdin = "data/stdin.txt"
        with open(stdin, "w") as f:
            f.write("default Group1")

        sys.stdin = open(stdin)

        progress_paths = {
            "scores_path": input_file,
            "model_path": model_path,
            "base_model_path": base_model_path,
            "config_file": config_file,
            "index_file": index,
        }

        # create test config file to update
        shutil.copyfile("data/config.yaml", config_file)

        with open(config_file, "r") as f:
            config_data = read_yaml(f)

        # adding required run parameters to config file
        config_data["hold_out"] = True
        config_data["nfolds"] = 2
        config_data["num_iter"] = 10
        config_data["max_states"] = 100
        config_data["npcs"] = 10
        config_data["separate_trans"] = True
        config_data["robust"] = True
        config_data["percent_split"] = 20

        config_data["scan_scale"] = "log"
        config_data["min_kappa"] = 1e3
        config_data["max_kappa"] = None
        config_data["kappa"] = "scan"
        config_data["n_models"] = 1
        config_data["out_script"] = out_script
        config_data["run_cmd"] = False

        with open(config_file, "w") as f:
            write_yaml(config_data, f)

        _ = learn_model_command(progress_paths, verbose=True, get_cmd=True)

        assert os.path.exists(os.path.join(base_model_path, out_script))
        os.remove(os.path.join(base_model_path, out_script))
        os.remove(config_file)
