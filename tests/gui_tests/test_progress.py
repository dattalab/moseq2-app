import os
import shutil
from copy import deepcopy
import ruamel.yaml as yaml
from moseq2_viz.util import read_yaml
from unittest import TestCase
from os.path import exists, join
from moseq2_extract.helpers.wrappers import extract_wrapper
from moseq2_app.gui.progress import generate_missing_metadata, get_session_paths, update_progress, \
    restore_progress_vars, get_pca_progress, load_progress, \
    get_extraction_progress, print_progress, check_progress, find_progress, generate_intital_progressfile


class TestNotebookProgress(TestCase):

    def setUp(self):

        if exists('data/session_config.yaml'):
            os.remove('data/session_config.yaml')

        self.base_dir = 'data/'
        self.base_progress_vars = {'base_dir': self.base_dir,
                              'config_file': '',
                              'index_file': '',
                              'train_data_dir': '',
                              'pca_dirname': '',
                              'scores_filename': '',
                              'scores_path': '',
                              'changepoints_path': '',
                              'model_path': '',
                              'crowd_dir': '',
                              'syll_info': '',
                              'plot_path': join(self.base_dir, 'plots/'),
                              'snapshot': 'test'}

    def test_generate_missing_metadata(self):

        outdir = './data/test_session/'
        sess_name = 'test'

        if exists(join(outdir, 'metadata.json')):
            os.remove(join(outdir, 'metadata.json'))

        generate_missing_metadata(outdir, sess_name)
        assert exists(join(outdir, 'metadata.json'))

    def test_get_session_paths(self):

        data_dir = 'data/'
        paths = get_session_paths(data_dir)

        assert len(paths.keys()) == 2

        paths = get_session_paths(data_dir, extracted=True)

        assert len(paths.keys()) == 0

    def test_update_progress(self):

        base_dir = 'data/'
        progress_file = join(base_dir, 'progress.yaml')

        with open(progress_file, 'w') as f:
            yaml.safe_dump(self.base_progress_vars, f)

        new_prog = update_progress(progress_file, 'config_file', 'test_path')
        assert new_prog['config_file'] == 'test_path'

        read_progs = read_yaml(progress_file)

        assert read_progs != self.base_progress_vars

        os.remove(progress_file)

    def test_find_progress(self):
        base_dir = 'data/'
        progress_file = join(base_dir, 'progress.yaml')

        with open(progress_file, 'w') as f:
            yaml.safe_dump(self.base_progress_vars, f)

        new_prog = find_progress(deepcopy(self.base_progress_vars))

        assert new_prog != self.base_progress_vars

        config_data = read_yaml('data/config.yaml')

        with open('data/session_config.yaml', 'w') as f:
            yaml.safe_dump(config_data, f)

        new_prog2 = find_progress(deepcopy(self.base_progress_vars))
        assert new_prog != new_prog2

        os.remove('data/session_config.yaml')

    def test_generate_initial_progressfile(self):

        loaded_prog = generate_intital_progressfile('data/progress.yaml')

        assert loaded_prog != self.base_progress_vars

        find_progress(deepcopy(loaded_prog))

        generate_intital_progressfile('data/progress.yaml')
        assert exists('data/progress.yaml')

    def test_load_progress(self):
        ret = load_progress('data/fake_progress.yaml')
        assert ret == None

    def test_restore_progress_vars(self):

        base_dir = 'data/'
        progress_file = join(base_dir, 'progress.yaml')

        with open(progress_file, 'w') as f:
            yaml.safe_dump(self.base_progress_vars, f)

        new_prog = update_progress(deepcopy(progress_file), 'config_file', 'test_path')
        assert new_prog['config_file'] == 'test_path'

        test_restore1 = restore_progress_vars(progress_file, overwrite=True)
        assert test_restore1 != new_prog

        test_restore2 = restore_progress_vars(progress_file, init=True, overwrite=True)
        assert test_restore2 != self.base_progress_vars

        os.remove(progress_file)

    def test_get_pca_progress(self):
        i_dict = {'pca_dirname': '',
                  'scores_path': '',
                  'changeoints_path': ''}

        pca_prog = get_pca_progress(i_dict, i_dict)

        assert all(list(pca_prog.values())) == False

    # def test_get_extraction_progress(self):
    #     base_dir = 'data/'

    #     config_data = read_yaml('data/config.yaml')

    #     extract_wrapper('data/azure_test/nfov_test.mkv',
    #                     None,
    #                     config_data,
    #                     num_frames=60,
    #                     skip=True)
    #     assert exists('data/azure_test/proc/results_00.mp4')

    #     path_dict, num_extracted = get_extraction_progress(base_dir)

    #     assert num_extracted == 1
    #     assert len(list(path_dict.keys())) == 2

    #     shutil.rmtree('data/azure_test/proc/')

    def test_print_progress(self):

        print_progress(self.base_dir, self.base_progress_vars)

    def test_check_progress(self):
        stdin = 'data/stdin.txt'
        base_dir = 'data/'
        progress_file = join(base_dir, 'progress.yaml')

        with open(progress_file, 'w') as f:
            yaml.safe_dump(self.base_progress_vars, f)

        with open(stdin, 'w') as f:
            f.write('N')

        check_progress(progress_file)

        os.remove(progress_file)
