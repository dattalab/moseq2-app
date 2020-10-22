import os
import sys
import ruamel.yaml as yaml
from unittest import TestCase
from os.path import exists, join
from moseq2_app.gui.progress import generate_missing_metadata, get_session_paths, update_progress, \
    restore_progress_vars, handle_progress_restore_input, show_progress_bar, count_total_found_items, get_pca_progress, \
    get_extraction_progress, print_progress, check_progress


class TestNotebookProgress(TestCase):

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

        assert len(paths.keys()) == 1

        paths = get_session_paths(data_dir, extracted=True)

        assert len(paths.keys()) == 0

    def test_update_progress(self):

        base_dir = 'data/'
        progress_file = join(base_dir, 'progress.yaml')

        base_progress_vars = {'base_dir': base_dir,
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
                              'plot_path': os.path.join(base_dir, 'plots/')}

        with open(progress_file, 'w') as f:
            yaml.safe_dump(base_progress_vars, f)

        new_prog = update_progress(progress_file, 'config_file', 'test_path')
        assert new_prog['config_file'] == 'test_path'

        with open(progress_file, 'r') as f:
            read_progs = yaml.safe_load(f)

        assert read_progs != base_progress_vars

        os.remove(progress_file)

    def test_restore_progress_vars(self):

        base_dir = 'data/'
        progress_file = join(base_dir, 'progress.yaml')

        base_progress_vars = {'base_dir': base_dir,
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
                              'plot_path': os.path.join(base_dir, 'plots/')}

        with open(progress_file, 'w') as f:
            yaml.safe_dump(base_progress_vars, f)

        new_prog = update_progress(progress_file, 'config_file', 'test_path')
        assert new_prog['config_file'] == 'test_path'

        test_restore = restore_progress_vars(progress_file)
        assert test_restore == new_prog

        os.remove(progress_file)

    def test_handle_progress_restore_input(self):

        stdin = 'data/stdin.txt'
        base_dir = 'data/'
        progress_file = join(base_dir, 'progress.yaml')

        base_progress_vars = {'base_dir': base_dir,
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
                              'plot_path': os.path.join(base_dir, 'plots/')}

        with open(progress_file, 'w') as f:
            yaml.safe_dump(base_progress_vars, f)


        with open(stdin, 'w') as f:
            f.write('Y')

        sys.stdin = open(stdin)
        progress_vars = handle_progress_restore_input(base_progress_vars, progress_file)

        assert progress_vars == base_progress_vars

        new_prog = update_progress(progress_file, 'config_file', 'test_path')

        with open(stdin, 'w') as f:
            f.write('Y')

        sys.stdin = open(stdin)
        progress_vars = handle_progress_restore_input(base_progress_vars, progress_file)

        assert new_prog == progress_vars

        with open(stdin, 'w') as f:
            f.write('N')

        sys.stdin = open(stdin)
        progress_vars = handle_progress_restore_input(base_progress_vars, progress_file)

        assert progress_vars == base_progress_vars

        with open(stdin, 'w') as f:
            f.write('q')

        sys.stdin = open(stdin)
        progress_vars = handle_progress_restore_input(base_progress_vars, progress_file)

        assert progress_vars == None

        os.remove(progress_file)

    def test_show_progress_bar(self):

        nfound = 5
        total = 10
        desc='test'

        show_progress_bar(nfound, total, desc)

    def test_count_total_found_items(self):

        i_dict = {'1': True,
                  '2': False}

        n = count_total_found_items(i_dict)
        assert n == 1

    def test_get_pca_progress(self):
        i_dict = {'pca_dirname': '',
                  'scores_path': '',
                  'changeoints_path': ''}

        pca_prog = get_pca_progress(i_dict, i_dict)

        assert all(list(pca_prog.values())) == False

    def test_get_extraction_progress(self):
        base_dir = 'data/'

        base_progress_vars = {'base_dir': base_dir,
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
                              'plot_path': os.path.join(base_dir, 'plots/')}

        path_dict, num_extracted = get_extraction_progress(base_progress_vars)

        assert num_extracted == 0
        assert len(list(path_dict.keys())) == 1

    def test_print_progress(self):
        base_dir = 'data/'

        base_progress_vars = {'base_dir': base_dir,
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
                              'plot_path': os.path.join(base_dir, 'plots/')}

        print_progress(base_progress_vars)

    def test_check_progress(self):
        stdin = 'data/stdin.txt'
        base_dir = 'data/'
        progress_file = join(base_dir, 'progress.yaml')

        base_progress_vars = {'base_dir': base_dir,
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
                              'plot_path': os.path.join(base_dir, 'plots/')}

        with open(progress_file, 'w') as f:
            yaml.safe_dump(base_progress_vars, f)

        with open(stdin, 'w') as f:
            f.write('N')

        sys.stdin = open(stdin)
        progress_vars = check_progress(base_dir, progress_file)

        assert progress_vars == base_progress_vars

        os.remove(progress_file)
