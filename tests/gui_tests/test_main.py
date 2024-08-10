import os
import shutil
from moseq2_extract.io.video import read_frames
import ruamel.yaml as yaml
import multiprocessing as mp
from unittest import TestCase
from os.path import exists, join
from moseq2_viz.util import parse_index, read_yaml
from moseq2_extract.helpers.wrappers import extract_wrapper
from moseq2_app.gui.progress import generate_missing_metadata
from moseq2_app.main import (preview_extractions, validate_extractions,
                             interactive_group_setting, label_syllables, interactive_syllable_stats,
                             interactive_crowd_movie_comparison, interactive_transition_graph,
                             view_extraction)

# Source: https://bugs.python.org/issue33725#msg329923
mp.set_start_method('forkserver')

class TestMain(TestCase):

    def setUp(self):
        self.progress_paths = {
            'base_dir': 'data/',
            'config_file': 'data/config.yaml',
            'session_config': 'data/session_config.yaml',
            'index_file': 'data/test_index.yaml',
            'model_path': 'data/test_model.p',
            'crowd_dir': 'data/crowd_movies/',
            'syll_info': 'data/syll_info.yaml',
            'df_info_path': 'data/syll_df.parquet',
            'plot_path': 'data/plot'
        }

        if exists(self.progress_paths['syll_info']):
            os.remove(self.progress_paths['syll_info'])

        if exists(self.progress_paths['df_info_path']):
            os.remove(self.progress_paths['df_info_path'])

        if exists('data/moseq_scalar_dataframe.parquet'):
            os.remove('data/moseq_scalar_dataframe.parquet')

        if exists(self.progress_paths['crowd_dir']):
            shutil.rmtree(self.progress_paths['crowd_dir'])

    def tearDown(self):
        if exists(self.progress_paths['syll_info']):
            os.remove(self.progress_paths['syll_info'])

        if exists(self.progress_paths['df_info_path']):
            os.remove(self.progress_paths['df_info_path'])

        if exists('data/moseq_scalar_dataframe.parquet'):
            os.remove('data/moseq_scalar_dataframe.parquet')

        if exists(self.progress_paths['crowd_dir']):
            shutil.rmtree(self.progress_paths['crowd_dir'])

    @classmethod
    def tearDownClass(cls):
        group_movie_dir = 'data/grouped_crowd_dir/'
        if exists(group_movie_dir):
            shutil.rmtree(group_movie_dir)

    # def test_flip_classifier_tool(self):

    #     flip_classifier_tool('data/', 'data/classifier.pkl', max_frames=50)

    def test_view_extraction(self):
        out = view_extraction([], default=0)
        assert out == []

    def test_interactive_extraction_preview(self):

        preview_extractions(self.progress_paths['base_dir'])

    def test_validate_extractions(self):
        # generate sample metadata json for each session that is missing one
        generate_missing_metadata('data/azure_test/', 'azure_test')

        config_data = read_yaml('data/config.yaml')

        config_data['camera_type'] = 'auto'
        config_data['bg_roi_depth_range'] = [300, 600]
        config_data['chunk_size'] = 10
        config_data['threads'] = 6
        config_data['bg_roi_erode'] = (1, 1)

        extract_wrapper('data/azure_test/nfov_test.mkv',
                        None,
                        config_data,
                        num_frames=30,
                        skip=True)
        assert exists('data/azure_test/proc/results_00.mp4')

        validate_extractions(self.progress_paths['base_dir'])

        shutil.rmtree('data/azure_test/proc/')

    def test_interactive_group_setting(self):

        index_grid = interactive_group_setting(self.progress_paths['index_file'])

        index_grid.update_table()

        index_grid.update_clicked()

        index_grid.clear_clicked()

    # def test_interactive_syllable_labeler(self):

    #     label_syllables(self.progress_paths, max_syllables=None, n_explained=99)

    #     assert exists(self.progress_paths['syll_info'])
    #     assert exists(self.progress_paths['df_info_path'])
    #     assert exists('data/moseq_scalar_dataframe.parquet')

    #     pg = read_yaml(self.progress_paths['syll_info'])

    #     assert len(pg) == 10

    #     label_syllables(self.progress_paths, max_syllables=None, n_explained=99)

    def test_interactive_syllable_stat(self):

        label_syllables(self.progress_paths, max_syllables=10)

        assert exists(self.progress_paths['syll_info'])
        pg = read_yaml(self.progress_paths['syll_info'])

        assert len(pg) == 10

        interactive_syllable_stats(self.progress_paths, max_syllable=None, load_parquet=False)

        interactive_syllable_stats(self.progress_paths, max_syllable=5, load_parquet=True)

        _, sorted_index = parse_index(self.progress_paths['index_file'])

        sorted_index['pca_path'] = 'data/test_scores.h5'

        interactive_syllable_stats(self.progress_paths, max_syllable=15, load_parquet=True)

    def test_interactive_crowd_movie_comparison_preview(self):

        group_movie_dir = 'data/grouped_crowd_dir/'
        if exists(group_movie_dir):
            shutil.rmtree(group_movie_dir)

        label_syllables(self.progress_paths, max_syllables=None, n_explained=99)
        interactive_crowd_movie_comparison(self.progress_paths, group_movie_dir, get_pdfs=True, load_parquet=False)

        assert len([f for f in os.listdir(join(group_movie_dir, 'default/')) if f.endswith('.mp4')]) == 1

    def test_interactive_plot_transition_graph(self):
        label_syllables(self.progress_paths, max_syllables=None, n_explained=99)
        interactive_transition_graph(self.progress_paths, max_syllables=None, plot_vertically=False, load_parquet=True)
