import h5py
import numpy as np
from copy import deepcopy
from unittest import TestCase
from moseq2_app.util import index_to_dataframe
from moseq2_extract.io.video import load_timestamps_from_movie
from moseq2_viz.scalars.util import compute_all_pdf_data
from moseq2_app.roi.validation import get_scalar_df, check_timestamp_error_percentage, count_nan_rows, \
    count_missing_mouse_frames, count_frames_with_small_areas, count_stationary_frames, compute_kl_divergences, \
    get_kl_divergence_outliers, make_session_status_dicts, get_scalar_anomaly_sessions, run_heatmap_kl_divergence_test, \
    run_validation_tests, plot_heatmap, print_validation_results

class TestExtractionValidation(TestCase):

    def test_check_timestamp_error_percentage(self):

        paths = {
            'session_1': 'data/test_session/proc/results_00.mp4',
            'azure_test': 'data/azure_test/nfov_test.mkv'
        }

        h5path = paths['session_1'].replace('mp4', 'h5')
        timestamps = h5py.File(h5path, 'r')['timestamps'][()]

        percent_error = check_timestamp_error_percentage(timestamps)

        assert percent_error == 0.011003544858038812

        azure_ts = load_timestamps_from_movie(paths['azure_test'])
        percent_error = check_timestamp_error_percentage(azure_ts)

        assert percent_error == 0.2615314701204273

    def test_count_nan_rows(self):

        paths = {
            'session_1': 'data/test_session/proc/results_00.mp4'
        }

        # Get scalar dataframe including all sessions
        scalar_df = get_scalar_df(paths)

        assert scalar_df.shape == (900, 21)

        n_missing_frames = count_nan_rows(scalar_df)
        assert n_missing_frames == 0

    def test_count_missing_mouse_frames(self):

        paths = {
            'session_1': 'data/test_session/proc/results_00.mp4'
        }

        # Get scalar dataframe including all sessions
        scalar_df = get_scalar_df(paths)

        missing_mouse_frames = count_missing_mouse_frames(scalar_df)

        assert missing_mouse_frames == 0

    def test_count_frames_with_small_areas(self):

        paths = {
            'session_1': 'data/test_session/proc/results_00.mp4'
        }

        # Get scalar dataframe including all sessions
        scalar_df = get_scalar_df(paths)

        n_small_area_frames = count_frames_with_small_areas(scalar_df)

        assert n_small_area_frames == 8

    def test_count_stationary_frames(self):

        paths = {
            'session_1': 'data/test_session/proc/results_00.mp4'
        }

        # Get scalar dataframe including all sessions
        scalar_df = get_scalar_df(paths)

        stat_frames = count_stationary_frames(scalar_df)

        assert stat_frames == 13

    def test_get_scalar_df(self):
        paths = {
            'session_1': 'data/test_session/proc/results_00.mp4'
        }

        # Get scalar dataframe including all sessions
        scalar_df = get_scalar_df(paths)

        assert scalar_df.shape == (900, 21)

    def test_compute_kl_divergences(self):

        paths = {
            'session_1': 'data/test_session/proc/results_00.mp4'
        }

        # Get scalar dataframe including all sessions
        scalar_df = get_scalar_df(paths)

        pdfs, groups, sessions, sessionNames = compute_all_pdf_data(scalar_df, key='SessionName')

        kl_divs = compute_kl_divergences(pdfs, groups, sessions, sessionNames, oob=False)

        assert kl_divs.divergence[0] == 0.0
        assert len(kl_divs) == 1

    def test_get_kl_divergence_outliers(self):
        paths = {
            'session_1': 'data/test_session/proc/results_00.mp4'
        }

        # Get scalar dataframe including all sessions
        scalar_df = get_scalar_df(paths)

        pdfs, groups, sessions, sessionNames = compute_all_pdf_data(scalar_df, key='SessionName')

        kl_divs = compute_kl_divergences(pdfs, groups, sessions, sessionNames, oob=True)

        outliers = get_kl_divergence_outliers(kl_divs)

        assert len(outliers) == 0

    def test_make_session_status_dicts(self):

        paths = {
            'session_1': 'data/test_session/proc/results_00.mp4'
        }

        status_dicts = make_session_status_dicts(paths)
        assert len(list(status_dicts.keys())) == 1
        assert list(status_dicts['5c72bf30-9596-4d4d-ae38-db9a7a28e912']) == ['metadata', 'scalar_anomaly',
                                                'dropped_frames', 'corrupted', 'stationary', 'missing',
                                                'size_anomaly', 'position_heatmap']

    def test_run_heatmap_kl_divergence_test(self):
        paths = {
            'session_1': 'data/test_session/proc/results_00.mp4'
        }

        status_dicts = make_session_status_dicts(paths)

        scalar_df = get_scalar_df(paths)

        new_status_dicts = run_heatmap_kl_divergence_test(scalar_df, deepcopy(status_dicts))

        assert new_status_dicts == status_dicts

    def test_get_scalar_anomaly_sessions(self):
        paths = {
            'session_1': 'data/test_session/proc/results_00.mp4'
        }

        status_dicts = make_session_status_dicts(paths)

        scalar_df = get_scalar_df(paths)

        test_x = get_scalar_anomaly_sessions(scalar_df, status_dicts)
        print(test_x)
        assert test_x['5c72bf30-9596-4d4d-ae38-db9a7a28e912']['scalar_anomaly'] == False


    def test_run_validation_tests(self):
        paths = {
            'session_1': 'data/test_session/proc/results_00.mp4'
        }

        status_dicts = make_session_status_dicts(paths)

        scalar_df = get_scalar_df(paths)

        new_status_dicts = run_validation_tests(scalar_df, deepcopy(status_dicts))

        assert new_status_dicts == status_dicts
        assert new_status_dicts['5c72bf30-9596-4d4d-ae38-db9a7a28e912']['dropped_frames'] == False

    def test_plot_heatmap(self):

        heatmap = np.zeros((50, 50))

        plot_heatmap(heatmap, 'tests')

    def test_print_validation_results(self):
        paths = {
            'session_1': 'data/test_session/proc/results_00.mp4',
            'azure_test': 'data/azure_test/nfov_test.mkv'
        }

        status_dicts = make_session_status_dicts(paths)

        scalar_df = get_scalar_df(paths)

        print_validation_results(scalar_df, status_dicts)

    ## util.py test
    def test_index_to_dataframe(self):

        index_path = 'data/test_index.yaml'

        index_data, df = index_to_dataframe(index_path)

        assert df.shape == (2, 14)
        assert isinstance(index_data, dict)
