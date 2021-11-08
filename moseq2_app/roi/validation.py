'''

The module contains extraction validation functions that test extractions' scalar values,
 timestamps, and position heatmaps.

'''
import scipy
import numpy as np
import pandas as pd
from copy import deepcopy
from os.path import exists
import matplotlib.pyplot as plt
from moseq2_app.util import bcolors
from sklearn.covariance import EllipticEnvelope
from moseq2_viz.util import h5_to_dict, read_yaml
from moseq2_viz.scalars.util import compute_all_pdf_data


def check_timestamp_error_percentage(timestamps, fps=30, scaling_factor=1000):
    '''
    https://www.mathworks.com/help/imaq/examples/determining-the-rate-of-acquisition.html

    Returns the proportion of dropped frames relative to the respective recorded timestamps and frames per second.

    Parameters
    ----------
    timestamps (1D np.array): Session's recorded timestamp array.
    fps (int): Frames per second
    scaling_factor (float): factor to divide timestamps by to convert timestamp milliseconds into seconds.

    Returns
    -------
    percentError (float): Percentage of frames that were dropped/missed during acquisition.
    '''

    # Find the time difference between frames.
    diff = np.diff(timestamps)

    # Check if the timestamps are in milliseconds based on the mean difference amount
    if np.mean(diff) > 10:
        # rescale the timestamps to seconds
        diff /= scaling_factor

    # Find the average time difference between frames.
    avgTime = np.mean(diff)

    # Determine the experimental frame rate.
    expRate = 1 / avgTime

    # Determine the percent error between the determined and actual frame rate.
    diffRates = abs(fps - expRate)
    percentError = (diffRates / fps)

    return percentError

def count_nan_rows(scalar_df):
    '''

    Counts the number of rows with NaN scalar values.

    Parameters
    ----------
    scalar_df (pd.DataFrame): Computed Scalar DataFrame

    Returns
    -------
    n_missing_frames (int): Number of frames with NaN computed scalar values.
    '''

    return scalar_df.isnull().any(1).sum()


def count_missing_mouse_frames(scalar_df):
    '''

    Counts the number of frames where the mouse is not found.

    Parameters
    ----------
    scalar_df (pd.DataFrame): Computed Scalar DataFrame

    Returns
    -------
    missing_mouse_frames (int): Number of frames with recorded mouse area ~= 0
    '''

    return (scalar_df["area_px"] == 0).sum()


def count_frames_with_small_areas(scalar_df):
    '''

    Counts the number of frames where the mouse area is smaller than 2 standard deviations of
     all mouse areas.

    Parameters
    ----------
    scalar_df (pd.DataFrame): Computed Scalar DataFrame

    Returns
    -------
    corrupt_frames (int): Number of frames where the recorded mouse area is too small
    '''

    return (scalar_df["area_px"] < 2 * scalar_df["area_px"].std()).sum()


def count_stationary_frames(scalar_df):
    '''

    Counts the number of frames where mouse is not moving.

    Parameters
    ----------
    scalar_df (pd.DataFrame): Computed Scalar DataFrame

    Returns
    -------
    motionless_frames (int): Number of frames where the mouse is not moving
    '''
    
    # subtract 1 because first frame is always 0mm/s
    return (scalar_df["velocity_2d_mm"] < 0.1).sum() - 1


def get_scalar_df(path_dict):
    '''
    Computes a scalar dataframe that contains all the extracted sessions
    recorded scalar values along with their metadata.

    Parameters
    ----------
    path_dict (dict): dictionary of session folder names paired with their extraction paths

    Returns
    -------
    scalar_df (pd.DataFrame): DataFrame containing loaded scalar info from each h5 extraction file.
    '''

    scalar_dfs = []

    # Get scalar dicts for all the sessions
    for k, v in path_dict.items():
        if not v.endswith('.mp4'):
            continue
        # Get relevant extraction paths
        h5path = v.replace('mp4', 'h5')
        yamlpath = v.replace('mp4', 'yaml')

        if not exists(yamlpath):
            print(f'No valid yaml path for session: {k}')
            continue
        stat_dict = read_yaml(yamlpath)

        metadata = stat_dict['metadata']

        tmp = h5_to_dict(h5path, path='scalars')
        tmp = {**tmp, 'uuid': stat_dict['uuid'], 'group': 'default'}
        for mk in ['SessionName', 'SubjectName']:
            mv = metadata[mk]
            tmp[mk] = mv[0] if isinstance(mv, list) else mv

        sess_df = pd.DataFrame(tmp)
        scalar_dfs.append(sess_df)

    scalar_df = pd.concat(scalar_dfs)
    return scalar_df


def compute_kl_divergences(pdfs, groups, sessions, sessionNames, oob=False):
    '''
    Computes KL divergence for all sessions and returns the divergences
    Consider trying Jensen Shannon or Wasserstein instead!!

    Parameters
    ----------
    pdfs (list): list of 2d probability density functions (heatmaps) describing mouse position.
    groups (list): list of groups corresponding to the pdfs indices
    sessions (list): list of sessions corresponding to the pdfs indices
    sessionNames (list): list of sessionNames corresponding to the pdfs indices
    oob (boolean): Compute out-of-bag KL-divergences

    Returns
    -------
    kl_divergences (pd.Dataframe): dataframe with mouse group, session, subjectname, and kl divergence
    '''

    if oob:
        divergence_vals = []
        for i, pdf in enumerate(pdfs):
            oob_mean_pdf = pdfs[np.arange(len(pdfs)) != i].mean(0).flatten()
            divergence_vals.append(scipy.stats.entropy(pk=oob_mean_pdf, qk=pdf.flatten()))
    else:
        overall_mean_pdf = pdfs.mean(0).flatten()
        divergence_vals = [scipy.stats.entropy(pk=overall_mean_pdf, qk=pdf.flatten()) for pdf in pdfs]

    kl_divergences = pd.DataFrame({"group": groups,
                                   "session": sessions,
                                   "sessionName": sessionNames,
                                   "divergence": divergence_vals})

    return kl_divergences

def get_kl_divergence_outliers(kl_divergences):
    '''
    Returns the position PDFs that are over 2 standard deviations away from the mean position divergence.

    Parameters
    ----------
    kl_divergences (pd.Dataframe): dataframe with group, session, subjectName, and divergence

    Returns
    -------
    outliers (pd.Dataframe): dataframe of outlier sessions
    '''

    kl_sorted = kl_divergences.sort_values(by='divergence', ascending=False).reset_index(drop=True)

    kl_mean = kl_sorted['divergence'].mean()
    kl_std = kl_sorted['divergence'].std()

    outliers = kl_sorted[kl_sorted['divergence'] >= kl_mean + 2 * kl_std]
    return outliers

def make_session_status_dicts(paths):
    '''
    Returns the flag status dicts for all the found completed extracted sessions. Additionally performs
     dropped frames test on all sessions.

    Parameters
    ----------
    paths (dict): path dict of session names paired wit their mp4 paths.

    Returns
    -------
    status_dicts (dict): stacked dictionary object containing all the sessions' flag status dicts.
    '''

    status_dicts = {}

    # Get default flags
    flags = {
        'metadata': {},
        'scalar_anomaly': False,
        'dropped_frames': False,
        'corrupted': False,
        'stationary': False,
        'missing': False,
        'size_anomaly': False,
    }

    # Get flags
    for k, v in paths.items():
        if not v.endswith('.mp4'):
            continue

        # get yaml metadata
        yamlpath = v.replace('.mp4', '.yaml')
        h5path = v.replace('.mp4', '.h5')

        if not exists(yamlpath):
            print(f'No valid yaml path for session: {k}')
            continue

        stat_dict = read_yaml(yamlpath)
        status_dicts[stat_dict['uuid']] = deepcopy(flags)
        status_dicts[stat_dict['uuid']]['metadata'] = stat_dict['metadata']

        # read timestamps from h5
        try:
            timestamps = h5_to_dict(h5path, path='timestamps')['timestamps']
            # Count dropped frame percentage
            dropped_frames = check_timestamp_error_percentage(timestamps, fps=30)
            if dropped_frames >= 0.05:
                status_dicts[stat_dict['uuid']]['dropped_frames'] = dropped_frames
        except KeyError:
            print(f'{h5path} timestamps not found.')

    return status_dicts


def get_scalar_anomaly_sessions(scalar_df, status_dicts):
    '''
    Detects outlier sessions using an EllipticEnvelope model based on a subset of their mean scalar values.

    Parameters
    ----------
    scalar_df (pd.DataFrame): Computed Scalar DataFrame
    status_dicts (dict): stacked dictionary object containing all the sessions' flag status dicts.

    Returns
    -------
    status_dicts (dict): stacked dictionary object containing updated scalar_anomaly flags.
    '''

    # Scalar values to measure
    val_keys = ['area_mm', 'length_mm', 'width_mm', 'height_ave_mm', 'velocity_2d_mm', 'velocity_3d_mm']

    mean_df = scalar_df.groupby('uuid').mean()

    try:
        outliers = EllipticEnvelope(random_state=0).fit_predict(mean_df[val_keys].to_numpy())
    except Exception as e:
        # create a list of inlier list that matches the mean_df.index length
        outliers = [1] * len(mean_df.index)

    # Get scalar anomalies based on quartile ranges
    for i, index in enumerate(mean_df.index):
        if outliers[i] == -1:
            status_dicts[index]['scalar_anomaly'] = True

    return status_dicts


def run_heatmap_kl_divergence_test(scalar_df, status_dicts):
    '''

    Finds the position PDF outlier sessions and updates the status_dicts with the respective position heatmap flag.

    Parameters
    ----------
    scalar_df (pd.DataFrame): Computed Scalar DataFrame
    status_dicts (dict): stacked dictionary object containing all the sessions' flag status dicts.

    Returns
    -------
    status_dicts (dict): stacked dictionary object containing updated position_heatmap flags.
    '''

    pdfs, groups, sessions, sessionNames = compute_all_pdf_data(scalar_df, key='uuid')

    kl_divergences = compute_kl_divergences(pdfs, groups, sessions, sessionNames)

    outliers = list(set(get_kl_divergence_outliers(kl_divergences).sessionName))

    outlier_indices = [i for i, e in enumerate(sessionNames) if e in outliers]
    outlier_pdfs = pdfs[outlier_indices]

    for o, pdf in zip(outliers, outlier_pdfs):
        status_dicts[o]['position_heatmap'] = pdf

    return status_dicts


def run_validation_tests(scalar_df, status_dicts):
    '''

    Main function that runs all the available extraction validation tests and updates the status_dicts
     flags accordingly.

    Parameters
    ----------
    scalar_df (pd.DataFrame): Computed Scalar DataFrame
    status_dicts (dict): stacked dictionary object containing all the sessions' flag status dicts.

    Returns
    -------
    status_dicts (dict): stacked dictionary object containing all the sessions' updated flag status dicts.
    '''

    sessionNames = list(scalar_df.uuid.unique())

    for s in sessionNames:
        df = scalar_df[scalar_df['uuid'] == s]

        # Count stationary frames
        stat_percent = count_stationary_frames(df) / len(df)
        if stat_percent >= 0.05:
            status_dicts[s]['stationary'] = stat_percent

        # Get frames with missing mouse
        missing_percent = count_missing_mouse_frames(df) / len(df)
        if missing_percent >= 0.05:
            status_dicts[s]['missing'] = missing_percent

        # Get frames with mouse sizes that are too small
        size_anomaly = count_frames_with_small_areas(df) / len(df)
        if size_anomaly >= 0.05:
            status_dicts[s]['size_anomaly'] = size_anomaly
            
        # Get corrupted frame ratio
        corrupted_percent = count_nan_rows(df) / len(df)
        if corrupted_percent >= 0.05:
            status_dicts[s]['corrupted'] = corrupted_percent

    return status_dicts

def plot_heatmap(heatmap, title):
    '''
    Plots and displays outlier heatmap with the SessionName as the title.

    Parameters
    ----------
    heatmap (2d np.array): outlier position PDF
    title (str): plot title

    Returns
    -------
    '''

    im = plt.imshow(np.array(heatmap) / np.array(heatmap).max())
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f'{title}')
    plt.show(block=False)

def print_validation_results(scalar_df, status_dicts):
    '''

    Displays all the outlier sessions flag names and values. Additionally plots the flagged
     position heatmap.

    Parameters
    ----------
    anomaly_dict (dict): Dict object containing specific session flags to print

    Returns
    -------
    '''

    # Run tests
    anomaly_dict = run_validation_tests(scalar_df, status_dicts)

    n_warnings = 0

    n_sessions = len(anomaly_dict)
    print_dict = {}

    # Count Errors and warnings
    for k in anomaly_dict:
        warning = False
        for k1, v1 in anomaly_dict[k].items():
            # v1 is polymorphic; bool for flags when False, dict for metadata
            # float that represent the percentage for dropped frame, corrupted
            # stationary, missing and size anomaly when the flags are True
            if k1 != 'metadata':
                # v1 is bool when k1 is not metadata 
                if isinstance(v1, float):
                    warning = True
                elif v1:
                    warning = True
        
        # count the number of warnings
        # add the sessions with warning to print_dict
        if warning:
            n_warnings += 1
            print_dict[k] = anomaly_dict[k]

    print(f'{bcolors.WARNING}{n_warnings}/{n_sessions} were flagged with warning(s).{bcolors.ENDC}')
    print(f'Sessions with {bcolors.WARNING}"Warning"{bcolors.ENDC} flags may need visually inspection for outliers.\n')

    # Print results
    for k in print_dict:
        # Get session name
        session_name = print_dict[k]['metadata']['SessionName']
        subject_name = print_dict[k]['metadata']['SubjectName']
        print(f'{bcolors.BOLD}{bcolors.UNDERLINE}Session: {session_name}; Subject: {subject_name} flags:{bcolors.ENDC}')
        for k1, v1 in print_dict[k].items():
            if k1 != 'metadata':
                # scalar anomaly flag
                if v1:
                    x = f'{k1} flag raised'
                    print(f'\t{bcolors.WARNING}Warning: {x}{bcolors.ENDC}')
                # all the other flags
                elif isinstance(v1, float):
                    x = f'{k1} flag raised: {v1*100:.2f}%'
                    print(f'\t{bcolors.WARNING}Warning: {x}{bcolors.ENDC}') 
        # I assume this is printing a new line
        print()
