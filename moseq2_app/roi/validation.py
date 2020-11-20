'''

The module contains extraction validation functions that test extractions' scalar values,
 timestamps, and position heatmaps.

'''

import h5py
import scipy
import numpy as np
import pandas as pd
from copy import deepcopy
import ruamel.yaml as yaml
import matplotlib.pyplot as plt
from moseq2_extract.util import scalar_attributes
from moseq2_viz.scalars.util import compute_all_pdf_data

def check_timestamp_error_percentage(timestamps, fps=30):
    '''
    https://www.mathworks.com/help/imaq/examples/determining-the-rate-of-acquisition.html

    Returns the proportion of dropped frames relative to the respective recorded timestamps and frames per second.

    Parameters
    ----------
    timestamps (1D np.array): Session's recorded timestamp array.
    fps (int): Frames per second

    Returns
    -------
    percentError (float): Percentage of frames that were dropped/missed during acquisition.
    '''

    # Find the time difference between frames.
    diff = np.diff(timestamps) / 1000

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

    nanrows = scalar_df.isnull().sum(axis=1).to_numpy()

    n_missing_frames = len(nanrows[nanrows > 0])

    return n_missing_frames

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

    missing_mouse_frames = len(scalar_df[np.isclose(scalar_df['area_px'], 0)])

    return missing_mouse_frames

# warning: min height may be too high
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

    corrupt_frames = len(scalar_df[scalar_df['area_px'] < 2*scalar_df['area_px'].std()])

    return corrupt_frames

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

    motionless_frames = len(scalar_df[scalar_df['velocity_2d_mm'] < 0.1])-1 # subtract 1 because first frame is always 0mm/s

    return motionless_frames

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

    scalars = scalar_attributes()
    scalar_dfs = []

    # Get scalar dicts for all the sessions
    for k, v in path_dict.items():
        # Get relevant extraction paths
        h5path = path_dict[k].replace('mp4', 'h5')
        yamlpath = path_dict[k].replace('mp4', 'yaml')

        with open(yamlpath, 'r') as f:
            stat_dict = yaml.safe_load(f)

        metadata = stat_dict['metadata']

        f = h5py.File(h5path, 'r')['scalars']
        tmp = {'uuid': stat_dict['uuid'], 'group': 'default'}
        for key in scalars:
            tmp[key] = f[key][()]

        sess_df = pd.DataFrame.from_dict(tmp)
        for mk, mv in metadata.items():
            if isinstance(mv, list):
                mv = mv[0]
            sess_df[mk] = mv

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
        'scalar_anomaly': {},
        'dropped_frames': False,
        'corrupted': False,
        'stationary': False,
        'missing': False,
        'size_anomaly': False,
        'position_heatmap': False
    }

    # Get flags
    for k, v in paths.items():
        # get yaml metadata
        yamlpath = paths[k].replace('mp4', 'yaml')
        with open(yamlpath, 'r') as f:
            stat_dict = yaml.safe_load(f)
            status_dicts[stat_dict['uuid']] = deepcopy(flags)
            status_dicts[stat_dict['uuid']]['metadata'] = stat_dict['metadata']

        # read timestamps from h5
        h5path = paths[k].replace('mp4', 'h5')
        try:
            timestamps = h5py.File(h5path, 'r')['timestamps'][()]

            # Count dropped frame percentage
            dropped_frames = check_timestamp_error_percentage(timestamps, fps=30)
            if dropped_frames >= 0.05:
                status_dicts[stat_dict['uuid']]['dropped_frames'] = dropped_frames
        except KeyError:
            pass
            print(f'{h5path} timestamps not found.')

    return status_dicts

def get_iqr_anomaly_sessions(scalar_df, status_dicts):
    '''

    Finds sessions that have a mean scalar value (for a subset of scalars), that are outside of
     the accepted inter-quartile range.

    Parameters
    ----------
    scalar_df (pd.DataFrame): Computed Scalar DataFrame
    status_dicts (dict): stacked dictionary object containing all the sessions' flag status dicts.

    Returns
    -------
    status_dicts (dict): stacked dictionary object containing updated scalar_anomaly flags.
    '''

    mean_df = scalar_df.groupby('uuid', as_index=False).mean()

    # Get sessions within interquartile range
    q1 = scalar_df.quantile(.25)
    q2 = scalar_df.quantile(.75)

    # Scalar values to measure
    val_keys = ['area_mm', 'length_mm', 'width_mm', 'height_ave_mm', 'velocity_2d_mm', 'velocity_3d_mm']

    # Get scalar anomalies based on quartile ranges
    for key in val_keys:
        mask = mean_df[key].between(q1[key], q2[key], inclusive=True)
        iqr = mean_df.loc[~mask]

        for s in list(iqr.uuid):
            status_dicts[s]['scalar_anomaly'][key] = True

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

    try:
        status_dicts = run_heatmap_kl_divergence_test(scalar_df, status_dicts)
    except:
        pass

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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

    errors = ['missing', 'dropped_frames', 'corrupted']
    n_errs, n_warnings = 0, 0

    n_sessions = len(anomaly_dict)
    print_dict = {}

    # Count Errors and warnings
    for k in anomaly_dict:
        error, warning = False, False
        for k1, v1 in anomaly_dict[k].items():
            if k1 != 'metadata':
                if k1 in errors and v1 != False:
                    error = True
                elif isinstance(v1, dict):
                    # scalar anomalies
                    if len(v1) > 0:
                        warning = True
                elif isinstance(v1, (float, type(np.array))):
                    warning = True

        if warning:
            n_warnings += 1
            print_dict[k] = anomaly_dict[k]
        if error:
            n_errs += 1
            print_dict[k] = anomaly_dict[k]

    print(f'{bcolors.FAIL}{n_errs}/{n_sessions} were flagged with error.{bcolors.ENDC}')
    print(f'{bcolors.WARNING}{n_warnings}/{n_sessions} were flagged with warning(s).{bcolors.ENDC}')
    print(f'Sessions with {bcolors.FAIL}"Error"{bcolors.ENDC} flags must be re-extracted or excluded.')
    print(f'Sessions with {bcolors.WARNING}"Warning"{bcolors.ENDC} flags can be visually inspected for the plotted/listed scalar inconsistencies.\n')

    # Print results
    for k in print_dict:
        # Get session name
        session_name = print_dict[k]['metadata']['SessionName']
        subject_name = print_dict[k]['metadata']['SubjectName']
        print(f'{bcolors.BOLD}{bcolors.UNDERLINE}Session: {session_name}; Subject: {subject_name} flags:{bcolors.ENDC}')
        for k1, v1 in print_dict[k].items():
            x = ''
            if k1 != 'metadata':
                if k1 in errors and isinstance(v1, float):
                    t = 'Error'
                    x = f'{k1} - {v1*100:.2f}%'
                elif k1 == 'position_heatmap' and not isinstance(v1, bool): 
                    x = 'position heatmaps'
                    t = 'Warning - Position Heatmap was flagged'
                    try:
                        plot_heatmap(v1, f'{session_name}_{subject_name}')
                    except:
                        pass
                elif isinstance(v1, dict):
                    t = 'Warning'
                    if len(v1) > 0:
                        x = list(v1.keys())
                elif v1 == True:
                    t = 'Warning'
                    x = f'{k1} was flagged'
            if len(x) > 0:
                if 'Warning' in t:
                    t = f'{bcolors.WARNING}{t}'
                elif 'Error' in t:
                    t = f'{bcolors.FAIL}{t}'
                print(f'\t{t}: {x}{bcolors.ENDC}')
        print()
