"""
Utility functions for handling all scalar data-related operations.
"""

import h5py
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from itertools import starmap
from os.path import exists
from collections import defaultdict
from cytoolz import valmap, get, merge, keyfilter
from moseq2_viz.model.util import get_transitions, prepare_model_dataframe
from moseq2_viz.util import (h5_to_dict, strided_app, h5_filepath_from_sorted,
                             get_timestamps_from_h5, parse_index)


def _star_itemmap(func, d):
    return dict(starmap(func, d.items()))


def star_valmap(func, d):
    """
    map the function to itratables
    
    Args:
    func (func): function to map onto the values in iterables
    d (dict): iterables

    Returns:
    (dict): dictionary of key and function mapped to the values
    """
    keys = list(d.keys())
    return dict(zip(keys, starmap(func, d.values())))


def convert_pxs_to_mm(coords, resolution=(512, 424), field_of_view=(70.6, 60), true_depth=673.1):
    """
    Convert x, y coordinates in pixel space to mm
    
    Args:
    coords (list): list of [x,y] pixel coordinate lists.
    resolution (tuple): video frame size.
    field_of_view (tuple): camera focal lengths.
    true_depth (float): detected distance between depth camera and bucket floor.

    Returns:
    new_coords (list): list of same [x,y] coordinates in millimeters.
    """

    # http://stackoverflow.com/questions/17832238/kinect-intrinsic-parameters-from-field-of-view/18199938#18199938
    # http://www.imaginativeuniversal.com/blog/post/2014/03/05/quick-reference-kinect-1-vs-kinect-2.aspx
    # http://smeenk.com/kinect-field-of-view-comparison/
    cx = resolution[0] // 2
    cy = resolution[1] // 2

    xhat = coords[:, 0] - cx
    yhat = coords[:, 1] - cy

    fw = resolution[0] / (2 * np.deg2rad(field_of_view[0] / 2))
    fh = resolution[1] / (2 * np.deg2rad(field_of_view[1] / 2))

    new_coords = np.zeros_like(coords)
    new_coords[:, 0] = true_depth * xhat / fw
    new_coords[:, 1] = true_depth * yhat / fh

    return new_coords


def is_legacy(features: dict):
    """
    Check a dictionary of features to see if they correspond with an older version of moseq.

    Args:
    features (dict): dict of scalar_df column names.

    Returns:
    (bool): true if the dict is from an old dataset
    """

    old_features = ('centroid_x', 'centroid_y', 'width', 'length', 'area', 'height_ave')
    return any(x in old_features for x in features)


def generate_empty_feature_dict(nframes) -> dict:
    """
    Generates a dict of numpy.array of zeros of length nframes for each feature parameter.

    Args:
    nframes (int): length of video

    Returns:
    (dict): dictionary feature to numpy 0 arrays of length nframes key-value pairs.
    """

    features = (
        'centroid_x_px', 'centroid_y_px', 'velocity_2d_px', 'velocity_3d_px',
        'width_px', 'length_px', 'area_px', 'centroid_x_mm', 'centroid_y_mm',
        'velocity_2d_mm', 'velocity_3d_mm', 'width_mm', 'length_mm', 'area_mm',
        'height_ave_mm', 'angle', 'velocity_theta'
    )

    def make_empy_arr():
        return np.zeros((abs(nframes),), dtype='float32')
    return {k: make_empy_arr() for k in features}


def convert_legacy_scalars(old_features, force: bool = False, true_depth: float = 673.1) -> dict:
    """
    Convert scalars in the legacy format to the new format, with explicit units.

    Args:
    old_features (str, h5 group, or dictionary of scalars): filename, h5 group, or dictionary of scalar values.
    force (bool): boolean flag that forces the conversion of centroid_[xy]_px into mm.
    true_depth (float): true depth of the floor relative to the camera (673.1 mm by default)

    Returns:
    features (dict): dictionary of scalar values
    """

    if isinstance(old_features, h5py.Group) and 'centroid_x' in old_features:
        print('Loading scalars from h5 dataset')
        old_features = h5_to_dict(old_features, '/')

    elif isinstance(old_features, (str, np.str_)) and exists(old_features):
        print('Loading scalars from file')
        old_features = h5_to_dict(old_features, 'scalars')

    if 'centroid_x_mm' in old_features and force:
        centroid = np.hstack((old_features['centroid_x_px'][:, None],
                              old_features['centroid_y_px'][:, None]))
        nframes = len(old_features['centroid_x_mm'])
    elif not force:
        print('Features already converted')
        return old_features
    else:
        centroid = np.hstack((old_features['centroid_x'][:, None],
                              old_features['centroid_y'][:, None]))
        nframes = len(old_features['centroid_x'])

    features = generate_empty_feature_dict(nframes)

    centroid_mm = convert_pxs_to_mm(centroid, true_depth=true_depth)
    centroid_mm_shift = convert_pxs_to_mm(centroid + 1, true_depth=true_depth)

    px_to_mm = np.abs(centroid_mm_shift - centroid_mm)

    features['centroid_x_px'] = centroid[:, 0]
    features['centroid_y_px'] = centroid[:, 1]

    features['centroid_x_mm'] = centroid_mm[:, 0]
    features['centroid_y_mm'] = centroid_mm[:, 1]

    # based on the centroid of the mouse, get the mm_to_px conversion
    copy_keys = ('width', 'length', 'area')
    for key in copy_keys:
        # first try to grab _px key, then default to old version name
        features[f'{key}_px'] = old_features[f'{key}_px'] if f'{key}_px' in old_features else old_features[key]

    if 'height_ave_mm' in old_features.keys():
        features['height_ave_mm'] = old_features['height_ave_mm']
    else:
        features['height_ave_mm'] = old_features['height_ave']

    features['width_mm'] = features['width_px'] * px_to_mm[:, 1]
    features['length_mm'] = features['length_px'] * px_to_mm[:, 0]
    features['area_mm'] = features['area_px'] * px_to_mm.mean(axis=1)

    features['angle'] = old_features['angle']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        vel_x = np.diff(np.concatenate((features['centroid_x_px'][:1], features['centroid_x_px'])))
        vel_y = np.diff(np.concatenate((features['centroid_y_px'][:1], features['centroid_y_px'])))
        vel_z = np.diff(np.concatenate((features['height_ave_mm'][:1], features['height_ave_mm'])))

        features['velocity_2d_px'] = np.hypot(vel_x, vel_y)
        features['velocity_3d_px'] = np.sqrt(
            np.square(vel_x)+np.square(vel_y)+np.square(vel_z))

        vel_x = np.diff(np.concatenate((features['centroid_x_mm'][:1], features['centroid_x_mm'])))
        vel_y = np.diff(np.concatenate((features['centroid_y_mm'][:1], features['centroid_y_mm'])))

        features['velocity_2d_mm'] = np.hypot(vel_x, vel_y)
        features['velocity_3d_mm'] = np.sqrt(
            np.square(vel_x)+np.square(vel_y)+np.square(vel_z))

        features['velocity_theta'] = np.arctan2(vel_y, vel_x)

    return features


def get_scalar_map(index, fill_nans=True, force_conversion=False):
    """
    Return a dictionary of scalar values loaded from an index dictionary.

    Args:
    index (dict): dictionary of index file contents.
    fill_nans (bool): boolean flag that indicates whether to replace NaN values with 0.
    force_conversion (bool): boolean flag that forces the conversion of centroid_[xy]_px into mm.

    Returns:
    scalar_map (dict): dictionary of all the scalar values acquired after extraction.
    """

    scalar_map = {}
    score_idx = h5_to_dict(index['pca_path'], 'scores_idx')

    try:
        iter_items = index['files'].items()
    except AttributeError:
        # index['files'] was not loaded as a dictionary
        iter_items = enumerate(index['files'])

    for i, v in iter_items:
        if isinstance(index['files'], list):
            uuid = index['files'][i]['uuid']
        elif isinstance(index['files'], dict):
            uuid = i

        scalars = h5_to_dict(v['path'][0], 'scalars')
        conv_scalars = convert_legacy_scalars(scalars, force=force_conversion)

        if conv_scalars is not None:
            scalars = conv_scalars

        idx = score_idx[uuid]
        scalar_map[uuid] = {}

        for k, v_scl in scalars.items():
            if fill_nans:
                scalar_map[uuid][k] = np.full((len(idx), ), np.nan, dtype='float32')
                scalar_map[uuid][k][~np.isnan(idx)] = v_scl
            else:
                scalar_map[uuid][k] = v_scl

    return scalar_map


def get_scalar_triggered_average(scalar_map, model_labels, max_syllable=40, nlags=20,
                                 include_keys=['velocity_2d_mm', 'velocity_3d_mm', 'width_mm',
                                               'length_mm', 'height_ave_mm', 'angle'],
                                 zscore=False):
    """
    Get the averages of selected scalar keys for each syllable.

    Args:
    scalar_map (dict): dictionary of all the scalar values acquired after extraction.
    model_labels (dict): dictionary of uuid to syllable label array pairs.
    max_syllable (int): the index of the maximum syllable to use.
    nlags (int): the number of lags to use when averaging over a series of PCs.
    include_keys (list): list of scalar values to load averages of.
    zscore (bool): indicate whether to z-score loaded values.

    Returns:
    syll_average (dict): dictionary of scalars for each syllable sequence.
    """

    win = int(nlags * 2 + 1)

    # cumulative average of PCs for nlags

    if np.mod(win, 2) == 0:
        win = win + 1

    # cumulative average of PCs for nlags
    # grab the windows where 0=syllable onset

    syll_average = {}
    count = np.zeros((max_syllable, ), dtype='int')

    for scalar in include_keys:
        syll_average[scalar] = np.zeros((max_syllable, win), dtype='float32')

    for k, v in scalar_map.items():

        labels = model_labels[k]
        seq_array, locs = get_transitions(labels)

        for i in range(max_syllable):
            hits = locs[np.where(seq_array == i)[0]]

            if len(hits) < 1:
                continue

            count[i] += len(hits)

            for scalar in include_keys:
                use_scalar = v[scalar]
                if scalar == 'angle':
                    use_scalar = np.diff(use_scalar)
                    use_scalar = np.insert(use_scalar, 0, 0)
                if zscore:
                    use_scalar = nanzscore(use_scalar)

                padded_scores = np.pad(use_scalar, (win // 2, win // 2),
                                       'constant', constant_values=np.nan)
                win_scores = strided_app(padded_scores, win, 1)
                syll_average[scalar][i] += np.nansum(win_scores[hits, :], axis=0)

    for i in range(max_syllable):
        for scalar in include_keys:
            syll_average[scalar][i] /= count[i]

    return syll_average


def nanzscore(data):
    """
    Apply Z-score numpy.array that may contain NaN values.

    Args:
    data (np.ndarray): array of scalar values.

    Returns:
    data (np.ndarray): z-scored data.
    """

    return (data - np.nanmean(data)) / np.nanstd(data)


def _pca_matches_labels(pca, labels):
    """
    Ensure that the number of frames in the pca dataset matches the number of frames in the assigned labels.

    Args:
    pca (np.array): array of session PC scores.
    labels (np.array): array of session syllable labels

    Returns:
    (bool): indicates whether the PC scores length matches the corresponding assigned labels.
    """

    return len(pca) == len(labels)


def process_scalars(scalar_map: dict, include_keys: list, zscore: bool = False) -> dict:
    """
    Fill NaNs and possibly zscore scalar values.

    Args:
    scalar_map (dict): dictionary of all the scalar values acquired after extraction.
    include_keys (list): scalar keys to process.
    zscore (bool): boolean flag that indicates whether to z-score loaded values.

    Returns:
    scalar_map (dict): dict that contains the updated NaN-filled values.
    """

    out = defaultdict(list)
    for k, v in scalar_map.items():
        for scalar in include_keys:
            use_scalar = v[scalar]
            if scalar == 'angle':
                use_scalar = np.diff(use_scalar)
                use_scalar = np.insert(use_scalar, 0, 0)
            if zscore:
                use_scalar = nanzscore(use_scalar)
            out[k].append(use_scalar)
    return valmap(np.array, out)


def compute_mouse_dist_to_center(roi, centroid_x_px, centroid_y_px):
    """
    Compute the distance between the mouse and arena centroid.

    Args:
    roi (tuple): Tuple of session's arena dimensions.
    centroid_x_px (np.array): x-coordinate of the mouse centroid throughout the recording
    centroid_y_px (np.array): y-coordinate of the mouse centroid throughout the recording

    Returns:
    dist_to_center (np.array): array of distance to the arena center in pixels.
    """

    # Get (x,y) bucket center coordinate.
    # The bounds and centers are kept as Python scalars so that the arithmetic
    # below stays in the centroids' own dtype -- a numpy scalar would promote
    # the float32 centroid arrays to float64 and change the dtype of the
    # dist_to_center_px column.
    ymin, xmin = (int(v) for v in np.min(np.where(roi), axis=1))
    ymax, xmax = (int(v) for v in np.max(np.where(roi), axis=1))
    center_x = float(np.mean([xmin, xmax]))
    center_y = float(np.mean([ymin, ymax]))

    # Get (x,y) distances to bucket center throughout the session recording.
    # add back bounding box offset
    dx = (centroid_x_px + xmin) - center_x
    dy = (centroid_y_px + ymin) - center_y

    # Compute distance to center
    return np.hypot(dx, dy)


def scalars_to_dataframe(index: dict, include_keys: list = ['SessionName', 'SubjectName', 'StartTime'],
                         disable_output=False, force_conversion=True, model_path=None):
    """
    Generate a dataframe containing scalar values for all sessions and the dataframe will include syllable information when model_path in not None.

    Args:
    index (dict): a sorted_index dictionary generated by `parse_index` or `get_sorted_index`
    include_keys (list): a list of other moseq related keys to include in the dataframe
    disable_output (bool): boolean flag that indicates whether to show tqdm output.
    force_conversion (bool): boolean flag that forces the conversion of centroid_[xy]_px into mm.
    model_path (str): path to model object to pull labels from and include in the dataframe

    Returns:
    scalar_df (pandas DataFrame): DataFrame of loaded scalar values with their selected metadata.
    """
    warnings.filterwarnings('ignore', '', FutureWarning)

    has_model = False # indicator for whether users inputted a model_path to load syllable labels from
    model_uuids = None
    model_type = ''
    if model_path is not None and exists(model_path):
        labels_df = prepare_model_dataframe(model_path, index['pca_path']).set_index('uuid')
        model_type = labels_df['model_type'].unique()[0]
        has_model = True
        # loading the session uuids that the model was trained on
        model_uuids = labels_df.reset_index().uuid.unique()

    # check if files is dictionary from sorted_index or list from unsorted index, then sort
    if isinstance(index['files'], list):
        _, index = parse_index(index)

    dfs = []
    # Iterate through index file session info and paths
    for k, v in tqdm(index['files'].items(), disable=disable_output, desc='Creating MoSeq DataFrame'):
        if has_model:
            # skipping the session uuids (found in the index file) that are not included in the model uuids.
            if k not in model_uuids:
                continue
        # Get path to extraction h5 file
        pth = h5_filepath_from_sorted(v)
        # Load scalars from h5
        dset = h5_to_dict(pth, 'scalars')

        # Get ROI shape to compute distance to center
        try:
            roi = h5_to_dict(pth, path='metadata/extraction/roi')['roi']
            dset['dist_to_center_px'] = compute_mouse_dist_to_center(roi, dset['centroid_x_px'], dset['centroid_y_px'])
        except KeyError:
            print(f'ROI was not found in the given h5 file. \n'
                  f'Not including the dist_to_center_px column in outputted scalar_df for session-uuid {k}')
            pass

        timestamps = get_timestamps_from_h5(pth)

        # convert scalar names into modern format if they are legacy
        if is_legacy(dset) and force_conversion:
            dset = convert_legacy_scalars(dset, force=force_conversion)

        dset = merge(dset, {
            'group': v['group'],
            'uuid': k,
            'h5_path': pth,
            'timestamps': timestamps,
            'frame index': np.arange(len(timestamps))},
            keyfilter(lambda x: x in include_keys, v['metadata'])
        )

        try:
            _tmp_df = pd.DataFrame(dset)
        except ValueError as e:
            print(f'Error in session with uuid: {k}')
            if len(dset['timestamps']) != len(dset['centroid_x_px']):
                print('Length of timestamps do not equal number of frames. Skipping this session.')
            print(e)
            continue

        # make sure we have labels for this UUID before merging
        if has_model and k in labels_df.index:
            to_merge = labels_df.loc[k]
            # only drop labels if model was a separate_trans model
            if _tmp_df['group'].unique() !=  to_merge['group'].unique():
                if 'SeparateTrans' in model_type:
                    warnings.warn('Group labels from index.yaml and model results do not match! Setting group labels '
                                  'to ones used in the model.')
                    _tmp_df = _tmp_df.drop(columns=['group'])
                else:
                    print("Group name for UUID", k, "in the index.yaml file does not match the group name in the model file.")
                    print("Overwriting group label with those from the index.yaml file.")
                    to_merge = to_merge.drop(columns=['group'])
            else:
                to_merge = to_merge.drop(columns=['group'])

            _tmp_df = pd.merge(to_merge, _tmp_df, on=['frame index'], how='outer')
            _tmp_df = _tmp_df.sort_values(by='syllable index').reset_index(drop=True)

            # filter included keys to only those that exist in the dataset dataframe
            metadata_keys = list(filter(lambda x: x in _tmp_df.columns, include_keys)) + ['uuid', 'h5_path', 'group']
            # remove duplicates
            metadata_keys = list(set(metadata_keys))

            # fill any NaNs for metadata columns
            _tmp_df[metadata_keys] = _tmp_df[metadata_keys].ffill().bfill()
            # interpolate NaN timestamp values
            _tmp_df['timestamps'] = _tmp_df['timestamps'].interpolate()

        dfs.append(_tmp_df)
        warnings.filterwarnings('ignore', '', UserWarning)

    # return scalar_dict
    scalar_df = pd.concat(dfs, ignore_index=True)
    return scalar_df


def compute_all_pdf_data(scalar_df, normalize=False, centroid_vars=['centroid_x_mm', 'centroid_y_mm'],
                         key='SubjectName', bins=20):
    """
    Compute a position PDF for all sessions and return the pdfs with corresponding lists of groups, session uuids, and subjectNames.

    Args:
    scalar_df (pandas.DataFrame): DataFrame containing all scalar data + uuid columns for all stacked sessions
    normalize (bool): boolean flag that ndicates whether normalize the pdfs.
    centroid_vars (list): list of strings for column values to use when computing mouse position.
    key (str): metadata column to return info from.

    Returns:
    pdfs (list): list of 2d np.arrays of PDFs for each session.
    groups (list): list of strings of groups corresponding to pdfs index.
    sessions (list): list of strings of session uuids corresponding to pdfs index.
    subjectNames (list): list of strings of subjectNames corresponding to pdfs index.
    """

    sessions, groups, subjectNames, pdfs = [], [], [], []

    for uuid, _df in scalar_df.groupby('uuid', sort=False):
        sessions.append(uuid)
        groups.append(_df['group'].iat[0])
        subjectNames.append(_df[key].iat[0])

        pos = _df[centroid_vars].dropna(how='any')
        if len(pos) >= 1:
            H, _, _ = np.histogram2d(np.array(pos.iloc[:, 1]), np.array(pos.iloc[:, 0]), bins=bins, density=normalize)
        else:
            print(f'Failed to generate position heatmap for session with uuid: {uuid}')
            H = np.zeros((bins, bins))

        pdfs.append(H)

    return np.array(pdfs), groups, sessions, subjectNames


def compute_mean_syll_scalar(scalar_df, scalar='velocity_3d_mm', max_sylls=40, syllable_key='labels (usage sort)'):
    """
    Computes the mean syllable scalar-value based on the time-series scalar dataframe and the selected scalar.

    Args:
    scalar_df (pandas.DataFrame): DataFrame containing all scalar data + uuid and syllable columns for all stacked sessions (moseq_df)
    scalar (str or list): Selected scalar column(s) to compute mean value for syllables
    max_sylls (int): the index of the maximum syllable to include.
    syllable_key (str): column in scalar_df that points to the syllable labels to use.

    Returns:
    mean_df (pandas.DataFrame): updated input DataFrame with a speed value for each syllable merge in as a new column.
    """
    if syllable_key not in scalar_df:
        raise ValueError('scalar_df must be loaded with labels. Supply a model path to scalars_to_dataframe.')

    mask = (scalar_df[syllable_key] >= 0) & (scalar_df[syllable_key] <= max_sylls)
    mean_df = scalar_df[mask].groupby(['group', 'uuid', syllable_key])[scalar].mean()
    mean_df = mean_df.reset_index()

    return mean_df


def _column_mean(series):
    """
    Average one column the way ``DataFrame.mean()`` used to.

    Args:
    series (pandas.Series): a single column.

    Returns:
    the column's mean, raising TypeError when it has none.

    An object column is summed and then coerced, which is what makes a column of
    2-D arrays average elementwise, and what lets a column of digit strings
    average into a number. pandas stopped coercing strings and stopped reducing
    object columns at all, so both have to be done here.
    """

    values = series.to_numpy()
    if values.dtype != object:
        return series.mean()

    missing = pd.isna(values)
    if missing.any():
        values = values.copy()
        np.putmask(values, missing, 0)
    total = values.sum(axis=0)
    if isinstance(total, str):
        total = float(total)
    return total / float((~missing).sum())


def _frame_mean(frame):
    """
    Average every column that can be averaged, and drop the rest.

    Args:
    frame (pandas.DataFrame): the group being reduced.

    Returns:
    (pandas.Series): one entry per column that reduced.

    Silently dropping the columns that raise is what ``DataFrame.mean()`` did
    before it started refusing the whole frame over any one bad column.
    """

    means = {}
    for column in frame.columns:
        try:
            means[column] = _column_mean(frame[column])
        except (TypeError, ValueError):
            continue
    return pd.Series(means)


def get_syllable_pdfs(pdf_df, normalize=True, syllables=range(40), groupby='group',
                      syllable_key='labels (usage sort)'):
    """

    Compute the mean syllable position PDF/Heatmap for the given groupings.

    Args:
    pdf_df (pandas.DataFrame): model results dataframe including a position PDF column containing 2D numpy arrays.
    normalize (bool): boolean flag that indicates whether normalize the pdf scales.
    syllables (list): list of syllables to get a grouping of.
    groupby (str): column name to group the df keys by. (either group, or SessionName)
    syllable_key (str): name of the column that contains the requested syllable label sequences.

    Returns:
    group_syll_pdfs (list): 2D list of computed pdfs of shape ngroups x nsyllables
    groups (list): list of corresponding names to each row in the group_syll_pdfs list
    """

    # Get unique groups to iterate by
    groups = pdf_df[groupby].unique()
    # Selecting every column keeps the grouping columns in the result, which is
    # where the consumer reads the syllable label back from.
    mean_pdfs = (pdf_df.groupby([groupby, syllable_key])[list(pdf_df.columns)]
                 .apply(_frame_mean))

    if normalize:
        mean_pdfs['pdf'] = mean_pdfs['pdf'].apply(lambda x: x / np.nanmax(x))

    if groupby in mean_pdfs.columns:
        mean_pdfs = mean_pdfs.drop(groupby, axis=1)

    return mean_pdfs, groups


def compute_syllable_position_heatmaps(scalar_df, syllable_key='labels (usage sort)', syllables=range(40),
                                       centroid_keys=['centroid_x_mm', 'centroid_y_mm'], normalize=False, bins=20):
    """
    Compute position heatmaps for each syllable on a session-by-session basis

    Args:
    scalar_df (pandas.DataFrame): dataframe containing scalar data & labels for all sessions
    syllable_key (str): dataframe column to access syllable labels
    syllables (list): List of syllables to compute heatmaps for.
    centroid_keys (list): list of column names containing the centroid values used to compute mouse position.
    normalize (bool): boolean flag for whether normalizes the histogram to be a probability density
    bins (int): number of bins to cut the position data into

    Returns:
    complete_df (pandas.DataFrame): model results dataframe with a new PDF column corresponding to each session-syllable pair.
    """
    if syllable_key not in scalar_df:
        raise ValueError('You need to supply a model path to `scalars_to_dataframe` in order to merge syllable labels into `scalar_df`')

    def _compute_histogram(df):
        centroid_df = df[centroid_keys].dropna(how='any')

        if len(centroid_df) >= 1:
            H, _, _ = np.histogram2d(np.array(centroid_df.iloc[:, 1]), np.array(centroid_df.iloc[:, 0]), bins=bins, density=normalize)
        else:
            # syllable not found in group
            print(f'Unable to generate position heatmap for syllable {df[syllable_key].to_numpy()}')
            H = np.zeros((bins, bins))

        return H

    filtered_df = scalar_df[scalar_df[syllable_key].isin(syllables)]
    hists = filtered_df.groupby(['group', 'uuid', 'SessionName', 'SubjectName', syllable_key]).apply(_compute_histogram)

    return hists
