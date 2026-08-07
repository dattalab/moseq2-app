"""
Data pre-processing and visualization functions for model results.
"""

import os
import cv2
import h5py
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from tqdm.auto import tqdm
from os.path import dirname
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from moseq2_viz.model.util import sort_syllables_by_stat, sort_syllables_by_stat_difference


def _validate_and_order_syll_stats_params(complete_df, stat='usage', ordering='stat', max_sylls=40, groups=None, ctrl_group=None, exp_group=None,
            colors=None, figsize=(10, 5)):
    """
    Validate input parameters, reset incorrect parameters and order the syllables based on stat parameter.

    Args:
    complete_df (pandas.DataFrame): dataframe containing the summary statistics about scalars and syllable data (mean_df/stats_df)
    stat (str): choice of statistic to plot eg. usage, duration, or speed
    ordering (str, list, None): statistics for sorting syllables on or a list for syllable ordering.
    max_sylls (int): the index of the maximum number of syllables to include
    groups (list): list of groups to include in plot. If groups=None, all groups will be plotted.
    ctrl_group (str): name of control group for computing usage difference beween two groups.
    exp_group (str): name of experimental group for computing usage difference beween two groups.
    colors (list): list of user-selected colors to represent the data
    figsize (tuple): tuple value of length = 2, representing (height x width) of the plotted figure dimensions

    Returns:
    ordering (list): list of syllable indices to display on x-axis
    groups (list): list of unique groups to plot
    colors (list): list of unique colors for each plotted group
    figsize (tuple): plotted figure size (height, width)
    """

    if not isinstance(figsize, (tuple, list)):
        print('Invalid figsize. Input a integer-tuple or list of len(figsize) = 2. Setting figsize to (10, 5)')
        figsize = (10, 5)

    unique_groups = complete_df['group'].unique()

    if groups is None or len(groups) == 0:
        groups = unique_groups
    elif isinstance(groups, str):
        groups = [groups]

    if isinstance(groups, (list, tuple, np.ndarray)):
        diff = set(groups) - set(unique_groups)
        if len(diff) > 0:
            warnings.warn(f'Invalid group(s) entered: {", ".join(diff)}. Using all groups: {", ".join(unique_groups)}.')
            groups = unique_groups

    if stat.lower() not in complete_df.columns:
        raise ValueError(f'Invalid stat entered: {stat}. Must be a column in the supplied dataframe.')

    if ordering is None:
        ordering = np.arange(max_sylls)
    elif ordering == "stat":
        ordering, _ = sort_syllables_by_stat(complete_df, stat=stat, max_sylls=max_sylls)
    elif ordering == "diff":
        if ctrl_group is None or exp_group is None or not np.all(np.isin([ctrl_group, exp_group], groups)):
            raise ValueError(f'Attempting to sort by {stat} differences, but {ctrl_group} or {exp_group} not in {groups}.')
        ordering = sort_syllables_by_stat_difference(complete_df, ctrl_group, exp_group,
                                                     max_sylls=max_sylls, stat=stat)
    if colors is None:
        colors = []
    if len(colors) == 0 or len(colors) != len(groups):
        if len(colors) != len(groups):
            warnings.warn(f'Number of inputted colors {len(colors)} does not match number of groups {len(groups)}. Using default.')
        colors = sns.color_palette(n_colors=len(groups))

    return ordering, groups, colors, figsize


def clean_frames(frames, medfilter_space=None, gaussfilter_space=None,
                 tail_filter=None, tail_threshold=5):
    """
    Smooth frames using Median or Gaussian filters.

    Args:
    frames (numpy.array): frames to filter.
    medfilter_space (list): Median space filter kernel size, list of length 1.
    gaussfilter_space (list): x sigma and y sigma for Gaussian space filter, list of length 2.
    tail_filter (cv2.getStructuringElement): structuringElement to filter out mouse tails.
    tail_threshold (int): filtering threshold value

    Returns:
    out (numpy.array): filtered numpy array.
    """

    out = np.copy(frames)


    if tail_filter is not None:
        for i in range(frames.shape[0]):
            mask = cv2.morphologyEx(out[i], cv2.MORPH_OPEN, tail_filter) > tail_threshold
            out[i] = out[i] * mask.astype(frames.dtype)

    if medfilter_space is not None and np.all(np.array(medfilter_space) > 0):
        for i in range(frames.shape[0]):
            for medfilt in medfilter_space:
                if medfilt % 2 == 0:
                    warnings.warn(f'medfilter_space kernel must be odd. Reducing {medfilt} to {medfilt - 1}')
                    medfilt -= 1
                out[i] = cv2.medianBlur(out[i], medfilt)

    if gaussfilter_space is not None and np.all(np.array(gaussfilter_space) > 0):
        for i in range(frames.shape[0]):
            out[i] = cv2.GaussianBlur(out[i], (21, 21),
                                      gaussfilter_space[0], gaussfilter_space[1])

    return out


def save_fig(fig, output_file, suffix=None, **kwargs):
    """
    Save matplotlib figures to PNG and PDF formats.

    Args:
    fig (pyplot.Figure): open figure to save
    output_file (str): file name for the figure without the file extension
    suffix (str): string to append to the end of output_file
    kwargs (dict): dictionary containing additional figure saving parameters.

    Returns:
    """

    os.makedirs(dirname(output_file), exist_ok=True)

    if suffix is not None:
        output_file = output_file + suffix

    fig.savefig(f'{output_file}.png', **kwargs)
    fig.savefig(f'{output_file}.pdf', **kwargs)


def make_crowd_matrix(slices, nexamples=50, pad=30, raw_size=(512, 424), outmovie_size=(300, 300), frame_path='frames',
                      crop_size=(80, 80), max_dur=60, min_dur=0, scale=1,
                      center=False, rotate=False, select_median_duration_instances=False, min_height=10, legacy_jitter_fix=False,
                      seed=0, **kwargs):
    """
    Creates crowd movie video numpy array.

    Args:
    slices (np.ndarray): video slices of specific syllable label
    nexamples (int): maximum number of mice to include in crowd_matrix video
    pad (int): number of frame padding in video
    raw_size (tuple): video dimensions.
    frame_path (str): variable to access frames in h5 file
    crop_size (tuple): mouse crop size
    max_dur (int or None): maximum syllable duration.
    min_dur (int): minimum syllable duration.
    scale (int): mouse size scaling factor.
    center (bool): boolean flag that indicates whether mice are centered.
    rotate (bool): boolean flag that indicates wehther to rotate mice and orient them.
    select_median_duration_instances (bool): flag that indicates wehther to select examples with syallable duration closer to median.
    min_height (int): minimum max height from floor to use.
    legacy_jitter_fix (bool): flag that indicates wehther to apply jitter fix for K1 camera.
    kwargs (dict): extra keyword arguments

    Returns:
    crowd_matrix (np.ndarray): crowd movie for a specific syllable.
    """

    if rotate and not center:
        raise NotImplementedError('Rotating without centering not supported')

    rng = np.random.default_rng(seed)

    # set up x, y value to crop out the mouse with respect to the mouse centriod
    xc0, yc0 = crop_size[1] // 2, crop_size[0] // 2
    xc = np.arange(-xc0, xc0 + 1, dtype='int16')
    yc = np.arange(-yc0, yc0 + 1, dtype='int16')

    # compute syllable duration in the sample
    durs = np.array([i[1]-i[0] for i, _, _ in slices])
    
    if max_dur is not None:
        idx = np.where(np.logical_and(durs < max_dur, durs > min_dur))[0]
        use_slices = [_slice for i, _slice in enumerate(slices) if i in idx]
        # return the sort order of durations of the remaining slices
        dur_order = np.argsort(durs[idx])
    else:
        max_dur = durs.max()
        idx = np.where(durs > min_dur)[0]
        use_slices = [_slice for i, _slice in enumerate(slices) if i in idx]
        # return the sort order of durations of the remaining slices
        dur_order = np.argsort(durs[idx])

    if len(use_slices) > nexamples:
        if select_median_duration_instances:
            # choose the nexamples near median duration
            selction_begin = int(len(dur_order)//2 - nexamples//2)
            # ensure nexamples are picked
            selection_end = selction_begin + nexamples
            use_slices = [_slice for i, _slice in enumerate(use_slices) if i in dur_order[selction_begin:selection_end]]
        else:
            # Permute positions rather than the list itself: the slices are
            # ragged, so numpy 2 will not build an array from them. Drawing the
            # permutation over len(use_slices) consumes the generator
            # identically, so the selected order is unchanged.
            order = rng.permutation(len(use_slices))[:nexamples]
            use_slices = [use_slices[i] for i in order]

    if len(use_slices) == 0 or max_dur < 0:
        return None

    crowd_matrix = np.zeros((max_dur + pad * 2, raw_size[1], raw_size[0]), dtype='uint8')

    for idx, _, fname in use_slices:
        # pad frames before syllable onset, and add max_dur and padding after syllable onset
        use_idx = (idx[0] - pad, idx[0] + max_dur + pad)
        idx_slice = slice(*use_idx)

        # get the frames, combine in a way that's alpha-aware
        with h5py.File(fname, 'r') as h5:
            nframes = len(h5[frame_path])

            if 'centroid_x' in h5['scalars']:
                use_names = ('scalars/centroid_x', 'scalars/centroid_y')
            elif 'centroid_x_px' in h5['scalars']:
                use_names = ('scalars/centroid_x_px', 'scalars/centroid_y_px')

            if use_idx[0] < 0 or use_idx[1] >= nframes - 1:
                continue
            
            # select centroids
            centroid_x = h5[use_names[0]][idx_slice]
            centroid_y = h5[use_names[1]][idx_slice]

            # center the mice such that when it is syllable onset, the mice's centroids are in the center
            if center:
                centroid_x -= centroid_x[pad]
                centroid_x += raw_size[0] // 2
                centroid_y -= centroid_y[pad]
                centroid_y += raw_size[1] // 2

            angles = h5['scalars/angle'][idx_slice]
            frames = clean_frames((h5[frame_path][idx_slice] / scale).astype('uint8'), **kwargs)

            # flip the mouse in the correct orientation if necessary
            if 'flips' in h5['metadata/extraction']:
                # h5 format as of v0.1.3
                flips = h5['metadata/extraction/flips'][idx_slice]
                angles[np.where(flips == True)] -= np.pi
            elif 'flips' in h5['metadata']:
                # h5 format prior to v0.1.3
                flips = h5['metadata/flips'][idx_slice]
                angles[np.where(flips == True)] -= np.pi
            else:
                flips = np.zeros(angles.shape, dtype='bool')

        angles = np.rad2deg(angles)

        for i in range(len(centroid_x)):

            if np.any(np.isnan([centroid_x[i], centroid_y[i]])):
                continue
            
            # set up the rows and columnes to crop the video
            rr = (yc + int(centroid_y[i])).astype('int16')
            cc = (xc + int(centroid_x[i])).astype('int16')
            if np.any(rr >= raw_size[1]) or np.any(cc >= raw_size[0]): continue

            if ((rr[-1] - rr[0]) != crop_size[0]) or ((cc[-1] - cc[0]) != crop_size[1]):
                continue

            if np.any(rr < 0) or np.any(cc < 0):
                top = 0
                if np.any(rr < 0):
                    top = rr.min()
                    rr = rr - rr.min()
                left = 0
                if np.any(cc < 0):
                    left = cc.min()
                    cc = cc - cc.min()
                new_frame_clip = cv2.copyMakeBorder(frames[i].copy(), abs(top), 0, abs(left), 0, cv2.BORDER_CONSTANT, value=0)
                if left > 0:
                    new_frame_clip = new_frame_clip[:, :crop_size[1]]
                if top > 0:
                    new_frame_clip = new_frame_clip[:crop_size[0]]
            else:
                new_frame_clip = frames[i].copy()

            rot_mat = cv2.getRotationMatrix2D((xc0, yc0), angles[i], 1)

            # add the new instance to the exisiting crowd matrix
            old_frame = crowd_matrix[i]
            new_frame = np.zeros_like(old_frame)

            # change from fliplr, removes jitter since we now use rot90 in moseq2-extract
            if flips[i] and legacy_jitter_fix:
                new_frame_clip = np.fliplr(new_frame_clip)
            elif flips[i]:
                new_frame_clip = np.rot90(new_frame_clip, k=-2)

            new_frame_clip = cv2.warpAffine(new_frame_clip.astype('float32'),
                                            rot_mat, crop_size).astype(frames.dtype)

            if i >= pad and i <= pad + (idx[1] - idx[0]):
                cv2.circle(new_frame_clip, (xc0, yc0), 3, (255, 255, 255), -1)
            
            new_frame[rr[0]:rr[-1], cc[0]:cc[-1]] = new_frame_clip

            if rotate:
                rot_mat = cv2.getRotationMatrix2D((raw_size[0] // 2, raw_size[1] // 2),
                                                -angles[pad] + flips[pad] * 180,
                                                1)
                new_frame = cv2.warpAffine(new_frame, rot_mat, raw_size).astype(new_frame.dtype)

            # zero out based on min_height before taking the non-zeros
            new_frame[new_frame < min_height] = 0
            old_frame[old_frame < min_height] = 0

            new_frame_nz = new_frame > 0
            old_frame_nz = old_frame > 0

            blend_coords = np.logical_and(new_frame_nz, old_frame_nz)
            overwrite_coords = np.logical_and(new_frame_nz, ~old_frame_nz)

            old_frame[blend_coords] = .5 * old_frame[blend_coords] + .5 * new_frame[blend_coords]
            old_frame[overwrite_coords] = new_frame[overwrite_coords]

            crowd_matrix[i] = old_frame
    
    # compute non-zero pixels across all frames
    non_zero_coor = np.argwhere(np.any(crowd_matrix>0, 0))
    
    try:
        # find min max coordinates to crop out the blanks
        min_xy = np.min(non_zero_coor, 0)
        max_xy = np.max(non_zero_coor, 0)  
        # crop out the blanks while making sure the slice is not zero
        if np.all(max_xy - min_xy) > 0:
            crowd_matrix = crowd_matrix[:, min_xy[0]:max_xy[0], min_xy[1]:max_xy[1]]
            
            # pad crowd movies to outmovie_size if the dimension is smaller than outmoive_size
            if np.all(outmovie_size > max_xy - min_xy):
                x_pad, y_pad = (outmovie_size - (max_xy - min_xy))//2
                crowd_matrix = np.pad(crowd_matrix, ((0,0), (x_pad, x_pad), (y_pad, y_pad)), 'constant', constant_values=0)
                
    except ValueError:
        print('No mouse in the crowd movie')

    return crowd_matrix


def position_plot(scalar_df, centroid_vars=['centroid_x_mm', 'centroid_y_mm'],
                  sort_vars=['SubjectName', 'uuid'], group_var='group', plt_kwargs=dict(linewidth=1)):
    """
    Create a position summary graph that shows all the mice's centroid path throughout the respective sessions.

    Args:
    scalar_df (pandas.DataFrame): dataframe containing all scalar data
    centroid_vars (list): list of scalar variables to track mouse position
    sort_vars (list): list of variables to sort the dataframe by.
    group_var (str): groups df column to graph position plots for.
    plt_kwargs (dict): extra keyword arguments for plt.plot

    Returns:
    fig (pyplot.figure): matplotlib figure object
    ax (pyplot.axis): matplotlib axis object
    g (sns.FacetGrid): FacetGrid object the data was plotted with
    """

    assert len(centroid_vars) == 2, 'must supply 2 centroid vars (x, y) to plot position'

    if isinstance(sort_vars, str) :
        sort_vars = [sort_vars]
    else:
        sort_vars = list(sort_vars)

    scalar_df = scalar_df.sort_values(by=[group_var] + sort_vars)

    if 'uuid' in sort_vars:
        uuid_map = scalar_df.groupby('uuid').first()
    
    g = sns.FacetGrid(data=scalar_df, col='uuid', col_wrap=5, height=2.5, hue=group_var)
    g.map(plt.plot, centroid_vars[0], centroid_vars[1], **plt_kwargs)
    g.set_titles(template='{col_name}')
    for i, a in enumerate(g.axes.flat):
        a.set_title(f"{uuid_map.iloc[i]['SubjectName']}\n{uuid_map.iloc[i]['SessionName']}", fontsize=8)
        a.set_aspect('equal')
    g.fig.tight_layout()
    g.add_legend()

    return g.fig, g.axes, g


def scalar_plot(scalar_df, sort_vars=['group', 'uuid'], group_var='group',
                show_scalars=['velocity_2d_mm', 'velocity_3d_mm',
                              'height_ave_mm', 'width_mm', 'length_mm'],
                headless=False, colors=None, plt_kwargs=dict(height=2, aspect=0.8)):
    """
    Create scatter plot of given scalar variables representing extraction results.

    Args:
    scalar_df (pandas.DataFrame): dataframe containing scalar data.
    sort_vars (list): list of variables to sort the dataframe by.
    group_var (str): groups scalar plots into separate distributions.
    show_scalars (list): list of scalar variables to plot.
    headless (bool): bool flag to run in headless environment
    colors (list): list of color strings to indicate groups
    plt_kwargs (dict): extra arguments for the swarmplot

    Returns:
    fig (pyplot figure): plotted scalar scatter plot
    ax (pyplot axis): plotted scalar axis
    """
    if headless:
        plt.switch_backend('agg')

    if colors is None or len(colors) == 0:
        colors = sns.color_palette()

    plt_kwargs['aspect'] = 0.6 * len(scalar_df[group_var].unique())

    # sort scalars into a neat summary using group_vars
    summary = scalar_df.groupby(sort_vars)[show_scalars].aggregate(['mean', 'std']).reset_index()
    # aggregate() leaves MultiIndex columns, so the names here are tuples such
    # as ('velocity_2d_mm', 'mean'). pandas 1.x matched bare strings against
    # the first level; pandas 2 requires the full label. Selecting the tuples
    # and renaming the id column back reproduces the 1.x result exactly:
    # columns group, variable_0, variable_1, value.
    id_cols = [c for c in summary.columns if c[0] == group_var]
    value_cols = [c for c in summary.columns if c[0] in show_scalars]
    summary = summary.melt(id_vars=id_cols, value_vars=value_cols)
    summary = summary.rename(columns={id_cols[0]: group_var})
    groups = summary[group_var].unique()
    
    g = sns.FacetGrid(data=summary, row='variable_0', col='variable_1', sharey=False,
                      hue=group_var, hue_order=groups, palette=colors, **plt_kwargs)
    g.map(sns.swarmplot, group_var, 'value', order=groups)
    g.set_titles(template='{col_name}')
    for row_name, a in zip(show_scalars, g.axes[:, 0]):
        a.set_ylabel(row_name)

    # rotate x-axis labels
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    g.fig.tight_layout()

    return g.fig, g.axes


def plot_syll_stats_with_sem(scalar_df, syll_info=None, sig_sylls=None, stat='usage', ordering='stat', max_sylls=40,
                             groups=None, ctrl_group=None, exp_group=None, colors=None, join=False, figsize=(10, 5)):
    """
    Plot a line and/or point-plot of a given pre-computed syllable statistic (usage, duration, or speed), with a SEM error bar with respect to the group.

    Args:
    scalar_df (pandas.DataFrame): dataframe containing the summary statistics about scalars and syllable data (mean_df/stats_df)
    syll_info (dict): dictionary of syllable numbers mapped to dict containing the label, description and crowd movie path.
    sig_sylls (list): List of syllable numbers that are statistically significant to optionally mark in the graph.
    stat (str): choice of statistic to plot: either usage, duration, or speed
    ordering (str, list, None): statistics for sorting syllables on or the list of syllable orders.
    max_sylls (int): the index of the maximum number of syllables to include
    groups (list): list of groups to include in plot. If groups=None, all groups will be plotted.
    ctrl_group (str): Control group to graph.
    exp_group (str): Experimental group to compare with control group.
    colors (list): list of user-selected colors to represent the data
    join (bool): flag that indicate whether to connect points of pointplot
    figsize (tuple): tuple value of length of 2, representing (columns x rows) of the plotted figure dimensions

    Returns:
    fig (pyplot.figure): plotted scalar scatter plot
    legend (pyplot.legend): figure legend
    """

    xlabel = f'Syllables sorted by {stat}'
    if ordering == 'diff':
        xlabel += ' difference'
    ordering, groups, colors, figsize = _validate_and_order_syll_stats_params(scalar_df,
                                                                              stat=stat,
                                                                              ordering=ordering,
                                                                              max_sylls=max_sylls,
                                                                              groups=groups,
                                                                              ctrl_group=ctrl_group,
                                                                              exp_group=exp_group,
                                                                              colors=colors,
                                                                              figsize=figsize)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # plot each group's stat data separately, computes groupwise SEM, and orders data based on the stat/ordering parameters
    hue = 'group' if groups is not None else None
    # dodge offsets the hue levels by dodge/(n_levels - 1), so asking for it
    # with a single group divides by zero from seaborn 0.12 onward. With one
    # group there is nothing to offset, which is what older seaborn did here.
    dodge = len(groups) > 1
    # errorbar=('ci', 68) is seaborn's own stated replacement for ci=68, and
    # join=False corresponds to suppressing the connecting line. Both spellings
    # still work at 0.13 but are removed in 0.15.
    ax = sns.pointplot(data=scalar_df, x='syllable', y=stat, hue=hue, order=ordering,
                       linestyle='-' if join else 'none', dodge=dodge,
                       errorbar=('ci', 68), ax=ax, hue_order=groups,
                       palette=colors)

    # where some data has already been plotted to ax
    handles, labels = ax.get_legend_handles_labels()

    # add syllable labels if they exist
    if syll_info is not None:
        mean_xlabels = []
        for o in (ordering):
            mean_xlabels.append(f'{syll_info[o]["label"]} - {o}')

        plt.xticks(range(max_sylls), mean_xlabels, rotation=90)

    # if a list of significant syllables is given, mark the syllables above the x-axis
    if sig_sylls is not None:
        markings = []
        for s in sig_sylls:
            markings.append(ordering.index(s))
        plt.scatter(markings, [-.005] * len(markings), color='r', marker='*')

        # manually define a new patch
        patch = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                              markersize=9, label='Significant Syllable')
        handles.append(patch)

    # add legend and axis labels
    legend = ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1, 1))
    plt.xlabel(xlabel, fontsize=12)
    sns.despine()

    return fig, legend


def plot_mean_group_heatmap(pdfs, groups, normalize=True, norm_color=mpl.colors.LogNorm()):
    """
    Compute and plot the overall group mean of the computed PDFs.

    Args:
    pdfs (list): list of 2d probability density functions (heatmaps) describing mouse position.
    groups (list): list of groups to compute means and plot
    normalize (bool): flag to normalize the pdfs between 0-1
    norm_color (mpl.colors Color Scheme or None): indicates a color scheme to use when plotting heatmaps.

    Returns:
    fig (pyplot figure): plotted scalar scatter plot
    """

    uniq_groups = np.unique(groups)
    groups = np.array(groups)
    pdfs = np.array(pdfs)

    fig, ax = plt.subplots(nrows=len(uniq_groups), ncols=1, sharex=True, sharey=True,
                           figsize=(5, 6 * len(uniq_groups)))
    if not isinstance(ax, np.ndarray):
        ax = np.array(ax)
    for a, group in zip(ax.flat, uniq_groups):
        idx = groups == group
        avg_hist = pdfs[idx].mean(axis=0)
        if normalize:
            _min_val = (avg_hist[avg_hist > 0]).min()
            avg_hist = (avg_hist + _min_val) / (avg_hist.max() + _min_val)

        im = a.imshow(avg_hist, norm=norm_color)
        # fraction to make the colorbar match image height
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, format='%.0e')

        a.set_xticks([])
        a.set_yticks([])
        a.set_title(group, fontsize=14)

    return fig


def plot_verbose_heatmap(pdfs, sessions, groups, subjectNames, normalize=False, norm_color=mpl.colors.LogNorm()):
    """
    Plot the PDF position heatmap for each session, titled with the group and subjectName.

    Args:
    pdfs (list): list of 2d probability density functions (heatmaps) describing mouse position.
    sessions (list): list of sessions corresponding to the pdfs indices
    groups (list): list of groups corresponding to the pdfs indices
    subjectNames (list): list of subjectNames corresponding to the pdfs indices
    normalize (bool): flag to normalize the pdfs between 0-1
    norm_color (mpl.colors object): indicates a color scheme to use when plotting heatmaps.

    Returns:
    fig (pyplot.figure): plotted scalar scatter plot
    """

    uniq_groups = np.unique(groups)
    count = [len([grp1 for grp1 in groups if grp1 == grp]) for grp in uniq_groups]
    figsize = (np.round(2.8 * len(uniq_groups)), np.round(3.0 * np.max(count)))

    fig, ax = plt.subplots(nrows=np.max(count), ncols=len(uniq_groups), sharex=True,
                           sharey=True, figsize=figsize)

    if not isinstance(ax, np.ndarray):
        ax = np.array([[ax]])
    if ax.ndim < 2:
        ax = ax[:, None]
    for i, group in enumerate(tqdm(uniq_groups, desc='Plotting Session Heatmaps')):
        idx = np.array(groups) == group
        tmp_sessions = np.asarray(sessions)[idx]
        names = np.asarray(subjectNames)[idx]
        for sess, name, a in zip(tmp_sessions, names, ax[:, i]):
            idx = np.array(sessions) == sess

            avg_hist = pdfs[idx].mean(axis=0)

            if normalize:
                _min_val = (avg_hist[avg_hist > 0]).min()
                avg_hist = (avg_hist + _min_val) / (avg_hist.max() + _min_val)

            im = a.imshow(avg_hist, norm=norm_color)
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, format='%.0e')

            a.set_xticks([])
            a.set_yticks([])

            a.set_title(f'{group}: {name}', fontsize=10)

    return fig


def plot_cp_comparison(model_results, pc_cps, plot_all=False, best_model=None, bw_adjust=0.4):
    """
    Plot the duration distributions for model labels and model free changepoints.

    Args:
    model_cps (dict): Multiple parsed model results aggregated into a single dict.
    pc_cps (np.array): Computed PC changepoints
    plot_all (bool): Boolean flag that indicates whether to plot all model changepoints for all keys included in model_cps dict.
    best_model (str): key name to the model with the closest median syllable duration
    bw_adjust (float): fraction to modify bandwith of kernel density estimate. (lower = higher definition)

    Returns:
    fig (pyplot.figure): syllable usage ordered by frequency, 90% usage marked
    ax (pyplot.axis): plotted scalar axis
    """

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # Plot KDEs, clipping the changepoints at 10 seconds
    ax = sns.kdeplot(pc_cps[pc_cps<10], color='orange', label='PCA Changepoints', ax=ax, bw_adjust=bw_adjust)

    _mdl = model_results[best_model]
    model_cps = _mdl['changepoints']
    kappa = _mdl['model_parameters']['kappa']

    # Plot best model description
    s = f'Best Model CP Stats: Mean, median, mode (s) = {np.nanmean(model_cps):.4f},' \
        f' {np.nanmedian(model_cps):.4f}, {mode(model_cps, keepdims=True)[0][0]:.4f}'
    # Plot PC CP description
    t = f'PC CP Stats: Mean, median, mode (s) = {np.nanmean(pc_cps):.4f}, ' \
        f'{np.nanmedian(pc_cps):.4f}, {mode(pc_cps, keepdims=True)[0][0]:.4f}'
    
    # saving model kappa scan stats
    model_stats = None
    # record changepoint stats
    changepoint_stats = [np.nanmean(pc_cps), np.nanmedian(pc_cps), mode(pc_cps, keepdims=True)[0][0]]

    if not plot_all and best_model is not None:
        # clipping the changepoints at 10 seconds
        ax = sns.kdeplot(model_cps[model_cps<10], ax=ax, color='blue', label=f'Model Changepoints Kappa={kappa:1.02E}',
                         bw_adjust=bw_adjust)
    else:
        model_stats = []
        palette = sns.color_palette('dark', n_colors=len(model_results))
        for i, (k, v) in enumerate(model_results.items()):
            # Set default curve formatting
            ls, alpha = '--', 0.5
            if k == best_model:
                ls, alpha = '-', 1 # Solid line for best fit
            kappa = v['model_parameters']['kappa']
            model_cps = v['changepoints']
            # append model stats
            model_stats.append([k, kappa, np.nanmean(model_cps), np.nanmedian(model_cps), mode(model_cps, keepdims=True)[0][0]]+ changepoint_stats)
            # clipping the changepoints at 10 seconds
            ax = sns.kdeplot(model_cps[model_cps<10], ax=ax, linestyle=ls, alpha=alpha, bw_adjust=bw_adjust,
                        color=palette[i], label=f'Model Changepoints Kappa={kappa:1.02E}')
        model_stats = pd.DataFrame(model_stats, columns=['file_name', 'model_kappa', 'model_mean', 'model_median', 'model_mode', 'pc_mean', 'pc_median', 'pc_mode'])

    # Format plot
    ax.set_xlim(0, 2)

    # get figure maximum y-value
    _, xmax = ax.get_xlim()
    _, ymax = ax.get_ylim()

    # space out the 2 plotted text glyphs
    ax.text(xmax, ymax, s,
            horizontalalignment='right',
            verticalalignment='top',
            fontsize=12)
    ax.text(xmax, ymax, t,
            horizontalalignment='right',
            verticalalignment='bottom',
            fontsize=12)

    ax.set_xlabel('Block duration (s)')
    ax.set_ylabel('Probability density')
    ax.legend(frameon=False, bbox_to_anchor=(1, 0), loc='lower left')
    sns.despine()

    return fig, ax, model_stats