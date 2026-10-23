"""
Functions for creating fingerprint plots and linear classifier
"""

from operator import pos
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneOut,
    RepeatedStratifiedKFold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.base import clone

import pandas as pd
import numpy as np
from os.path import join
from os import makedirs


def robust_min(v):
    """find relative min

    Args:
        v (pandas.Series): the pandas.Series to find the robust min.

    Returns:
        (numpy.float): the robust min in the series.
    """
    return v.quantile(0.01)


def robust_max(v):
    """find relative max

    Args:
        v (pandas.Series): the pandas.Series to find the robust max.

    Returns:
        (numpy.float): the robust max in the series.
    """
    return v.quantile(0.99)


def _apply_to_col(df, fn, **kwargs):
    """apply function to columns

    Args:
        df (pandas.DataFrame): dataframe to apply the function on
        fn (function): the function to apply to the columns

    Returns:
        (pandas.DataFrame): dataframe after applying the functions
    """
    return df.apply(fn, axis=0, **kwargs)


def create_fingerprint_dataframe(
    scalar_df,
    mean_df,
    stat_type="mean",
    n_bins=None,
    groupby_list=["group", "uuid"],
    range_type="robust",
    scalars=["velocity_2d_mm", "height_ave_mm", "length_mm", "dist_to_center_px"],
):
    """
    create fingerprint dataframe from scalar_df and mean_df

    Args:
        scalar_df (pandas.DataFrame): scalar summary dataframe generated from scalars_to_dataframe
        mean_df (pandas.DataFrame): syllable mean dataframe from compute_behavioral_statistics
        bin_num (int, optional): number of bins for the features. Defaults to None.
        groupby_list (list, optional): the list of levels the fingerprint dataframe should be grouped by. Defaults to ['group', 'uuid'].

    Returns:
        summary ([pandas.DataFrame]): fingerprint dataframe
        range_dict ([dict]): dictionary that hold min max values of the features
    """

    # deep copy the dfs
    scalar_df = scalar_df.copy()
    mean_df = mean_df.copy()
    # rescale velocity to cm/s
    vel_cols = [c for c in scalars if "velocity" in c]
    vel_cols_stats = [f"{c}_{stat_type}" for c in scalars if "velocity" in c]

    if len(vel_cols) > 0:
        scalar_df[vel_cols] *= 30
        try:
            mean_df[vel_cols_stats] *= 30
        except KeyError:
            mean_df[vel_cols] *= 30

    # pivot mean_df to be groupby x syllable
    syll_summary = mean_df.pivot_table(
        index=groupby_list, values="usage", columns="syllable"
    )
    syll_summary.columns = pd.MultiIndex.from_arrays(
        [["MoSeq"] * syll_summary.shape[1], syll_summary.columns]
    )
    min_p = syll_summary.min().min()
    max_p = syll_summary.max().max()

    # The aggregate is widened to double: the scalar columns are float32, and
    # the ranges are carried alongside the syllable range appended below, which
    # is a Python float.
    ranges = scalar_df.reset_index(drop=True)[scalars].agg(
        ["min", "max", robust_min, robust_max]
    ).astype("float64")
    # add syllable ranges to this df
    ranges["MoSeq"] = [min_p, max_p, min_p, max_p]
    range_idx = ["min", "max"] if range_type == "full" else ["robust_min", "robust_max"]

    def bin_scalars(data: pd.Series, n_bins=50, range_type="full"):
        _range = ranges.loc[range_idx, data.name]
        bins = np.linspace(_range.iloc[0], _range.iloc[1], n_bins)

        binned_data = data.value_counts(normalize=True, sort=False, bins=bins)
        binned_data = binned_data.sort_index().reset_index(drop=True)
        binned_data.index.name = "bin"
        return binned_data

    # use total number of syllables
    if n_bins is None:
        n_bins = (
            syll_summary.shape[1] + 1
        )  # num of bins (default to match the total number of syllables)

    binned_scalars = scalar_df.groupby(groupby_list)[scalars].apply(
        _apply_to_col, fn=bin_scalars, range_type=range_type, n_bins=n_bins
    )

    scalar_fingerprint = binned_scalars.pivot_table(
        index=groupby_list, columns="bin", values=binned_scalars.columns
    )

    fingerprints = scalar_fingerprint.join(syll_summary, how="outer")

    return fingerprints, ranges.loc[range_idx]


def plotting_fingerprint(
    summary,
    save_dir,
    range_dict,
    figsize=(20, 18),
    preprocessor=None,
    num_level=1,
    level_names=["Group"],
    vmin=None,
    vmax=None,
    plot_columns=[
        "dist_to_center_px",
        "velocity_2d_mm",
        "height_ave_mm",
        "length_mm",
        "MoSeq",
    ],
    col_names=[
        ("Position", "Dist. from center (px)"),
        ("Speed", "Speed (mm/s)"),
        ("Height", "Height (mm)"),
        ("Length", "Length (mm)"),
        ("MoSeq", "Syllable ID"),
    ],
):
    """
    plot the fingerprint heatmap

    Args:
        summary (pandas.DataFrame): fingerprint dataframe
        range_dict (pandas.DataFrame): pd.DataFrame that hold min max values of the features
        preprocessor (sklearn.preprocessing object, optional): Scalar for scaling the data by session. Defaults to None.
        num_level (int, optional): the number of groupby levels. Defaults to 1.
        level_names (list, optional): list of names of the levels. Defaults to ['Group'].
        vmin (int, optional): min value the figure color map covers. Defaults to 0.
        vmax (float, optional): max value the figure color map covers. Defaults to 0.2.
        plot_columns (list, optional): columns to plot
        col_names = (list, optional): list of (column name, x label) pairs

    Raises:
        Exception: num_levels greater than the existing levels
    """
    # ensure number of groups is not over the number of available levels
    if num_level > len(summary.index.names):
        raise Exception(
            "Too many levels to unpack. num_level should be less than",
            len(summary.index.names),
        )

    name_map = dict(zip(plot_columns, col_names))

    levels = []
    level_plot = []
    level_ticks = []
    for i in range(num_level):
        level = summary.index.get_level_values(i)
        level_label = LabelEncoder().fit_transform(level)
        find_mid = (
            np.diff(
                np.r_[0, np.argwhere(np.diff(level_label)).ravel(), len(level_label)]
            )
            / 2
        ).astype("int32")
        # store level value
        levels.append(level)
        level_plot.append(level_label)
        level_ticks.append(
            np.r_[0, np.argwhere(np.diff(level_label)).ravel()] + find_mid
        )

    # col_num = number of grouping/level + column in summary
    col_num = num_level + len(plot_columns)

    # https://matplotlib.org/stable/tutorials/intermediate/gridspec.html
    fig = plt.figure(1, figsize=figsize, facecolor="white")

    gs = GridSpec(
        2,
        col_num,
        wspace=0.1,
        hspace=0.1,
        width_ratios=[1] * num_level + [8] * (col_num - num_level),
        height_ratios=[10, 0.1],
        figure=fig,
    )

    # plot the level(s)
    for i in range(num_level):
        temp_ax = fig.add_subplot(gs[0, i])
        temp_ax.set_title(level_names[i], fontsize=20)
        temp_ax.imshow(level_plot[i][:, np.newaxis], aspect="auto", cmap="Set3")
        plt.yticks(level_ticks[i], levels[i][level_ticks[i]], fontsize=20)

        temp_ax.get_xaxis().set_ticks([])

    # compile data to plot while recording vmin and vmax in the data
    plot_dict = {}
    # initialize vmin and vmax
    temp_vmin = np.inf
    temp_vmax = -np.inf

    for col in plot_columns:
        data = summary[col].to_numpy()
        # process data with preprocessor
        if preprocessor is not None:
            data = preprocessor.fit_transform(data.T).T

        if np.min(data) < temp_vmin:
            temp_vmin = np.min(data)
        if np.max(data) > temp_vmax:
            temp_vmax = np.max(data)

        plot_dict[col] = data

    if vmin is None:
        vmin = temp_vmin
    if vmax is None:
        vmax = temp_vmax

    # plot the data
    for i, col in enumerate(plot_columns):
        name = name_map[col]
        temp_ax = fig.add_subplot(gs[0, i + num_level])
        temp_ax.set_title(name[0], fontsize=20)
        data = plot_dict[col]

        # top to bottom is 0-20 for y axis
        if col == "MoSeq":
            extent = [
                summary[col].columns[0],
                summary[col].columns[-1],
                len(summary) - 1,
                0,
            ]
        else:
            extent = [
                range_dict[col].iloc[0],
                range_dict[col].iloc[1],
                len(summary) - 1,
                0,
            ]

        pc = temp_ax.imshow(
            data,
            aspect="auto",
            interpolation="none",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        temp_ax.set_xlabel(name[1], fontsize=10)
        temp_ax.set_xticks(
            np.linspace(np.ceil(extent[0]), np.floor(extent[1]), 6).astype(int)
        )
        # https://stackoverflow.com/questions/14908576/how-to-remove-frame-from-matplotlib-pyplot-figure-vs-matplotlib-figure-frame
        temp_ax.set_yticks([])
        temp_ax.axis = "tight"

    # plot colorbar
    cb = fig.add_subplot(gs[1, -1])
    plt.colorbar(pc, cax=cb, orientation="horizontal")

    # specify labels for feature scaling
    if preprocessor:
        cb.set_xlabel("Min Max")
    else:
        cb.set_xlabel("Percentage Usage")

    # saving the figure
    makedirs(save_dir, exist_ok=True)
    fig.savefig(join(save_dir, "moseq_fingerprint.pdf"))
    fig.savefig(join(save_dir, "moseq_fingerprint.png"))


def classifier_fingerprint(
    summary,
    features=["MoSeq"],
    preprocessor=None,
    classes=["group"],
    param_search=True,
    C_list=None,
    model_type="lr",
    cv="loo",
    n_splits=5,
):
    """
    run classifier using the fingerprint dataframe

    Args:
        summary (pandas.DataFrame): fingerprint dataframe
        features (list, optional): Features for the classifier. ['MoSeq'] for MoSeq syllables or a list of MoSeq scalar values. Defaults to ['MoSeq'].
        preprocessor (sklearn.preprocessing object, optional): Scalar for scaling the data by feature. Defaults to None.
        target (list, optional): labels the classifier predicts. Defaults to ['group'].
        param_search (bool, optional): run GridSearchCV to find the regularization param for classifier. Defaults to True.
        C_list (numpy.array, optional): list of C regularization paramters to search through. Defaults to None. If None, C_list will search through np.logspace(-6,3, 50)
        model_type (str, optional): name of the linear classifier. 'lr' for logistic regression or 'svc' for linearSVC. Defaults to 'lr'.
        cv (str, optional): cross validation type. 'loo' for LeaveOneOut 'skf' for StratifiedKFold. Defaults to 'loo'.
        n_splits (int, optional): number of splits for StratifiedKFold. Defaults to 5.

    Returns:
        y_true (np.array): array for true label
        y_pred (np.array): array for predicted label
        real_f1 (np.array): array for f1 score
        true_coef (np.array): array for model weights
        y_shuffle_true (np.array): array for shffuled label
        y_shuffle_pred (np.array): array for shuffled predicted label
        shuffle_f1 (np.array): array for shuffled f1 score
        shuffle_coef (np.array): array for shuffled model weights
    """
    # set up data for classifier
    X = summary[features].to_numpy()
    print(X.shape)

    summary = summary.reset_index()
    y = summary[classes].squeeze()

    # set up model type
    Model = LogisticRegression if model_type == "lr" else LinearSVC
    # scikit-learn's logistic regression chose liblinear until 0.22 and lbfgs
    # after, and the two land on visibly different coefficients: naming the
    # solver keeps the weights this reports comparable with the ones it has
    # always reported. LinearSVC has no such parameter.
    solver = {"solver": "liblinear"} if model_type == "lr" else {}
    # `multi_class="ovr"` is gone: scikit-learn removed it from
    # LogisticRegression in 1.9, which the resolver serves from Python 3.11.
    # It was never doing anything here. The parameter chose one-vs-rest over
    # multinomial, and one-vs-rest is what both of these estimators already
    # did -- liblinear has no multinomial mode, and LinearSVC is one-vs-rest
    # by default. Measured under scikit-learn 1.7.2, where both spellings
    # still work: with and without it, LogisticRegression returns identical
    # coefficients and identical predictions, and so does LinearSVC.
    clf = Model(**solver)

    if param_search:
        if C_list is None:
            C_list = np.logspace(-6, 3, 50)

        parameters = {"C": C_list}
        grid_search = GridSearchCV(
            clf,
            parameters,
            cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5),
            scoring="accuracy",
        )
        grid_search.fit(X, y)
        # set the best parameter for the classifier
        clf = clf.set_params(**grid_search.best_params_)
        print("classifier", clf)

    if isinstance(cv, str):
        if cv == "loo":
            cv = LeaveOneOut()
        elif cv == "skf":
            cv = StratifiedKFold(n_splits=n_splits)
    elif not hasattr(cv, "split"):
        # user has supplied their own cv object
        raise ValueError(
            'cv must either be a string containing "loo" or "skf" or a cross validation object'
        )

    out = defaultdict(list)

    for split, (train_ix, test_ix) in enumerate(cv.split(X, y)):
        # split data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # feature scalar
        if preprocessor is not None:
            preprocessor = clone(preprocessor)
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

        # fit model on real data
        clf = clone(clf)  # reset model to prevent data leakage
        clf.fit(X_train, y_train)
        # evaluate model
        out["y_true"].append(y_test)

        y_hat = clf.predict(X_test)
        out["y_pred"].append(y_hat)

        out["coefs"].append(clf.coef_[0])

    for i in range(100):
        y_shuffle = np.random.permutation(y)
        for split, (train_ix, test_ix) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_ix], X[test_ix]
            # shuffle y for shuffle analysis
            y_shuffle_train, y_shuffle_test = y_shuffle[train_ix], y_shuffle[test_ix]

            if preprocessor is not None:
                preprocessor = clone(preprocessor)
                X_train = preprocessor.fit_transform(X_train)
                X_test = preprocessor.transform(X_test)

            # shuffle data
            clf = clone(clf)
            clf.fit(X_train, y_shuffle_train)
            # evaluate
            out["shuff_y_true"].append(y_shuffle_test)
            y_hat = clf.predict(X_test)
            out["shuff_y_pred"].append(y_hat)
            out["shuff_coefs"].append(clf.coef_[0])
            out["shuff_split"].append(split)

    out = {k: np.concatenate(v) if "y" in k else np.array(v) for k, v in out.items()}
    out["accuracy"] = accuracy_score(out["y_true"], out["y_pred"])
    out["shuff_accuracy"] = accuracy_score(out["shuff_y_true"], out["shuff_y_pred"])

    return out


def _plot_cm(y_true, y_pred, ax, ax_labels, title):
    """plot confusion matrix

    Args:
        y_true (np.array): array for true label
        y_pred (np.array): array for predicted label
        ax (pyplot.axis): matplotlib axis for plotting the figure
        ax_labels (numpy.array): labels for the confusion matrices
        title (str): figure title

    Returns:
        im(pyplot.figure): figure for the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    cm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm, cmap="binary_r", vmin=0, vmax=1)
    plt.xticks(range(len(ax_labels)), ax_labels)
    plt.yticks(range(len(ax_labels)), ax_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Real")
    ax.set_title(title)
    return im


def plot_cm(y_true, y_pred, y_shuffle_true, y_shuffle_pred):
    """
    plot confusion matrix

    Args:
        y_true (np.array): array for true label
        y_pred (np.array): array for predicted label
        y_shuffle_true (np.array): array for shffuled label
        y_shuffle_pred (np.array): array for shuffled predicted label
    """
    fig = plt.figure(figsize=(23, 10), facecolor="white")
    gs = GridSpec(ncols=3, nrows=1, wspace=0.1, figure=fig, width_ratios=[10, 10, 0.3])
    fig_ax = fig.add_subplot(gs[0, 0])
    labels = np.unique(y_true)
    _plot_cm(
        y_true,
        y_pred,
        fig_ax,
        labels,
        f"Real Accuracy {accuracy_score(y_true, y_pred):0.2f}",
    )

    fig_ax = fig.add_subplot(gs[0, 1])
    im = _plot_cm(
        y_shuffle_true,
        y_shuffle_pred,
        fig_ax,
        labels,
        f"Shuffle Accuracy {accuracy_score(y_shuffle_true, y_shuffle_pred):0.2f}",
    )
    fig_ax.set_ylabel("")
    fig_ax.set_yticklabels([])

    # plot colorbar
    cb = fig.add_subplot(gs[0, 2])
    fig.colorbar(
        mappable=im,
        cax=cb,
        label="Fraction of labels",
    )
    fig.tight_layout()
    plt.show()
