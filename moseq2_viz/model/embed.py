"""
Functions to run PCA and LDA on syllable usages and scalars
"""

import numpy as np
import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from moseq2_viz.model.util import get_Xy_values
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def run_2d_embedding(
    mean_df,
    stat="usage",
    output_file="2d_embedding.pdf",
    embedding="PCA",
    n_components=2,
    plot_all_subjects=True,
):
    """
    Compute a 2D embedding (PCA or LDA) of the mean syllable statistic of choice. The function will output a figure of the 2D representation of the embedding.

    Args:
    mean_df (pd DataFrame): Dataframe of the mean syllable statistics for all sessions
    stat (str): name of statistic (column) in mean_df to embed.
    output_file (str): path to saved outputted figure
    embedding (str): type of embedding to run. Either ['lda', 'pca']
    n_components (int): Number of components to compute.
    plot_all_subjects (bool): indicates whether to plot individual subject embeddings along with their respective group means.

    Returns:
    fig (matplotlib.figure): figure containing plotted 2d embedding.
    ax (matplonlib.axes): axes instance for plotted figure.
    """

    if embedding.lower() == "lda":
        embedder = LDA(
            solver="eigen",
            shrinkage="auto",
            n_components=n_components,
            store_covariance=True,
        )
    elif embedding.lower() == "pca":
        embedder = PCA(n_components=n_components)
    else:
        print('Unsupported input. Only input embedding="lda" or "pca".')
        return None, None

    syllable_df = mean_df.groupby(
        ["syllable", "uuid", "group"], as_index=False
    ).mean(numeric_only=True)

    unique_groups = sorted(syllable_df.group.unique())

    X, y, mapping, rev_mapping = get_Xy_values(syllable_df, unique_groups, stat=stat)

    L = embedder.fit_transform(X, y)
    if L.shape[1] <= 1:
        print("Not enough dimensions to plot, try a different embedding method.")
        return None, None

    fig, ax = plot_embedding(
        L,
        y,
        mapping,
        rev_mapping,
        output_file=output_file,
        embedding=embedding,
        plot_all_subjects=plot_all_subjects,
    )

    return fig, ax


def run_2d_scalar_embedding(
    scalar_df,
    output_file="2d_scalar_embedding.pdf",
    embedding="PCA",
    n_components=2,
    plot_all_subjects=True,
):
    """
    Compute a 2D embedding (PCA or LDA) of the mean measured scalar values for all groups. The function will output a figure of the 2D representation of the embedding.

    Args:
    scalar_df (pd DataFrame): Dataframe of the frame-by-frame scalar measurements for all sessions
    output_file (str): path to saved outputted figure
    embedding (str): type of embedding to run. Either ['lda', 'pca']
    n_components (int): Number of components to compute.
    plot_all_subjects (bool): indicates whether to plot individual subject embeddings along with their respective group means.

    Returns:
    fig (matplotlib figure): figure containing plotted 2d embedding.
    ax (matplonlib axes): axes instance for plotted figure.
    """

    # Initialize embedding method
    if embedding.lower() == "lda":
        embedder = LDA(
            solver="eigen",
            shrinkage="auto",
            n_components=n_components,
            store_covariance=True,
        )
    elif embedding.lower() == "pca":
        embedder = PCA(n_components=n_components)
    else:
        print('Unsupported input. Only input embedding="lda" or "pca".')
        return None, None

    # Get group mean scalar values
    scalar_mean_df = scalar_df.groupby(
        ["uuid", "group"], as_index=False
    ).mean(numeric_only=True)

    # Get unique list of groups
    unique_groups = sorted(scalar_mean_df.group.unique())

    # Get full set of scalar features to embed
    scalar_cols = (scalar_df.dtypes == "float32") | (scalar_df.dtypes == "float")
    feature_cols = list(scalar_cols[scalar_cols].index)

    # Exclude non-scalar columns
    to_exclude = ["timestamps", "frame index"]
    feats = [f for f in feature_cols if f not in to_exclude]

    # Format data to 2D array and target variable formats (X, y)
    X, y, mapping, reverse_mapping = get_Xy_values(
        scalar_mean_df, unique_groups, stat=feats
    )

    # Embed the data
    L = embedder.fit_transform(X, y)

    if L.shape[1] <= 1:
        print("Not enough dimensions to plot, try a different embedding method.")
        return None, None

    # Plot the 2D embedding
    fig, ax = plot_embedding(
        L,
        y,
        mapping,
        reverse_mapping,
        output_file=output_file,
        embedding=embedding,
        plot_all_subjects=plot_all_subjects,
    )

    return fig, ax


def plot_embedding(
    L,
    y,
    mapping,
    rev_mapping,
    output_file="embedding.pdf",
    embedding="PCA",
    x_dim=0,
    y_dim=1,
    symbols="o*v^s",
    plot_all_subjects=True,
):
    """
    Plot 2D embedding plot.

    Args:
    L (2D np.array): the embedding representations of the mean syllable statistic to plot.
    y (1D list): list of group names corresponding to each row in L.
    mapping (dict): dictionary conataining mappings from group string to integer for later embedding.
    rev_mapping (dict): inverse mapping dict to retrieve the group names given their mapped integer value.
    output_file (str): path to saved outputted figure
    embedding (str): type of embedding to run. Either ['lda', 'pca'].
    x_dim (int): component number to graph on x-axis
    y_dim (int): component number to graph on y-axis
    symbols (str): symbols to use to draw different groups.
    plot_all_subjects (bool): boolean flag that indicates whether to plot individual subject embeddings along with their respective group means.

    Returns:
    fig (matplotlib.figure): figure containing plotted 2d embedding.
    ax (matplonlib.axes): axes instance for plotted figure.
    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor="w")

    # Create color and symbol combination
    colors = sns.color_palette(
        n_colors=max(1, int(((len(y) + 1) / (len(symbols) - 1))))
    )
    symbols, colors = zip(*list(product(symbols, colors)))

    # Set figure axes
    ax.set_xlabel(f"{embedding} 1")
    ax.set_ylabel(f"{embedding} 2")
    ax.set_xticks([])
    ax.set_yticks([])

    # plot each group's embedding
    for i in range(len(mapping)):
        # get embedding indices
        # A bare boolean mask. numpy read the list wrapper as a sequence
        # of indices until it started rejecting it in 1.23; the mask is
        # what was always meant.
        idx = y == i

        # plotting individual subject data points
        if plot_all_subjects:
            plt.plot(
                L[idx][:, x_dim],
                L[idx][:, y_dim],
                symbols[i],
                color=colors[i],
                alpha=0.3,
                markersize=10,
            )

        # plot mean embedding with corresponding symbol and color
        mu = np.nanmean(L[idx], axis=0)
        plt.plot(mu[x_dim], mu[y_dim], symbols[i], color=colors[i], markersize=10)

        # plot text group name indicator at computed mean
        plt.text(
            mu[x_dim],
            mu[y_dim],
            rev_mapping[i] + " (%s)" % symbols[i],
            fontsize=18,
            color=colors[i],
            horizontalalignment="center",
            verticalalignment="center",
        )

    sns.despine()
    fig.savefig(output_file, bbox_inches="tight", format="pdf")

    return fig, ax
