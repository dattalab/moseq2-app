"""
Visualization and utility functions for transition matrices.
"""
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from cytoolz import sliding_window, complement
from matplotlib.lines import Line2D

def get_trans_graph_groups(model_fit):
    """
    Get the groups and their respective session uuids to use in transition graph generation.

    Args:
    model_fit (dict): trained model ARHMM containing training data UUIDs.

    Returns:
    label_group (list): list of groups for each included session
    model_uuids (list): list of corresponding UUIDs for each included session in the model
    """

    model_uuids = model_fit.get('train_list', model_fit['keys'])
    label_group = [model_fit['metadata']['groups'][k] for k in model_uuids]

    return label_group, model_uuids

def get_group_trans_mats(labels, label_group, group, max_sylls, normalize='bigram'):
    """
    Compute individual transition matrices for each given group.

    Args:
    labels (np.ndarray): list of frame labels for each included session
    label_group (list): list of groups for each included session
    group (list): list of unique groups included
    max_sylls (int): Maximum number of syllables to include
    normalize (str): indicates how to normalize the computed transition matrices.

    Returns:
    trans_mats (list of 2D np.ndarrays): list of transition matrices for each given group.
    usages (list of lists): list of corresponding usage statistics for each group.
    """
    # Importing within function to avoid import loops
    from moseq2_viz.model.util import get_syllable_statistics

    trans_mats = []
    usages = []

    # Computing transition matrices for each given group
    for plt_group in group:
        # Get sessions to include in trans_mat
        use_labels = [lbl for lbl, grp in zip(labels, label_group) if grp == plt_group]
        trans_mats.append(get_transition_matrix(use_labels,
                                                normalize=normalize,
                                                combine=True,
                                                max_syllable=max_sylls))

        # Getting usage information for node scaling
        usages.append(get_syllable_statistics(use_labels, max_syllable=max_sylls)[0])
    return trans_mats, usages

def compute_and_graph_grouped_TMs(config_data, labels, label_group, group):
    """
    compute a transition matrix for each given group.

    Args:
    config_data (dict): configuration dictionary containing graphing parameters
    labels (list): list of 1D numpy arrays containing syllable labels per frame for every included session
    label_group (list): list of corresponding group names to plot transition aggregated transition plots
    group (list): unique list of groups to plot.

    Returns:
    plt (pyplot.Figure): open transition graph figure to save
    """

    trans_mats, usages = get_group_trans_mats(labels, label_group, sorted(group), config_data['max_syllable'], config_data['normalize'])

    # Option to not scale node sizes proportional to the syllable usage.
    if not config_data['scale_node_by_usage']:
        usages = None

    print('Creating plot...')
    plt, _, _ = graph_transition_matrix(trans_mats,
                                        **config_data,
                                        usages=usages,
                                        groups=sorted(group),
                                        headless=True)
    # manually add legend for the difference graph
    if len(group) >1:
        legend_elements = [Line2D([0], [0], color='r', lw=2, label= f'Up-regulated transistion'),
                           Line2D([0], [0], color='b', lw=2, label= f'Down-regulated transistion'),
                           Line2D([0], [0], marker='o', color='w', label=f'Up-regulated usage',markerfacecolor='w', markeredgecolor = 'r', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label=f'Down-regulated usage',markerfacecolor='w', markeredgecolor = 'b', markersize=10)]
        plt.legend(handles = legend_elements, loc='upper left')

    return plt

def get_transitions(label_sequence):
    """
    Computes syllable transitions.

    Args:
    label_sequence (tuple): a tuple of syllable transitions and their indices

    Returns:
    transitions (np.array): filtered label sequence containing only the syllable changes
    locs (np.array): list of all the indices where the syllable label changes
    """

    arr = deepcopy(label_sequence)

    # get syllable transition locations
    locs = np.where(arr[1:] != arr[:-1])[0] + 1
    transitions = arr[locs]

    return transitions, locs


def normalize_transition_matrix(init_matrix, normalize):
    """
    Normalize a transition matrix by given criteria.

    Args:
    init_matrix (np.array): transition matrix to normalize.
    normalize (str): normalization criteria; ['bigram', 'rows', 'columns', or None]

    Returns:
    init_matrix (np.array): normalized transition matrix
    """
    if normalize is None or normalize not in ('bigram', 'rows', 'columns'):
        return init_matrix

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        if normalize == 'bigram':
            init_matrix /= init_matrix.sum()
        elif normalize == 'rows':
            init_matrix /= init_matrix.sum(axis=1, keepdims=True)
        elif normalize == 'columns':
            init_matrix /= init_matrix.sum(axis=0, keepdims=True)

    return init_matrix


def n_gram_transition_matrix(labels, n=2, max_label=99):
    """
    Compute the transition count for a fixed syllable sequence length 'n'.

    Args:
    labels (list of np.array of ints): syllable slabels loaded from a model fit.
    n (int): length of transition chain to compute transition probability for.
    max_label (int): max number of syllables to scan for in transition matrix.

    Returns:
    trans_mat (np.ndarray): array of n-transition counts for given max_label.
    """

    trans_mat = np.zeros((max_label, ) * n, dtype='float')
    for loc in sliding_window(n, labels):
        if any(l >= max_label for l in loc):
            continue
        trans_mat[loc] += 1
    return trans_mat


# per https://gist.github.com/tg12/d7efa579ceee4afbeaec97eb442a6b72
def get_transition_matrix(labels, max_syllable=100, normalize='bigram',
                          smoothing=0.0, combine=False, disable_output=False) -> list:
    """
    Compute the transition matrix from a set of model labels.

    Args:
    labels (list of numpy.array): labels loaded from a model fit
    max_syllable (int): the index of the maximum number of syllables to include
    normalize (str): how to normalize transition matrix, 'bigram' or 'rows' or 'columns'
    smoothing (float): constant to add to transition_matrix pre-normalization to smooth counts
    combine (bool): flag for computing a separate transition matrix for each element (False) or combine across all arrays in the list (True)
    disable_output (bool): flag to display a TQDM progress bar for transition matrix computation process.

    Returns:
    transition_matrix (list or np.ndarray): list of 2d np.arrays that represent the transitions
    """
    if not isinstance(labels[0], (list, np.ndarray, pd.Series)):
        labels = [labels]

    # Compute a singular transition matrix
    if combine:
        init_matrix = []

        for v in tqdm(labels, disable=disable_output, desc='Computing bigram transition probabilities'):
            # Get syllable transitions
            transitions = get_transitions(v)[0]

            trans_mat = n_gram_transition_matrix(transitions, n=2, max_label=max_syllable)
            init_matrix.append(trans_mat)

        init_matrix = np.sum(init_matrix, axis=0) + smoothing
        all_mats = normalize_transition_matrix(init_matrix, normalize)
    else:
        # Compute a transition matrix for each session label list
        all_mats = []
        for v in tqdm(labels, disable=disable_output, desc='Computing bigram transition probabilities'):
            # Get syllable transitions
            transitions = get_transitions(v)[0]

            trans_mat = n_gram_transition_matrix(transitions, n=2, max_label=max_syllable) + smoothing

            # Normalize matrix
            init_matrix = normalize_transition_matrix(trans_mat, normalize)
            all_mats.append(init_matrix)

    return all_mats


def convert_ebunch_to_graph(ebunch):
    """
    Convert transition matrices to transition DAGs.

    Args:
    ebunch (list of tuples): syllable transition data

    Returns:
    g (networkx.DiGraph): DAG object to graph
    """

    g = nx.DiGraph()
    g.add_weighted_edges_from(ebunch)

    return g


def convert_transition_matrix_to_ebunch(weights, transition_matrix,
                                        usages=None, usage_threshold=-.1,
                                        speeds=None, speed_threshold=0,
                                        edge_threshold=-.1, indices=None,
                                        keep_orphans=False, max_syllable=None):
    """
    Compute thresholded syllable transition data by usages and transition probabilities.

    Args:
    weights (np.ndarray): syllable transition edge weights
    transition_matrix (np.ndarray): syllable transition matrix
    usages (list): list of syllable usages
    usage_threshold (float): threshold syllable usage to include a syllable in list of orphans
    speeds (np.array): list of syllable speeds
    speed_threshold (int): threshold value for syllable speeds to include
    edge_threshold (float): threshold transition probability to consider an edge part of the graph.
    indices (list): indices of syllable bigrams to plot
    keep_orphans (bool): indicate whether to graph orphan syllables
    max_syllable (int): the index of the maximum number of syllables to include

    Returns:
    ebunch (list): syllable transition data.
    orphans (list): syllables with no edges.
    """
    # TODO: figure out if I ever need the transition_matrix variable
    # Cap the number of included syllable states
    if max_syllable is not None:
        weights = weights[:max_syllable, :max_syllable]
        transition_matrix = transition_matrix[:max_syllable, :max_syllable]

    def _filter_ebunch(arg):
        _, _, w = arg
        w = abs(w)
        if isinstance(edge_threshold, (list, tuple)):
            return (w > edge_threshold[0]) and (w < edge_threshold[1])
        return w > edge_threshold

    orphans = []

    # u, v, w where w is the weights
    ebunch = list(filter(_filter_ebunch, ((*inds, w) for inds, w in np.ndenumerate(weights))))
    if keep_orphans:
        orphans = list(filter(complement(_filter_ebunch), ((*inds, w) for inds, w in np.ndenumerate(weights))))

    if indices is not None:
        if keep_orphans:
            orphans += list(filter(lambda e: e[:-1] not in indices, ebunch))
        ebunch = list(filter(lambda e: e[:-1] in indices, ebunch))

    def _filter_by_stat(arg, stat, stat_threshold):
        from math import ceil, floor
        _in, _out, _ = arg
        _in = stat[_in]
        _out = stat[_out]
        if isinstance(stat_threshold, (list, tuple)):
            # round down the lower bound and round up the upper bound to 1000th decimal
            return ((_in > floor(stat_threshold[0]*1000)/1000. and _in <= ceil(stat_threshold[1]*1000)/1000.)
                    and (_out >= floor(stat_threshold[0]*1000)/1000. and _out <= ceil(stat_threshold[1]*1000)/1000.))
        return _in > stat_threshold and _out > stat_threshold

    if usages is not None:
        ebunch = list(filter(lambda e: _filter_by_stat(e, usages, usage_threshold), ebunch))
    if speeds is not None:
        ebunch = list(filter(lambda e: _filter_by_stat(e, speeds, speed_threshold), ebunch))

    return ebunch, [o[:-1] for o in orphans]


def make_difference_graphs(trans_mats, usages, group, group_names, usage_kwargs,
                           widths, pos, node_edge_colors, ax=None, node_sizes=[], indices=None,
                           difference_threshold=0.0005, difference_edge_width_scale=500, font_size=12,
                           usage_scale=5e4, difference_graphs=[], scalars=None, arrows=False, speed_kwargs={}):
    """
    compute transition graph differences bewtween two groups.

    Args:
    trans_mats (np.ndarray): syllable transition matrix.
    usages (list): list of syllable usage probabilities.
    group (list): list groups to graph transition graphs for.
    group_names (list): list groups names to display with transition graphs.
    usage_kwargs (dict): kwargs for graph threshold settings using usage. Keys can be 'usages', and 'usage_threshold'
    widths (list): list of edge widths for each created single-group graph.
    pos (nx.Layout): nx.Layout type object holding position coordinates for the nodes.
    node_edge_colors (list): node edge colors (of type str).
    ax (np.ndarray matplotlib.pyplot.Axis): Optional axes to plot graphs in
    node_sizes (list): node size scaling factor (of type int)
    indices (list): list of in->out syllable indices to keep in graph
    difference_threshold (float): threshold to consider 2 graph elements different.
    difference_edge_width_scale (int): scaling factor for edge widths in difference transition graphs.
    font_size (int): indicates the size of the numbers drawn on the transition graph nodes.
    usage_scale (float): syllable usage scaling factor.
    difference_graphs (list): list of created difference transition graphs.
    scalars (dict): dict of syllable scalar data per transition graph
    arrows (bool): indicates whether to display arrows between node transitions
    speed_kwargs (dict): kwargs for graph threshold settings using usage. Keys can be 'speeds', and 'speed_threshold'

    Returns:
    usages (list): list of syllable usage probabilities including usages differences across groups
    group_names (list): list groups names to display with transition graphs including difference graphs.
    difference_graphs (list): list of computed difference graphs
    widths (list): list of edge widths for each created graph appended with difference weights
    node_sizes (list): lists of node sizes corresponding to each graph including difference graph node sizes
    node_edge_colors (list): lists of node colors corresponding to each graph including difference graph node sizes
    """

    for i, tm in enumerate(trans_mats):
        for j, tm2 in enumerate(trans_mats[i + 1:]):
            if len(tm2) == 0:
                continue
            # get graph difference
            df = tm2 - tm

            if isinstance(scalars, dict):
                for key, _scalar_list in scalars.items():
                    if len(_scalar_list) > 0:
                        df_scalar = {k: _scalar_list[j + i + 1][k] - _scalar_list[i][k] for k in range(len(df))}
                        scalars[key].append(df_scalar)

            # make difference graph
            ebunch, _ = convert_transition_matrix_to_ebunch(df, df, edge_threshold=difference_threshold, indices=indices,
                                                            keep_orphans=False, max_syllable=tm.shape[0], **speed_kwargs, **usage_kwargs)
            # get graph from ebunch
            graph = convert_ebunch_to_graph(ebunch)

            difference_graphs.append(graph)

            # get edge widths
            weight = [np.abs(graph[u][v]['weight']) * difference_edge_width_scale
                      for u, v in graph.edges()]

            edge_colors = ['b' if (graph[u][v]['weight'] * difference_edge_width_scale > 0) else 'r'
                           for u, v in graph.edges()]
            widths.append(weight)

            # Handle node size and coloring
            if usages is not None:
                # get usage difference
                df_usage = {k: usages[j + i + 1][k] - usages[i][k] for k in range(len(df))}
                usages.append(df_usage)

                # get node sizes and colors based on usage differences
                node_size = np.abs(list(df_usage.values()))[:len(graph.nodes)] * usage_scale
                node_edge_color = ['b' if x > 0 else 'r' for x in df_usage.values()][:len(graph.nodes)]

                node_sizes.append(node_size)
                node_edge_colors.append(node_edge_color)
            else:
                # Set default node display values
                node_size = 400
                node_edge_color = 'r'
                node_sizes.append(node_size)
                node_edge_colors.append(node_edge_color)

            # get difference graph name
            curr_name = f'{group[i + j + 1]} - {group[i]}'
            group_names.append(curr_name)

            if ax is not None:
                draw_graph(graph, weight, pos, node_color='w', node_size=node_size,
                           node_edge_colors=node_edge_color, arrows=arrows, edge_colors=edge_colors,
                           font_size=font_size, ax=ax[i][i + j + 1], title=curr_name)

    return usages, group_names, difference_graphs, widths, node_sizes, node_edge_colors, scalars


def make_transition_graphs(trans_mats, usages, group, group_names, usage_kwargs,
                           pos, orphans, edge_threshold=.0025,
                           difference_threshold=.0005, orphan_weight=0,
                           ax=None, edge_width_scale=100, usage_scale=1e5,
                           difference_edge_width_scale=500, speed_kwargs={},
                           indices=None, font_size=12, scalars=None, arrows=False):
    """

    create transition matrices for all included groups, as well as their difference graphs.

    Args:
    trans_mats (np.ndarray): syllable transition matrix.
    usages (list): list of syllable usage probabilities.
    group (list): list groups to graph transition graphs for.
    group_names (list): list groups names to display with transition graphs.
    usage_kwargs (dict): kwargs for graph threshold settings using usage. Keys can be 'usages', and 'usage_threshold'
    pos (nx.Layout): nx.Layout type object holding position coordinates for the nodes.
    orphans (list): list of nodes with no edges.
    edge_threshold (float): threshold to include edge in graph.
    difference_threshold (float): threshold to consider 2 graph elements different.
    orphan_weight (int): scaling factor to plot orphan node sizes.
    ax (np.ndarray matplotlib.pyplot Axis): Optional axes to plot graphs in
    edge_width_scale (int): edge line width scaling factor.
    usage_scale (float): syllable usage scaling factor.
    difference_edge_width_scale (int): scaling factor for edge widths in difference transition graphs.
    speed_kwargs (dict): kwargs for graph threshold settings using usage. Keys can be 'speeds', and 'speeds_threshold'
    indices (list): list of in->out syllable indices to keep in graph
    font_size (int): indicates the size of the numbers drawn on the transition graph nodes.
    scalars (dict): dict of syllable scalar data per transition graph
    arrows (bool): indicates whether to display arrows between node transitions

    Returns:
    usages (list): list of syllable usage probabilities including possible appended difference usages.
    group_names (list): list groups names to display with transition graphs including difference graphs.
    widths (list): list of edge widths for each created graph appended with difference weights.
    node_sizes (2D list): lists of node sizes corresponding to each graph.
    node_edge_colors (2D list): lists of node colors corresponding to each graph including difference graph node sizes.
    graphs (list of nx.DiGraph): list of all group and difference transition graphs.
    """

    graphs = []
    widths = []
    node_sizes = []
    node_edge_colors = []

    for i, tm in enumerate(trans_mats):
        # make graph from transition matrix
        ebunch, orphans = convert_transition_matrix_to_ebunch(tm, tm, edge_threshold=edge_threshold, indices=indices,
                                                              keep_orphans=True, max_syllable=tm.shape[0], **usage_kwargs, **speed_kwargs)

        # get graph from ebunch
        graph = convert_ebunch_to_graph(ebunch)

        # get edge widths
        assert all(edge not in orphans for edge in graph.edges()), 'graph contains orphans'
        width = [tm[u][v] * edge_width_scale if (u, v) not in orphans else orphan_weight
                 for u, v in graph.edges()]
        widths.append(width)

        # set node sizes according to usages
        if usages is not None:
            node_size = [usages[i][k] * usage_scale for k in pos.keys()][:len(graph.nodes)]
            node_sizes.append(node_size)
        else:
            node_size = 400
            node_sizes.append(node_size)

        # Draw network to matplotlib figure
        if ax is not None:
            draw_graph(graph, width, pos, node_color='w',
                       node_size=node_size, node_edge_colors='r', arrows=arrows,
                       font_size=font_size, ax=ax[i][i], title=group_names[i])

        node_edge_colors.append('r')
        graphs.append(graph)

    # get group difference graphs
    if len(group) > 1:
        usages, group_names, graphs, widths, node_sizes, node_edge_colors, scalars = \
            make_difference_graphs(trans_mats, usages, group, group_names,
            widths=widths, pos=pos,
            ax=ax,
            node_edge_colors=node_edge_colors,
            node_sizes=node_sizes,
            difference_threshold=difference_threshold,
            difference_edge_width_scale=difference_edge_width_scale,
            difference_graphs=graphs,
            scalars=scalars,
            arrows=arrows,
            indices=indices, 
            font_size=font_size,
            usage_kwargs=usage_kwargs,
            speed_kwargs=speed_kwargs)

    return usages, group_names, widths, node_sizes, node_edge_colors, graphs, scalars


def get_pos(graph_anchor, layout, nnodes):
    """
    Get node positions in the graph based on the graph anchor and a user selected layout.

    Args:
    graph_anchor (nx.Digraph): graph to get node layout for
    layout (str): networkx layout type
    nnodes (int): number of nodes in the graph

    Returns:
    pos (nx layout): computed node position layout
    """

    if isinstance(layout, str) and layout.lower() == 'spring':
        k = 1.5 / np.sqrt(nnodes)
        pos = nx.spring_layout(sorted(graph_anchor.nodes()), k=k, seed=0)
    elif isinstance(layout, str) and layout.lower() == 'circular':
        pos = nx.circular_layout(sorted(graph_anchor.nodes()))
    elif isinstance(layout, str) and layout.lower() == 'spectral':
        pos = nx.spectral_layout(sorted(graph_anchor.nodes()))
    elif isinstance(layout, (dict, OrderedDict)):
        # user passed pos directly
        pos = layout
    else:
        raise RuntimeError('Did not understand layout type')

    return pos


def draw_graph(graph, width, pos, node_color,
               node_size, node_edge_colors, ax, arrows=False,
               font_size=12, edge_colors='k', title=None):
    """
    Draw transition graph to existing matplotlib axes.

    Args:
    graph (nx.DiGraph): list of created nx.DiGraphs converted from transition matrices
    width (list): list of edge widths corresponding to each graph's edges.
    pos (nx.Layout): nx.Layout type object holding position coordinates for the nodes.
    node_color (list): list of node colors for each graph.
    node_sizes (int or list): list of node sizes for each graph.
    node_edge_colors (list): list of node edge colors for each graph.
    ax (mpl.pyplot.axis): axis to draw graphs on.
    arrows (bool): whether to draw arrow edges
    font_size (int): Node label font size
    edge_colors (str): color of the transition edges drawn in the graph.
    title (str): title/group name of the transition graph.

    Returns:
    """

    # Draw nodes and edges on matplotlib figure
    # reconstruct node_size list to match the sorted node labels
    if isinstance(node_size, list):
        node_size_dict = {}
        for i,v in enumerate(node_size):
            node_size_dict[i]=v
        node_size = [node_size_dict.get(n) if node_size_dict.get(n) else 0 for n in graph.nodes]
    
    nx.draw(graph, pos=pos, with_labels=True, font_size=font_size, alpha=1,
            width=width, edgecolors=node_edge_colors, ax=ax, cmap='jet',
            node_size=node_size, node_color=node_color, arrows=arrows,
            edge_color=edge_colors, linewidths=1.5)

    # Set titles
    ax.set_title(title)


def graph_transition_matrix(trans_mats, usages=None, groups=None,
                            edge_threshold=.0025, anchor=0, usage_threshold=0,
                            layout='spring', edge_width_scale=100, fig=None, ax=None,
                            width_per_group=8, headless=False, difference_threshold=.0005,
                            weights=None, usage_scale=1e4, keep_orphans=False,
                            max_syllable=None, orphan_weight=0, arrows=False, font_size=12,
                            difference_edge_width_scale=500, **kwargs):
    """
    Create transition graph plot given a transition matrix and some metadata.

    Args:
    trans_mats (np.ndarray): syllable transition matrix
    usages (list): list of syllable usage probabilities
    groups (list): list groups to graph transition graphs for.
    edge_threshold (float): threshold to include edge in graph
    anchor (int): syllable index as the base syllable
    usage_threshold (int): threshold to include syllable usages
    layout (str): layout format
    edge_width_scale (int): edge line width scaling factor
    fig (pyplot.figure): figure to plot to
    ax (pyplot.Axes): axes object
    width_per_group (int): graph width scaling factor per group
    headless (bool): exclude first node.
    difference_threshold (float): threshold to consider 2 graph elements different
    weights (list): list of edge weights
    usage_scale (float): syllable usage scaling factor
    keep_orphans (bool): plot orphans.
    max_syllable (int): the index of the maximum number of syllables to include
    orphan_weight (int): scaling factor to plot orphan node sizes
    arrows (bool): indicate whether to plot arrows as transitions.
    difference_edge_width_scale (float): difference graph edge line width scaling factor
    kwargs (dict): extra keyword arguments

    Returns:
    fig (pyplot.figure): figure containing transition graphs.
    ax (pyplot.axis): figure axis object.
    pos (dict): dict figure information.
    """
    from moseq2_viz.model.util import normalize_usages

    if headless:
        plt.switch_backend('agg')

    if usages is not None:
        usages = deepcopy(usages)

    # Set weights to transition probabilities
    if weights is None:
        weights = trans_mats

    # Ensure transition matrices are N-dim numpy arrays
    assert isinstance(trans_mats, (np.ndarray, list)), "Transition matrix must be a numpy array or list of arrays"

    # Ensure transition matrices are 2D
    if isinstance(trans_mats, np.ndarray):
        assert trans_mats.ndim == 2, 'Transition matrix needs to be 2 dimensional'
        trans_mats = [trans_mats]

    # Get shared node anchors based on usages
    ngraphs = len(trans_mats)
    if usages is not None:
        usages = [normalize_usages(u) for u in usages]
    anchor = anchor if anchor < len(trans_mats) else 0
    if usages is not None:
        usages_anchor = usages[anchor]
    else:
        usages_anchor = None

    # Create transition graph metadata from transition matrix
    ebunch_anchor, orphans = convert_transition_matrix_to_ebunch(
        weights[anchor], trans_mats[anchor], edge_threshold=edge_threshold,
        keep_orphans=keep_orphans, usages=usages_anchor,
        usage_threshold=usage_threshold, max_syllable=max_syllable)

    # Create transition graph object
    graph_anchor = convert_ebunch_to_graph(ebunch_anchor)
    nnodes = len(graph_anchor.nodes())

    # Get node position layout
    pos = get_pos(graph_anchor, layout, nnodes)

    # Create figure to plot
    if fig is None or ax is None:
        fig, ax = plt.subplots(ngraphs, ngraphs,
                               figsize=(ngraphs*width_per_group,
                                        ngraphs*width_per_group))

    # Format axis
    if ngraphs == 1:
        ax = [[ax]]

    if isinstance(groups, str):
        groups = [groups]

    # Get group name list to append difference graph names
    group_names = deepcopy(groups)
    
    # Make graphs and difference graphs
    _ = make_transition_graphs(trans_mats, usages, groups, group_names,
        pos=pos, orphans=orphans, indices=[e[:-1] for e in ebunch_anchor],
        usage_kwargs={'usages': usages_anchor}, edge_threshold=edge_threshold,
        difference_edge_width_scale=difference_edge_width_scale,
        difference_threshold=difference_threshold, orphan_weight=orphan_weight,
        ax=ax, edge_width_scale=edge_width_scale, usage_scale=usage_scale,
        arrows=arrows, font_size=font_size)

    for a in np.array(ax).flat:
        a.axis('off')

    return fig, ax, pos