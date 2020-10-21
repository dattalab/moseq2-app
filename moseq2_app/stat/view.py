import random
import warnings
import itertools
import numpy as np
import networkx as nx
from collections import deque
from bokeh.layouts import column
from bokeh.layouts import gridplot
from bokeh.palettes import Spectral4
from bokeh.transform import linear_cmap
from bokeh.models.tickers import FixedTicker
from bokeh.palettes import Category10_10 as palette
from bokeh.plotting import figure, show, from_networkx
from bokeh.models import (ColumnDataSource, LabelSet, BoxSelectTool, Circle, ColorBar,
                          Legend, LegendItem, HoverTool, MultiLine, NodesAndLinkedEdges, TapTool, ColorPicker)

def graph_dendrogram(obj):
    '''
    Graphs the distance sorted dendrogram representing syllable neighborhoods. Distance sorting
    is computed by processing the respective syllable AR matrices.

    Parameters
    ----------
    obj (InteractiveSyllableStats object): Syllable Stats object containing syllable stat information.

    Returns
    -------
    '''

    ## Cladogram figure
    cladogram = figure(title='Distance Sorted Syllable Dendrogram',
                       width=850,
                       height=500,
                       output_backend="svg")

    # Show syllable info on hover
    cladogram.add_tools(HoverTool(tooltips=[('label', '@labels')]), TapTool(), BoxSelectTool())

    # Get distance sorted label ordering
    labels = list(map(int, obj.results['ivl']))
    sources = []

    # Each (icoord, dcoord) pair represents a single branch in the dendrogram
    for ii, (i, d) in enumerate(zip(obj.icoord, obj.dcoord)):
        d = list(map(lambda x: x, d))
        tmp = [[x, y] for x, y in zip(i, d)]
        lbls = []

        # Get labels
        for t in tmp:
            if t[1] == 0:
                lbls.append(labels[int(t[0])])
            else:
                lbls.append('')

        # Set coordinate DataSource
        source = ColumnDataSource(dict(x=i, y=d, labels=lbls))
        sources.append(source)

        # Draw glyphs
        cladogram.line(x='x', y='y', source=source)

    # Set x-axis ticks
    cladogram.xaxis.ticker = FixedTicker(ticks=labels)
    cladogram.xaxis.major_label_overrides = {i: str(l) for i, l in enumerate(labels)}

    return cladogram

def clamp(val, minimum=0, maximum=255):
    '''
    Caps the given R/G/B value to set min and max values

    https://thadeusb.com/weblog/2010/10/10/python_scale_hex_color/

    Parameters
    ----------
    val (float): value for given color tuple member
    minimum (int): min thresholding value
    maximum (int): max thresholding value

    Returns
    -------
    val (int): thresholded color tuple member
    '''

    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val

def colorscale(hexstr, scalefactor):
    """
    Scales a hex string by ``scalefactor``. Returns scaled hex string.

    To darken the color, use a float value between 0 and 1.
    To brighten the color, use a float value greater than 1.

    >>> colorscale("#DF3C3C", .5)
    #6F1E1E
    >>> colorscale("#52D24F", 1.6)
    #83FF7E
    >>> colorscale("#4F75D2", 1)
    #4F75D2
    """

    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = int(clamp(r * scalefactor))
    g = int(clamp(g * scalefactor))
    b = int(clamp(b * scalefactor))

    return "#%02x%02x%02x" % (r, g, b)

def draw_stats(fig, df, groups, colors, sorting, groupby, stat, errorbar, line_dash='solid'):
    '''
    Helper function to bokeh_plotting that iterates through the given DataFrame and plots the
    data grouped by some user defined column ('group', 'SessionName'), with the errorbars of their
    choice.

    Parameters
    ----------
    fig (bokeh figure): Figure to draw line plot glyphs on
    df (pd.DataFrame): DataFrame containing all relevant data to plot
    groups (list of str): List of group names to iterate by
    colors (list of str): List of group-corresponding colors to iterate by
    sorting (list of int): Pre-selected syllable index order
    groupby (str): string that indicates which DataFrame column is being grouped.
    stat (str): String that indicates the statistic that is being plotted.
    errorbar (str): String that indicates the type of error bars to be plotted.

    Returns
    -------
    pickers (list of ColorPickers): List of interactive color picker widgets to update the graph colors
    '''

    pickers = []
    for i, color in zip(range(len(groups)), colors):
        # Get resorted mean syllable data
        aux_df = df[df[groupby] == groups[i]].groupby('syllable', as_index=False).mean().reindex(sorting)

        # Get SEM values
        if errorbar == 'SEM':
            sem = df[df[groupby] == groups[i]].groupby('syllable')[[stat]].sem().reindex(sorting)
            aux_sem = df[df[groupby] == groups[i]].groupby('syllable', as_index=False).sem().reindex(sorting)
        else:
            sem = df[df[groupby] == groups[i]].groupby('syllable')[[stat]].std().reindex(sorting)
            aux_sem = df[df[groupby] == groups[i]].groupby('syllable', as_index=False).std().reindex(sorting)

        miny = aux_df[stat] - sem[stat]
        maxy = aux_df[stat] + sem[stat]

        errs_x = [(i, i) for i in range(len(aux_df.index))]
        errs_y = [(min_y, max_y) for min_y, max_y in zip(miny, maxy)]

        # Get Labeled Syllable Information
        desc_data = df.groupby(['syllable', 'label', 'desc', 'crowd_movie_path'], as_index=False).mean()[
            ['syllable', 'label', 'desc', 'crowd_movie_path']].reindex(sorting)

        # Pack data into numpy arrays
        labels = desc_data['label'].to_numpy()
        desc = desc_data['desc'].to_numpy()
        cm_paths = desc_data['crowd_movie_path'].to_numpy()

        # stat data source
        source = ColumnDataSource(data=dict(
            x=range(len(aux_df.index)),
            y=aux_df[stat].to_numpy(),
            usage=aux_df['usage'].to_numpy(),
            speed=aux_df['speed'].to_numpy(),
            speed_2d=aux_df['velocity_2d_mm'].to_numpy(),
            speed_3d=aux_df['velocity_3d_mm'].to_numpy(),
            height=aux_df['height_ave_mm'].to_numpy(),
            dist_to_center=aux_df['dist_to_center'].to_numpy(),
            sem=aux_sem[stat].to_numpy(),
            number=sem.index,
            label=labels,
            desc=desc,
            movies=cm_paths,
        ))

        # SEM data source
        err_source = ColumnDataSource(data=dict(
            x=errs_x,
            y=errs_y,
            usage=aux_sem['usage'].to_numpy(),
            speed=aux_sem['speed'].to_numpy(),
            speed_2d=aux_sem['velocity_2d_mm'].to_numpy(),
            speed_3d=aux_sem['velocity_3d_mm'].to_numpy(),
            height=aux_sem['height_ave_mm'].to_numpy(),
            dist_to_center=aux_sem['dist_to_center'].to_numpy(),
            sem=aux_sem[stat].to_numpy(),
            number=sem.index,
            label=labels,
            desc=desc,
            movies=cm_paths,
        ))

        # Draw glyphs
        line = fig.line('x', 'y', source=source, alpha=0.8, muted_alpha=0.1, line_dash=line_dash,
                        legend_label=groups[i], color=color)
        circle = fig.circle('x', 'y', source=source, alpha=0.8, muted_alpha=0.1,
                            legend_label=groups[i], color=color, size=6)

        tooltips = """
                    <div>
                        <div><span style="font-size: 12px; font-weight: bold;">syllable: @number{0}</span></div>
                        <div><span style="font-size: 12px;">usage: @usage{0.000}</span></div>
                        <div><span style="font-size: 12px;">centroid speed: @speed{0.000} mm/s</span></div>
                        <div><span style="font-size: 12px;">2D velocity: @speed_2d{0.000} mm/s</span></div>
                        <div><span style="font-size: 12px;">3D velocity: @speed_3d{0.000} mm/s</span></div>
                        <div><span style="font-size: 12px;">Height: @height{0.000} mm</span></div>
                        <div><span style="font-size: 12px;">Normalized Distance to Center: @dist_to_center{0.000}</span></div>
                        <div><span style="font-size: 12px;">group-SEM: @sem{0.000}</span></div>
                        <div><span style="font-size: 12px;">label: @label</span></div>
                        <div><span style="font-size: 12px;">description: @desc</span></div>
                        <div>
                            <video
                                src="@movies"; height="260"; alt="@movies"; width="260"; preload="true";
                                style="float: left; type: "video/mp4"; "margin: 0px 15px 15px 0px;"
                                border="2"; autoplay loop
                            ></video>
                        </div>
                    </div>
                  """

        hover = HoverTool(renderers=[circle],
                          tooltips=tooltips,
                          point_policy='snap_to_data',
                          line_policy='nearest')
        fig.add_tools(hover)

        error_bars = fig.multi_line('x', 'y', source=err_source, alpha=0.8, muted_alpha=0.1, legend_label=groups[i],
                                    color=color)

        if groupby == 'group':
            picker = ColorPicker(title=f"{groups[i]} Line Color")
            picker.js_link('color', line.glyph, 'line_color')
            picker.js_link('color', circle.glyph, 'fill_color')
            picker.js_link('color', circle.glyph, 'line_color')
            picker.js_link('color', error_bars.glyph, 'line_color')

            pickers.append(picker)

    return pickers

def bokeh_plotting(df, stat, sorting, mean_df=None, groupby='group', errorbar='SEM', syllable_families=None, sort_name='usage'):
    '''
    Generates a Bokeh plot with interactive tools such as the HoverTool, which displays
    additional syllable information and the associated crowd movie.

    Parameters
    ----------
    df (pd.DataFrame): Mean syllable statistic DataFrame.
    stat (str): Statistic to plot
    sorting (list): List of the current/selected syllable ordering
    groupby (str): Value to group data by. Either by unique group name or session name.
    errorbar (str): Error bar type to display
    syllable_families (dict): dict containing cladogram figure
    sort_name (str): Syllable sorting name displayed in title.

    Returns
    -------
    p (bokeh figure): Displayed stat plot with optional color pickers.
    '''

    tools = 'pan, box_zoom, wheel_zoom, save, reset'

    # Instantiate Bokeh figure with the HoverTool data
    p = figure(title=f'Syllable Statistics - Sorted by {sort_name}',
               width=850,
               height=500,
               tools=tools,
               x_range=syllable_families['cladogram'].x_range,
               x_axis_label='Syllables',
               y_axis_label=f'{stat}',
               output_backend="svg")

    # TODO: allow users to set their own colors
    colors = itertools.cycle(palette)

    # Set grouping variable to plot separately
    if groupby == 'group':
        groups = list(df.group.unique())
        group_colors = colors
    else:
        groups = list(df.SessionName.unique())
        tmp_groups = df[df['SessionName'].isin(groups)]

        sess_groups = []
        for s in groups:
            sess_groups.append(list(tmp_groups[tmp_groups['SessionName'] == s].group)[0])

        color_map = {}
        for i, g in enumerate(sess_groups):
            if g not in color_map.keys():
                color_map[g] = i

        group_color_map = {g: palette[color_map[g]] for g in sess_groups}
        group_colors = list(group_color_map.values())

        colors = [colorscale(group_color_map[sg], 0.5 + random.random()) for sg in sess_groups]

    if groupby != 'group':
        draw_stats(p, mean_df, list(df.group.unique()), group_colors, sorting, 'group', stat, errorbar, line_dash='dashed')

    if list(sorting) == syllable_families['leaves']:
        pickers = draw_stats(p, df, list(df.group.unique()), group_colors, sorting, groupby, stat, errorbar)
    else:
        pickers = draw_stats(p, df, groups, colors, sorting, groupby, stat, errorbar)

    # Get xtick labels
    label_df = df.groupby(['syllable', 'label'], as_index=False).mean().reindex(sorting)

    xtick_numbers = list(label_df['syllable'])
    xtick_labels = list(label_df['label'])
    xticks = [f'({num}) {lbl}' for num, lbl in zip(xtick_numbers, xtick_labels)]

    # Setting dynamics xticks
    p.xaxis.ticker = FixedTicker(ticks=list(sorting))
    p.xaxis.major_label_overrides = {i: str(l) for i, l in enumerate(list(xticks))}
    p.xaxis.major_label_orientation = np.pi / 4

    # Setting interactive legend
    p.legend.click_policy = "mute"
    p.legend.location = "top_right"

    output_grid = []
    if len(pickers) > 0:
        color_pickers = gridplot(pickers, ncols=2)
        output_grid.append(color_pickers)
    output_grid.append(p)

    graph_n_pickers = column(output_grid)

    ## Display
    show(graph_n_pickers)

    return p

def format_graphs(graphs, group):
    '''
    Formats multiple transition graphs to be stacked in vertical column-order with graph positions
    corresponding to the difference graphs.

    For example for 3 groups output would look like this:
    [ a  b-a c-a ]
    [    b   c-b ]
    [        c   ]

    Parameters
    ----------
    graphs (list): list of generated Bokeh figures.
    group (list): list of unique groups

    Returns
    -------
    formatted_plots (2D list): list of lists corresponding to rows of figures being plotted.
    '''

    # formatting plots into diagonal grid format
    ncols = len(group)

    group_grid = np.array([[None] * ncols] * ncols)

    # Draw main graphs on diagonal
    counter = 0
    for i in range(0, ncols):
        group_grid[i, i] = graphs[counter]
        counter += 1

    # Populate remainder of grid with network difference graphs
    for b in range(1, ncols):
        i = 0
        for j in range(b, ncols):
            group_grid[i, j] = graphs[counter]
            counter += 1
            i += 1

    # Use a rotating stack to align the grid difference plots in their appropriate row-column position
    for e, i in enumerate(range(ncols - 1, 1, -1)):
        col_items = deque(group_grid[:, i][::-1])

        for _ in range(e + 1):
            col_items.rotate(-1)

        tmp = list(col_items)
        # Moving empty plots to the bottom of the column list/beginning of the row list
        if None in tmp:
            tmp.append(tmp.pop(tmp.index(None)))

        group_grid[:, i] = tmp

    return list(group_grid)

def get_neighbors_and_entropies(graph, node_indices, entropies, entropy_rates, group_name):
    '''
    Computes the incoming and outgoing syllable entropies, entropy rates, previous nodes and
     neighboring nodes for all the nodes included in node_indices.

    Parameters
    ----------
    graph (networkx DiGraph): Generated DiGraph to convert to Bokeh glyph and plot.
    node_indices (list): List of node indices included in the given graph
    entropies (list): Syllable usage entropy values for each included syllable in the graph
    entropy_rates (list): Syllable transition entropy rates for all possible edge transitions in the graph
    group_name (str): Graph's group name.

    Returns
    -------
    entropy_in (list): List of computed incoming syllable entropy values for all the given node indices
    entropy_out (list): List of computed outgoing syllable entropy values for all the given node indices
    prev_states (list): List of previous nodes for each node index in the graph.
    next_states (list): List of successor nodes/syllables for each node in the graph
    neighbor_edge_colors (list): List of colors determining whether an edge is incoming or outgoing from each node.
     Where orange = incoming, and purple = outgoing
    '''

    # get selected node neighboring edge colors
    neighbor_edge_colors = {}

    # get node directed neighbors
    prev_states, next_states = [], []

    # get average entropy_in and out
    entropy_in, entropy_out = [], []
    for n in node_indices:
        try:
            # Get predecessor and neighboring states
            pred = np.array(list(graph.predecessors(n)))
            neighbors = np.array(list(graph.neighbors(n)))

            e_ins, e_outs = [], []
            for p in pred:
                e_in = entropy_rates[p][n] + (entropies[n] + entropies[p])
                e_ins.append(e_in)

                neighbor_edge_colors[(p, n)] = 'orange'

            for nn in neighbors:
                e_out = entropy_rates[n][nn] + (entropies[nn] + entropies[n])
                e_outs.append(e_out)

                neighbor_edge_colors[(n, nn)] = 'purple'

            # Get predecessor and next state transition weights
            pred_weights = [graph.edges()[(p, n)]['weight'] for p in pred]
            next_weights = [graph.edges()[(n, p)]['weight'] for p in neighbors]

            # Get descending order of weights
            pred_sort_idx = np.argsort(pred_weights)[::-1]
            next_sort_idx = np.argsort(next_weights)[::-1]

            # Get transition likelihood-sorted previous and next states
            prev_states.append(pred[pred_sort_idx])
            next_states.append(neighbors[next_sort_idx])

            entropy_in.append(np.nanmean(e_ins))
            entropy_out.append(np.nanmean(e_outs))
        except nx.NetworkXError:
            # handle orphans
            print('missing', group_name, n)
            pass

    return entropy_in, entropy_out, prev_states, next_states, neighbor_edge_colors

def plot_interactive_transition_graph(graphs, pos, group, group_names, usages,
                                      syll_info, entropies, entropy_rates,
                                      scalars, scalar_color='default'):
    '''

    Converts the computed networkx transition graphs to Bokeh glyph objects that can be interacted with
    and updated throughout run-time.

    Parameters
    ----------
    graphs (list of nx.DiGraphs): list of created networkx graphs.
    pos (nx.Layout): shared node position coordinates layout object.
    group (list): list of unique group names.
    group_names (list): list of names for all the generated transition graphs + difference graphs
    usages (list of OrdreredDicts): list of OrderedDicts containing syllable usages.
    syll_info (dict): dict of syllable label information to display with HoverTool
    scalars (dict): dict of syllable scalar information to display with HoverTool

    Returns
    -------
    '''

    warnings.filterwarnings('ignore')

    rendered_graphs, plots = [], []

    for i, graph in enumerate(graphs):

        node_indices = [n for n in graph.nodes if n in usages[i].keys()]

        if len(plots) == 0:
            plot = figure(title=f"{group_names[i]}", x_range=(-1.2, 1.2), y_range=(-1.2, 1.2))
        else:
            # Connecting pan-zoom interaction across plots
            plot = figure(title=f"{group_names[i]}", x_range=plots[0].x_range, y_range=plots[0].y_range)

        tooltips = """
                        <div>
                            <div><span style="font-size: 12px; font-weight: bold;">syllable: @number{0}</span></div>
                            <div><span style="font-size: 12px;">label: @label</span></div>
                            <div><span style="font-size: 12px;">description: @desc</span></div>
                            <div><span style="font-size: 12px;">usage: @usage{0.000}</span></div>
                            <div><span style="font-size: 12px;">centroid speed: @speed{0.000} mm/s</span></div>
                            <div><span style="font-size: 12px;">2D velocity: @speed_2d{0.000} mm/s</span></div>
                            <div><span style="font-size: 12px;">3D velocity: @speed_3d{0.000} mm/s</span></div>
                            <div><span style="font-size: 12px;">Height: @height{0.000} mm</span></div>
                            <div><span style="font-size: 12px;">Normalized Distance to Center: @dist_to_center{0.000}</span></div>
                            <div><span style="font-size: 12px;">Entropy-In: @ent_in{0.000}</span></div>
                            <div><span style="font-size: 12px;">Entropy-Out: @ent_out{0.000}</span></div>
                            <div><span style="font-size: 12px;">Next Syllable: @next</span></div>
                            <div><span style="font-size: 12px;">Previous Syllable: @prev</span></div>
                            <div>
                                <video
                                    src="@movies"; height="260"; alt="@movies"; width="260"; preload="true";
                                    style="float: left; type: "video/mp4"; "margin: 0px 15px 15px 0px;"
                                    border="2"; autoplay loop
                                ></video>
                            </div>
                        </div>
                   """

        # adding interactive tools
        plot.add_tools(HoverTool(tooltips=tooltips, line_policy='interp'),
                       TapTool(),
                       BoxSelectTool())

        entropy_in, entropy_out, prev_states, next_states, neighbor_edge_colors = \
            get_neighbors_and_entropies(graph, node_indices, entropies[i], entropy_rates[i], group_names[i])

        # edge colors for difference graphs
        if i >= len(group):
            edge_color = {e: 'red' if graph.edges()[e]['weight'] > 0 else 'blue' for e in graph.edges()}
            edge_width = {e: graph.edges()[e]['weight'] * 350 for e in graph.edges()}
        else:
            edge_color = {e: 'black' for e in graph.edges()}
            edge_width = {e: graph.edges()[e]['weight'] * 200 for e in graph.edges()}

        selected_edge_colors = {e: neighbor_edge_colors[e] for e in graph.edges()}

        # setting edge attributes
        nx.set_edge_attributes(graph, edge_color, "edge_color")
        nx.set_edge_attributes(graph, selected_edge_colors, "line_color")
        nx.set_edge_attributes(graph, edge_width, "edge_width")

        # get usages
        group_usage = [usages[i][j] for j in node_indices if j in usages[i].keys()]

        # get speeds
        group_speed = [scalars['speed'][i][j] for j in node_indices if j in scalars['speed'][i].keys()]
        group_speed_2d = [scalars['speeds_2d'][i][j] for j in node_indices if j in scalars['speeds_2d'][i].keys()]
        group_speed_3d = [scalars['speeds_3d'][i][j] for j in node_indices if j in scalars['speeds_3d'][i].keys()]

        # get average height
        group_height = [scalars['heights'][i][j] for j in node_indices if j in scalars['heights'][i].keys()]

        # get mean distances to bucket centers
        group_dist = [scalars['dists'][i][j] for j in node_indices if j in scalars['dists'][i].keys()]

        # node colors for difference graphs
        if i >= len(group):
            node_color = {s: 'red' if usages[i][s] > 0 else 'blue' for s in node_indices}
            node_size = {s: max(15., 10 + abs(usages[i][s] * 500)) for s in node_indices}
        else:
            node_color = {s: 'red' for s in node_indices}
            node_size = {s: max(15., abs(usages[i][s] * 500)) for s in node_indices}

        # setting node attributes
        nx.set_node_attributes(graph, node_color, "node_color")
        nx.set_node_attributes(graph, node_size, "node_size")

        # create bokeh-fied networkx transition graph
        graph_renderer = from_networkx(graph, pos, scale=1, center=(0, 0))

        # getting hovertool info
        labels, descs, cm_paths = [], [], []

        for n in node_indices:
            labels.append(syll_info[str(n)]['label'])
            descs.append(syll_info[str(n)]['desc'])
            cm_paths.append(syll_info[str(n)]['crowd_movie_path'])

        # setting common data source to display via HoverTool
        graph_renderer.node_renderer.data_source.add(node_indices, 'number')
        graph_renderer.node_renderer.data_source.add(labels, 'label')
        graph_renderer.node_renderer.data_source.add(descs, 'desc')
        graph_renderer.node_renderer.data_source.add(cm_paths, 'movies')
        graph_renderer.node_renderer.data_source.add(prev_states, 'prev')
        graph_renderer.node_renderer.data_source.add(next_states, 'next')
        graph_renderer.node_renderer.data_source.add(group_usage, 'usage')
        graph_renderer.node_renderer.data_source.add(group_speed, 'speed')
        graph_renderer.node_renderer.data_source.add(group_speed_2d, 'speed_2d')
        graph_renderer.node_renderer.data_source.add(group_speed_3d, 'speed_3d')
        graph_renderer.node_renderer.data_source.add(group_height, 'height')
        graph_renderer.node_renderer.data_source.add(group_dist, 'dist_to_center')
        graph_renderer.node_renderer.data_source.add(np.nan_to_num(entropy_in), 'ent_in')
        graph_renderer.node_renderer.data_source.add(np.nan_to_num(entropy_out), 'ent_out')

        text_color = 'white'

        # node interactions
        if scalar_color == 'Centroid Speed':
            fill_color = linear_cmap('speed', "Spectral4", 0, max(group_speed))
        elif scalar_color == '2D velocity':
            fill_color = linear_cmap('speed_2d', "Spectral4", 0, max(group_speed_2d))
        elif scalar_color == '3D velocity':
            fill_color = linear_cmap('speed_3d', "Spectral4", 0, max(group_speed_3d))
        elif scalar_color == 'Height':
            fill_color = linear_cmap('height', "Spectral4", 0, max(group_height))
        elif scalar_color == 'Distance to Center':
            fill_color = linear_cmap('dist_to_center', "Spectral4", 0, max(group_dist))
        elif scalar_color == 'Entropy-In':
            fill_color = linear_cmap('ent_in', "Spectral4", 0, max(np.nan_to_num(entropy_in)))
        elif scalar_color == 'Entropy-Out':
            fill_color = linear_cmap('ent_out', "Spectral4", 0, max(entropy_out))
        else:
            fill_color = 'white'
            text_color = 'black'

        if fill_color != 'white':
            color_bar = ColorBar(color_mapper=fill_color['transform'], location=(0, 0))
            plot.add_layout(color_bar, 'below')

        graph_renderer.node_renderer.glyph = Circle(size='node_size', fill_color=fill_color, line_color='node_color')
        graph_renderer.node_renderer.selection_glyph = Circle(size='node_size', line_color='node_color', fill_color=fill_color)
        graph_renderer.node_renderer.nonselection_glyph = Circle(size='node_size', line_color='node_color', fill_color=fill_color)
        graph_renderer.node_renderer.hover_glyph = Circle(size='node_size', fill_color=Spectral4[1])

        # edge interactions
        graph_renderer.edge_renderer.glyph = MultiLine(line_color='edge_color', line_alpha=0.7,
                                                       line_width='edge_width', line_join='miter')
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color='line_color', line_width='edge_width',
                                                                 line_join='miter', line_alpha=0.8, )
        graph_renderer.edge_renderer.nonselection_glyph = MultiLine(line_color='edge_color', line_alpha=0.0,
                                                                    line_width='edge_width', line_join='miter')
        ## Change line color to match difference colors
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color='line_color', line_width=5,
                                                             line_join='miter', line_alpha=0.8)

        # selection policies
        graph_renderer.selection_policy = NodesAndLinkedEdges()
        graph_renderer.inspection_policy = NodesAndLinkedEdges()

        # added rendered graph to plot
        plot.renderers.append(graph_renderer)

        # get node positions
        if len(plots) == 0:
            x, y = zip(*graph_renderer.layout_provider.graph_layout.values())
            syllable = list(graph.nodes)
        else:
            new_layout = {k: rendered_graphs[0].layout_provider.graph_layout[k] for k in
                          graph_renderer.layout_provider.graph_layout.keys()}
            x, y = zip(*new_layout.values())
            syllable = [a if a in node_indices else '' for a in list(new_layout.keys())]

        # create DataSource for node info
        label_source = ColumnDataSource({'x': x,
                                         'y': y,
                                         'syllable': syllable
                                         })

        # create the LabelSet to render
        labels = LabelSet(x='x', y='y',
                          x_offset=-7, y_offset=-7,
                          text='syllable', source=label_source,
                          text_color=text_color, text_font_size="12px",
                          background_fill_color=None,
                          render_mode='canvas')

        # render labels
        plot.renderers.append(labels)

        o_line = plot.line(line_color='orange')
        p_line = plot.line(line_color='purple')
        r_line = plot.line(line_color='red')
        b_line = plot.line(line_color='blue')

        items = [
            LegendItem(label="Incoming Transition", renderers=[o_line]),
            LegendItem(label="Outgoing Transition", renderers=[p_line]),
        ]

        if i >= len(group):
            items += [LegendItem(label="Up-regulated", renderers=[r_line])]
            items += [LegendItem(label="Down-regulated", renderers=[b_line])]

        legend = Legend(items=items,
                       border_line_color="black", background_fill_color='white',
                       background_fill_alpha=1.0)
        plot.renderers.append(legend)

        plots.append(plot)
        rendered_graphs.append(graph_renderer)

    # Format grid of transition graphs
    formatted_plots = format_graphs(plots, group)

    # Create Bokeh grid plot object
    gp = gridplot(formatted_plots, plot_width=550, plot_height=550)
    show(gp)