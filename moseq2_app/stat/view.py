import random
import itertools
import numpy as np
from bokeh.layouts import column
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.models.tickers import FixedTicker
from bokeh.palettes import Category10_10 as palette
from bokeh.models import (ColumnDataSource, BoxSelectTool, HoverTool, TapTool, ColorPicker)

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