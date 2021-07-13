'''

View module to facilitate graphing of interactive statistics tools: the Dendrogram, statistics plot,
 and transition graph grid. Module primarily uses Bokeh to facilitate the interactive graphing functionality.

'''

import random
import warnings
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from os.path import relpath
from collections import deque
from bokeh.layouts import column
from bokeh.layouts import gridplot
from bokeh.palettes import Spectral4
from bokeh.transform import linear_cmap
from bokeh.models.tickers import FixedTicker
from bokeh.palettes import Category10_10 as palette
from bokeh.plotting import figure, show, from_networkx
from moseq2_app.stat.widgets import SyllableStatBokehCallbacks
from bokeh.models import (ColumnDataSource, LabelSet, BoxSelectTool, Circle, ColorBar, RangeSlider, CustomJS, TextInput,
                          Legend, LegendItem, HoverTool, MultiLine, NodesAndLinkedEdges, TapTool, ColorPicker)

def graph_dendrogram(obj, syll_info):
    '''
    Graphs the distance sorted dendrogram representing syllable neighborhoods. Distance sorting
    is computed by processing the respective syllable AR matrices.

    Parameters
    ----------
    obj (InteractiveSyllableStats object): Syllable Stats object containing syllable stat information.
    syll_info (dict): dict object containing syllable numbers paired with dicts of their labels and descriptions.

    Returns
    -------
    '''

    ## Cladogram figure
    cladogram = figure(title='Distance Sorted Syllable Dendrogram',
                       width=850,
                       height=500,
                       output_backend="svg")

    # Get distance sorted label ordering
    labels = list(map(int, obj.results['ivl']))
    sources = []
    # Each (icoord, dcoord) pair represents a single branch in the dendrogram
    for i, d in zip(obj.icoord, obj.dcoord):
        tmp = list(zip(i, d))
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

    xtick_labels = [syll_info.get(lbl, {'label': ''})['label'] for lbl in labels]
    xticks = [f'{lbl} ({num})' for num, lbl in zip(labels, xtick_labels)]

    # Set x-axis ticks
    cladogram.xaxis.ticker = FixedTicker(ticks=labels)
    cladogram.xaxis.major_label_overrides = {i: str(l) for i, l in enumerate(list(xticks))}
    cladogram.xaxis.major_label_orientation = np.pi / 4

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

def get_ci_vect_vectorized(x, n_boots=10000, n_samp=None, function=np.nanmean, pct=5):
    '''
    Compute min and max values within a (default) 95th percentile of the inputted syllable statistic.

    Parameters
    ----------
    x (pandas Series): 1D list of syllable statistics.
    n_boots (int): Number of bootstrapped examples to generate to compute the percentile.
    n_samp (int): Number of inputted syllables.
    function (numpy function): Function to apply to bootstrapped data.
    pct (int): Percentile to compute.

    Returns
    -------
    percentile (2d numpy array): an array of min and max values for each syllable's stat value
    '''

    if isinstance(x, pd.core.series.Series):
        x = x.values
    pct /= 2
    n_vals = len(x)
    if n_samp is None:
        n_samp = n_vals
    boots = function(x[np.random.choice(n_vals, size=(n_samp, n_boots))], axis=0)

    percentile = np.percentile(boots, [pct, 100 - pct])
    return percentile

def setup_syllable_search(src_dict, err_dict, err_source, searchbox, circle, line):
    '''
    Initializes the CustomJS script callback function used to update the plot upon changing the
     value of the TextInput, a.k.a, Syllable Spotlight/Search widget.
     Will update the interactive bokeh plot to only show syllables with labels that are substrings of the inputted
     search string.
    Parameters
    ----------
    src_dict (dict): dict object containing all the stats information contained in the main ColumnDataSource.
    err_dict (dict): dict object containing all the error margins for the src_dict statistics.
    err_source (bokeh.models.ColumnDataSource): data source to update based on slider values.
     (Will hide error bars for excluded syllables).
    searchbox (bokeh.models.TextInput): text input widget for users to input syllable labels to find.
    circle (bokeh.glyph circle): drawn bokeh glyph representing syllables, that will be updated in the callback function
    line (bokeh.glyph line): drawn line connecting all the circle nodes in the bokeh figure.
     (To be hidden in the callback function if the syllable list is discontinuous)

    Returns
    -------
    callback (bokeh.models.CustomJS): javascript callback function to embed to search bar and hover tool objects.
    '''

    js_condition = '''if(data['label'][i].toLowerCase().includes(searchbox.value.toLowerCase())) {\n'''
    js_cbs = SyllableStatBokehCallbacks(condition=js_condition)

    callback = CustomJS(
        args=dict(source=circle.data_source, err_source=err_source, searchbox=searchbox,
                  data=src_dict, err_data=err_dict, line=line), code=js_cbs.code)

    return callback

def setup_slider(src_dict, err_dict, err_source, slider, circle, line, thresh_stat='usage'):
    '''
    Initializes the CustomJS script callback function used to update the plot upon changing the
     value of the RangeSlider.
     The JS script will iterate through each syllable and update the ColumnDataSource objects based the selected
     statistic to threshold, and the current range to threshold.
     Once the ColumnDataSource objects are updated, the bokeh plot will automatically be updated.

    Parameters
    ----------
    src_dict (dict): dict object containing all the stats information contained in the main ColumnDataSource.
    err_dict (dict): dict object containing all the error margins for the src_dict statistics.
    err_source (bokeh.models.ColumnDataSource): data source to update based on slider values.
     (Will hide error bars for excluded syllables).
    slider (bokeh.models.RangeSlider): slider object with the currently displayed values used to filter
     out syllables in the callback function.
    circle (bokeh.glyph circle): drawn bokeh glyph representing syllables, that will be updated in the callback function
    line (bokeh.glyph line): drawn line connecting all the circle nodes in the bokeh figure.
     (To be hidden in the callback function if the syllable list is discontinuous)
    thresh_stat (str): name of the statistic to threshold by.

    Returns
    -------
    callback (bokeh.models.CustomJS): javascript callback function to embed to slider and hover tool objects.
    '''

    # map the dropdown values back to datasource names to retrieve in the javascript callback function
    dict_mapping = {
        'usage': 'usage',
        'velocity_2d_mm': 'speed_2d',
        'velocity_3d_mm': 'speed_3d',
        'height_ave_mm': 'height',
        'dist_to_center_px': 'dist_to_center'
    }

    js_condition = '''if((data[thresh_stat][i] >= slider.value[0]) && (data[thresh_stat][i] <= slider.value[1])) {\n'''
    js_cbs = SyllableStatBokehCallbacks(condition=js_condition)

    callback = CustomJS(
        args=dict(source=circle.data_source, err_source=err_source,
                  data=src_dict, err_data=err_dict, thresh_stat=dict_mapping[thresh_stat],
                  slider=slider, line=line), code=js_cbs.code)

    return callback

def setup_hovertool(renderers, callback=None):
    '''
    Initialize hover tool with tooltips showing all the syllable information and the crowd movies upon
     hovering over a syllable circle glyph.

    Parameters
    ----------
    renderers (list bokeh.Renderer Instances): drawn bokeh glyph representing syllables, that will be updated in the callback function
    callback (bokeh.models.CustomJS): javascript callback function to embed to hover tool objects to preserve alignment.

    Returns
    -------
    hover (bokeh.models.HoverTool): hover tool to embed into the created figure.
    '''
    
    # html divs to display within the HoverTool
    tooltips = """
                <div>
                    <div><span style="font-size: 12px; font-weight: bold;">syllable: @number{0}</span></div>
                    <div><span style="font-size: 12px;">usage: @usage{0.000}</span></div>
                    <div><span style="font-size: 12px;">2D velocity: @speed_2d{0.000} mm/s</span></div>
                    <div><span style="font-size: 12px;">3D velocity: @speed_3d{0.000} mm/s</span></div>
                    <div><span style="font-size: 12px;">Height: @height{0.000} mm</span></div>
                    <div><span style="font-size: 12px;">Distance to Center px: @dist_to_center{0.000}</span></div>
                    <div><span style="font-size: 12px;">group-error: +/- @sem{0.000}</span></div>
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

    hover = HoverTool(renderers=renderers,
                      callback=callback,
                      tooltips=tooltips,
                      point_policy='snap_to_data',
                      line_policy='nearest')

    return hover

def get_aux_stat_dfs(df, group, sorting, groupby='group', errorbar='CI 95%', stat='usage'):
    '''
    Computes the group-specific syllable statistics dataframe, and the selected error values to
     later draw the line plot and error bars.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing all relevant data to plot.
    group (str): group name to get auxiliary statistics dataframe for.
    sorting (1D list): list of syllable index values to resort the dataframe by.
    groupby (str): column to group the syllable stats by.
    errorbar (str): name of the error bar type to compute values for.
    stat (str): name of the statistic to plot.

    Returns
    -------
    aux_df (pd.DataFrame): dataframe that only contains the selected group's mean statistics.
    stat_err (pd.DataFrame): dataframe that contains the error values for the selected statistic.
    aux_err (pd.DataFrame): dataframe that contains the error values for all the statistics.
    errs_x (list): list of x-indices to plot the error bar lines within.
    errs_y (list): list of y-indices to plot the error bar lines within.
    '''

    # Get group specific dataframe indices
    df_group = df[df[groupby] == group]
    grouped = df_group.groupby('syllable')[[stat]]

    # Get resorted mean syllable data
    aux_df = df_group.groupby('syllable', as_index=False).mean().reindex(sorting)

    # Get SEM
    if errorbar == 'CI 95%':
        stat_err = df_group.groupby('syllable')[stat].apply(get_ci_vect_vectorized).reindex(sorting)
        aux_err = {}
        for s in ['usage', 'velocity_2d_mm', 'velocity_3d_mm', 'height_ave_mm', 'dist_to_center_px']:
            aux_err[s] = df_group.groupby('syllable')[s].apply(get_ci_vect_vectorized).reindex(sorting)
    elif errorbar == 'SEM':
        stat_err = grouped.sem().reindex(sorting)
        aux_err = df_group.groupby('syllable', as_index=False).sem().reindex(sorting)
    else:
        stat_err = grouped.std().reindex(sorting)
        aux_err = df_group.groupby('syllable', as_index=False).std().reindex(sorting)

    # Get min and max error bar values
    if errorbar == 'CI 95%':
        miny = [e[0] for e in stat_err]
        maxy = [e[1] for e in stat_err]
    else:
        miny = aux_df[stat] - stat_err[stat]
        maxy = aux_df[stat] + stat_err[stat]

    # get error bar coordinates to plot in the figure
    errs_x = [(i, i) for i in range(len(aux_df.index))]
    errs_y = [(min_y, max_y) for min_y, max_y in zip(miny, maxy)]

    return aux_df, stat_err, aux_err, errs_x, errs_y

def get_syllable_info(df, sorting):
    '''
    Returns the labels, descriptions and crowd movie paths for all the syllables to display in the x-axis,
     and hover tool.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing all relevant data to plot.
    sorting (1D list): list of syllable index values to resort the dataframe by.

    Returns
    -------
    labels (1D numpy array): syllable label list sorted by the given sorting order.
    desc (1D numpy array): syllable description list sorted by the given sorting order.
    cm_paths (1D numpy array): syllable crowd movie path list sorted by the given sorting order.
    '''

    # Get Labeled Syllable Information
    info_columns = ['syllable', 'label', 'desc', 'crowd_movie_path']
    desc_data = df.groupby(info_columns, as_index=False).mean()[info_columns].reindex(sorting)

    # Pack data into numpy arrays
    labels = desc_data['label'].to_numpy()
    desc = desc_data['desc'].to_numpy()

    cm_paths = []
    for cm in desc_data['crowd_movie_path'].to_numpy():
        try:
            cm_paths.append(relpath(cm))
        except ValueError:
            # cm path does not exist
            cm_paths.append('')

    return labels, desc, cm_paths

def get_datasources(aux_df, aux_sem, sem, labels, desc, cm_paths, errs_x, errs_y, stat):
    '''
    Creates Bokeh ColumnDataSources that will be used to draw all the bokeh glyphs (circle, line plots, and errorbars).
    Additionally, the data sources will be used by the figure's JS callbacks in order to update the graph in real time
    when users edit the slider or color picker widget values.

    Parameters
    ----------
    aux_df (pd.DataFrame): DataFrame that only contains the selected group's mean statistics.
    aux_sem (pd.DataFrame): DataFrame that contains the error values for all the statistics.
    sem (pd.DataFrame): DataFrame that contains the error values for the selected statistic.
    labels (1D numpy array): Syllable label list sorted by the given sorting order.
    desc (1D numpy array): Syllable description list sorted by the given sorting order.
    cm_paths (1D numpy array): Syllable crowd movie path list sorted by the given sorting order.
    errs_x (list): List of x-indices to plot the error bar lines within.
    errs_y (list): List of y-indices to plot the error bar lines within.
    stat (str): Statistic to display.

    Returns
    -------
    source (bokeh.models.ColumnDataSource): Bokeh data source of mean syllable stats,
     used to plot interactive figure glyphs, hovertools and widgets.
    src_dict (dict): dict version of the ColumnDataSource object that will be passed to the JS Callback.
    err_source (bokeh.models.ColumnDataSource): Bokeh data source of syllable stats error values,
     used to plot interactive figure errorbars, hovertool and widget values.
    err_dict (dict): dict version of the errorbar ColumnDataSource object that will be passed to the JS Callback.
    '''

    # stat data source
    src_dict = dict(
        x=list(range(len(aux_df.index))),
        y=aux_df[stat].to_numpy(),
        usage=aux_df['usage'].to_numpy(),
        speed_2d=aux_df['velocity_2d_mm'].to_numpy(),
        speed_3d=aux_df['velocity_3d_mm'].to_numpy(),
        height=aux_df['height_ave_mm'].to_numpy(),
        dist_to_center=aux_df['dist_to_center_px'].to_numpy(),
        sem=aux_sem[stat].to_numpy(),
        number=sem.index,
        label=labels,
        desc=desc,
        movies=cm_paths,
    )
    source = ColumnDataSource(data=src_dict)

    # SEM data source
    err_dict = dict(
        x=errs_x,
        y=errs_y,
        usage=aux_sem['usage'].to_numpy(),
        speed_2d=aux_sem['velocity_2d_mm'].to_numpy(),
        speed_3d=aux_sem['velocity_3d_mm'].to_numpy(),
        height=aux_sem['height_ave_mm'].to_numpy(),
        dist_to_center=aux_sem['dist_to_center_px'].to_numpy(),
        sem=aux_sem[stat].to_numpy(),
        number=sem.index,
        label=labels,
        desc=desc,
        movies=cm_paths,
    )
    err_source = ColumnDataSource(data=err_dict)

    return source, src_dict, err_source, err_dict

def draw_stats(fig, df, groups, colors, sorting, groupby, stat, errorbar, line_dash='solid', thresh_stat='usage', sig_sylls=[]):
    '''
    Helper function to bokeh_plotting that iterates through the given DataFrame and plots the
    data grouped by some user defined column ('group', 'SessionName', 'SubjectName'), with the errorbars of their
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
    pickers (list of ColorPickers): List of interactive color picker widgets to update the graph colors.
    slider (bokeh.models.RangeSlider): RangeSlider object used to threshold/filter the displayed syllables.
    '''
    warnings.filterwarnings('ignore')

    pickers = []

    slider = RangeSlider(start=0, end=0.001, value=(0, 0.001), step=0.001,
                         format="0[.]000", title=f"Display Syllables Within {thresh_stat} Range")

    searchbox = TextInput(value='', title='Syllable to Display:')

    for group, color in zip(groups, colors):

        aux_df, sem, aux_sem, errs_x, errs_y = get_aux_stat_dfs(df, group, sorting, groupby, errorbar, stat)

        # get syllable info
        labels, desc, cm_paths = get_syllable_info(df, sorting)

        # get bokeh data sources
        source, src_dict, err_source, err_dict = get_datasources(aux_df, aux_sem, sem,
                                                                 labels, desc, cm_paths,
                                                                 errs_x, errs_y, stat)

        # adjust slider min-max values
        if aux_df[thresh_stat].max() > slider.end:
            slider.end = aux_df[thresh_stat].max()
            slider.value = (0, slider.end)

        # Draw glyphs
        line = fig.line('x', 'y', source=source, alpha=0.8, muted_alpha=0.1, line_dash=line_dash,
                        legend_label=group, color=color)
        circle = fig.circle('x', 'y', source=source, alpha=0.8, muted_alpha=0.1,
                            legend_label=group, color=color, size=6)

        if len(sig_sylls) > 0:
            y = aux_df[stat].to_numpy()[sig_sylls]
            # Draw stars instead of circles
            diamond = fig.diamond_cross(sig_sylls, y, alpha=0.8, muted_alpha=0.1, legend_label='Significant Syllable',
                                        fill_color=color, line_width=3, line_color='red', size=10)

        error_bars = fig.multi_line('x', 'y', source=err_source, alpha=0.8,
                                    muted_alpha=0.1, legend_label=group, color=color)

        # setup searchbox callback function
        # callback function will allow user interaction to dynamically edit the circle.ColumnDataSource
        search_callback = setup_syllable_search(src_dict, err_dict, err_source, searchbox, circle, line)
        searchbox.js_on_change('value', search_callback)

        # setup slider callback function to update the plot
        # callback function will allow user interaction to dynamically edit the circle.ColumnDataSource
        slider_callback = setup_slider(src_dict, err_dict, err_source, slider, circle, line, thresh_stat)
        slider.js_on_change('value', slider_callback)

        # update hover tools to match the thresholded plot points
        hover = setup_hovertool(fig.renderers)
        fig.add_tools(hover)
        # set up color pickers and link the selection to all the drawn glyphs
        if groupby == 'group':
            picker = ColorPicker(title=f"{group} Line Color")
            picker.js_link('color', line.glyph, 'line_color')
            picker.js_link('color', circle.glyph, 'fill_color')
            picker.js_link('color', circle.glyph, 'line_color')
            picker.js_link('color', error_bars.glyph, 'line_color')

            pickers.append(picker)

    return pickers, slider, searchbox

def set_grouping_colors(df, groupby):
    '''
    Based on the selected grouping to plot, will return the unique group names,
     and their associated colors to use when plotting the default figure.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing all relevant data to plot.
    groupby (str): column to group the syllable stats by.

    Returns
    -------
    groups (list): list of group names to plot line plots for.
    group_colors (list): list of colors corresponding to each plotted group.
    colors (list): list of all the colors used to plot the glyphs
    '''

    colors = itertools.cycle(palette)

    # Set grouping variable to plot separately
    if groupby == 'group':
        groups = list(df.group.unique())
        group_colors = colors
    else:
        groups = list(df[groupby].unique())
        tmp_groups = df[df[groupby].isin(groups)]

        sess_groups = []
        for s in groups:
            sess_groups.append(list(tmp_groups[tmp_groups[groupby] == s].group)[0])

        # generate a list of unique groups
        unique_group = np.unique(sess_groups)
        # generate a dictionary for group index in the colo palette
        color_map = dict(zip(unique_group, range(len(unique_group))))
        for group, index in color_map.items():
            try:
                color_map[group] = palette[index]
            # handle index error when the number of groups is greater than the nubmer of colors in the palette
            except IndexError:
                print('Not enough color groups in the pallette')
                # set color index to the last item in pallette
                color_map[group] = palette[-1]
        group_colors = list(color_map.values())
        colors = [colorscale(color_map[sg], 0.5 + random.random()) for sg in sess_groups]
    return groups, group_colors, colors

def format_stat_plot(p, df, searchbox, slider, pickers, sorting):
    '''
    Edits the bokeh figures x-axis such that the syllable labels are also displayed, and are slanted 45 degrees.
     Sets the legend to be interactive where users can hide line plots by clicking on their legend item.
     Finally creates the bokeh gridplot to hold all of the displayed widgets above the graph to display.

    Parameters
    ----------
    p (bokeh.Figure): bokeh figure with all the glyphs already drawn.
    df (pd.DataFrame): DataFrame containing all relevant data to plot
    slider (bokeh.models.RangeSlider): slider object with the currently displayed values used to filter
     out syllables in the callback function.
    pickers (list of ColorPickers): List of interactive color picker widgets to insert into a gridplot to display.
    sorting (1D list): list of syllable index values to resort the dataframe by.

    Returns
    -------
    graph_n_pickers (bokeh.layout.column): Bokeh Layout object of the widgets and figure to display.
    '''

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

    # Create gridplot of color pickers
    output_grid = [searchbox, slider]
    if len(pickers) > 0:
        color_pickers = gridplot(pickers, ncols=2)
        output_grid.append(color_pickers)
    output_grid.append(p)

    # Pack widgets together with figure
    graph_n_pickers = column(output_grid)

    return graph_n_pickers

def bokeh_plotting(df, stat, sorting, mean_df=None, groupby='group', errorbar='SEM',
                   syllable_families=None, sort_name='usage', thresh='usage', sig_sylls=[]):
    '''
    Generates a Bokeh plot with interactive tools such as the HoverTool, which displays
    additional syllable information and the associated crowd movie.

    Parameters
    ----------
    df (pd.DataFrame): Mean syllable statistic DataFrame.
    stat (str): Statistic to plot
    sorting (list): List of the current/selected syllable ordering
    groupby (str): Value to group data by. Either by unique group name, session name, or subject name.
    errorbar (str): Error bar type to display
    syllable_families (dict): dict containing cladogram figure
    sort_name (str): Syllable sorting name displayed in title.
    thresh (str): Statistic to threshold syllables by using the Range Slider

    Returns
    -------
    p (bokeh figure): Displayed stat plot with optional color pickers.
    '''

    tools = 'pan, box_zoom, wheel_zoom, save, reset'

    # Instantiate Bokeh figure with the HoverTool data
    p = figure(title=f'Syllable {stat} Statistics - Sorted by {sort_name}',
               width=850,
               height=500,
               tools=tools,
               x_range=syllable_families['cladogram'].x_range,
               x_axis_label='Syllables',
               y_axis_label=f'{stat}',
               output_backend="svg")

    # get default colors to display each group's line plots
    groups, group_colors, colors = set_grouping_colors(df, groupby)

    if groupby != 'group':
        # draw session based statistics, without returning individual bokeh widgets to display
        draw_stats(p, mean_df, list(df.group.unique()), group_colors, sorting, 'group',
                   stat, errorbar, line_dash='dashed', thresh_stat=thresh, sig_sylls=sig_sylls)

    # draw line plots, setup hovertool, thresholding slider and group color pickers
    if list(sorting) == syllable_families['leaves']:
        pickers, slider, searchbox = draw_stats(p, df, list(df.group.unique()), group_colors,
                                     sorting, groupby, stat, errorbar, thresh_stat=thresh, sig_sylls=sig_sylls)
    else:
        pickers, slider, searchbox = draw_stats(p, df, groups, colors, sorting,
                                     groupby, stat, errorbar, thresh_stat=thresh, sig_sylls=sig_sylls)

    # Format Bokeh plot with widgets
    graph_n_pickers = format_stat_plot(p, df, searchbox, slider, pickers, sorting)

    # Display figure and widgets
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

def get_neighbors(graph, node_indices, group_name):
    '''
    Computes the incoming and outgoing syllable entropies, entropy rates, previous nodes and
     neighboring nodes for all the nodes included in node_indices.

    Parameters
    ----------
    graph (networkx DiGraph): Generated DiGraph to convert to Bokeh glyph and plot.
    node_indices (list): List of node indices included in the given graph
    group_name (str): Graph's group name.

    Returns
    -------
    prev_states (list): List of previous nodes for each node index in the graph.
    next_states (list): List of successor nodes/syllables for each node in the graph
    neighbor_edge_colors (list): List of colors determining whether an edge is incoming or outgoing from each node.
     Where orange = incoming, and purple = outgoing
    '''

    # get selected node neighboring edge colors
    neighbor_edge_colors = {}

    # get node directed neighbors
    prev_states, next_states = [], []

    for n in node_indices:
        try:
            # Get predecessor and neighboring states
            pred = np.array(list(graph.predecessors(n)))
            neighbors = np.array(list(graph.neighbors(n)))

            for p in pred:
                neighbor_edge_colors[(p, n)] = 'orange'

            for nn in neighbors:
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

        except nx.NetworkXError:
            # handle orphans
            print('missing', group_name, n)
            pass

    # correct colors for edges where the transition is both incoming and outgoing
    for k, v in neighbor_edge_colors.items():
        k1 = k[::-1]
        if k1 in neighbor_edge_colors:
            neighbor_edge_colors[k] = 'green'
            neighbor_edge_colors[k1] = 'green'

    return prev_states, next_states, neighbor_edge_colors

def format_plot(plot):
    '''
    Turns off all major and minor x,y ticks on the transition plot graphs

    Parameters
    ----------
    plot (bokeh Plot): Current graph being generated

    Returns
    -------
    '''

    plot.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    plot.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks

    plot.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    plot.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks

    plot.xaxis.major_label_text_color = None  # turn off x-axis tick labels leaving space
    plot.yaxis.major_label_text_color = None  # turn off y-axis tick labels leaving space

def get_minmax_tp(edge_width, diff=False):
    '''
    Computes the min and max transition probabilities given the rescaled edge-widths.
    If diff = True, the function will return 4 variables: min/max for down and up-regulated syllables,

    Parameters
    ----------
    edge_width (dict): dict of syllables paired with drawn edge widths
    diff (bool): indicates whether to compute min/max transition probs. for up and down-regulated syllables.

    Returns
    -------
    min_tp (float): min transition probability (min_down_tp if diff=True)
    max_tp (float): max transition probability (max_down_tp if diff=True)
     if diff == True
    min_up_tp (float): min transition probability in up-regulated syllable
    max_up_tp (float): max transition probability in up-regulated syllable
    '''

    if not diff:
        try:
            min_tp = min(list(edge_width.values())) / 200
            max_tp = max(list(edge_width.values())) / 200
        except ValueError:
            # handle empty list
            min_tp, max_tp = 0, 0
        return min_tp, max_tp
    else:
        # get min/max down
        try:
            min_down_tp = min([e for e in edge_width.values() if e < 0]) / 350
            max_down_tp = max([e for e in edge_width.values() if e < 0]) / 350
        except ValueError:
            # empty list
            min_down_tp, max_down_tp = 0, 0

        # get min/max up
        try:
            min_up_tp = min([e for e in edge_width.values() if e > 0]) / 350
            max_up_tp = max([e for e in edge_width.values() if e > 0]) / 350
        except ValueError:
            min_up_tp, max_up_tp = 0, 0

        return min_down_tp, max_down_tp, min_up_tp, max_up_tp

def get_difference_legend_items(plot, edge_width, group_name):
    '''
    Creates the difference graph legend items with the min and max transition probabilities
     for both the up and down-regulated transition probabilities.

    Parameters
    ----------
    plot (bokeh.figure): Bokeh plot to add legend to.
    edge_width (dict): Dictionary of edge widths
    group_name (str): Difference graph title.

    Returns
    -------
    diff_items (list): List of LegendItem objects to display
    '''

    r_line = plot.line(line_color='red')
    b_line = plot.line(line_color='blue')

    r_circle = plot.circle(line_color='red', fill_color='white')
    b_circle = plot.circle(line_color='blue', fill_color='white')

    G1 = group_name.split('-')[0]

    min_down_tp, max_down_tp, min_up_tp, max_up_tp = get_minmax_tp(edge_width, diff=True)

    min_down_line = plot.line(line_color='blue', line_width=min_down_tp * 350)
    max_down_line = plot.line(line_color='blue', line_width=max_down_tp * 350)

    min_up_line = plot.line(line_color='red', line_width=min_up_tp * 350)
    max_up_line = plot.line(line_color='red', line_width=max_up_tp * 350)

    diff_main_items = [
        LegendItem(label=f"Up-regulated Usage in {G1}", renderers=[r_circle]),
        LegendItem(label=f"Down-regulated Usage in {G1}", renderers=[b_circle]),
        LegendItem(label=f"Up-regulated P(transition) in {G1}", renderers=[r_line]),
        LegendItem(label=f"Down-regulated P(transition) in {G1}", renderers=[b_line]),
    ]

    diff_width_items = [
        LegendItem(label=f"Min Up-regulated P(transition): {min_up_tp:.4f}", renderers=[min_up_line]),
        LegendItem(label=f"Max Up-regulated P(transition): {max_up_tp:.4f}", renderers=[max_up_line]),
        LegendItem(label=f"Min Down-regulated P(transition): {min_down_tp:.4f}", renderers=[min_down_line]),
        LegendItem(label=f"Max Down-regulated P(transition): {max_down_tp:.4f}", renderers=[max_down_line]),
    ]

    return diff_main_items, diff_width_items

def set_fill_color(scalar_color, data_dict):
    '''
    Sets the node fill coloring based on the selected scalar value.
    Uses the inputted data_dict to get the key and array for the requested scalar.

    Parameters
    ----------
    scalar_color (str): name of scalar to color nodes by.
    data_dict (dict): dict containing dicts of scalar_df keys and their corresponding
     values to create the linear color map from.

    Returns
    -------
    fill_color (str or list): list of colors per node, or single color (white)
    empty (bool): indicator for whether to display a color bar.
    '''

    empty = False

    for arr in data_dict.values():
        if len(arr['values']) == 0:
            empty = True

    fill_color = 'white'

    try:
        if not empty and scalar_color in data_dict:
            fill_color = linear_cmap(data_dict[scalar_color]['key'],
                                     "Spectral4",
                                     0,
                                     max(data_dict[scalar_color]['values']))
    except ValueError:
        pass

    return fill_color, empty

def setup_trans_graph_tooltips(plot):
    '''
    Adds a hover tool, tap tool, and a box select tool to the plot, allowing the user to view a
     displayed preview of all the syllable node information, and highlight transitions coming to or from clicked node.

    Parameters
    ----------
    plot (bokeh figure): bokeh generated figure to add tools to.

    Returns
    -------
    '''

    tooltips = """
                <div>
                    <div><span style="font-size: 12px; font-weight: bold;">syllable: @number{0}</span></div>
                    <div><span style="font-size: 12px;">label: @label</span></div>
                    <div><span style="font-size: 12px;">description: @desc</span></div>
                    <div><span style="font-size: 12px;">usage: @usage{0.000}</span></div>
                    <div><span style="font-size: 12px;">2D velocity: @speed_2d{0.000} mm/s</span></div>
                    <div><span style="font-size: 12px;">3D velocity: @speed_3d{0.000} mm/s</span></div>
                    <div><span style="font-size: 12px;">Height: @height{0.000} mm</span></div>
                    <div><span style="font-size: 12px;">Distance to Center px: @dist_to_center_px{0.000}</span></div>
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

def format_trans_graph_edges(graph, neighbor_edge_colors, difference_graph=False):
    '''
    Computes the colors and widths of all the transition edges between nodes. Each individual group
    will always have edges colored black.If difference_graph is true,
    then the edges will be colored red if the difference is > 0, and blue otherwise.

    Parameters
    ----------
    graph (nx.DiGraph): networkx graph to read edge weights from and compute widths and colors with.
    neighbor_edge_colors (dict): dictionary of node transition tuples mapped to corresponding colors (str).
    difference_graph (bool): indicates whether to color the edges based on the transition difference between two groups.

    Returns
    -------
    edge_color (dict): dict of edge tuple(node1, node2) object mapped to string values describing edge colors.
    edge_width (dict): dict of edge tuple(node1, node2) object mapped to float values describing edge widths.
    selected_edge_colors (dict): dict of edge tuple(node1, node2) object mapped to
     string values describing neighboring edge colors.
    '''

    # edge colors for difference graphs
    if difference_graph:
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

    return edge_color, edge_width, selected_edge_colors

def get_trans_graph_group_stats(node_indices, usages, scalars):
    '''
    Computes and returns all the syllable statistics to display and filter the graph using the GUI.

    Parameters
    ----------
    node_indices (1d list): list of plotted node indices as serialized from a networkX.Digraph.nodes array
    usages (1d list): list of syllable usages corresponding to the node_indices.
    scalars (dict): dict of syllable scalar values

    Returns
    -------
    group_stats (dict): packed dict of syllable scalar strings mapped to
     1d lists of the values corresponding to the node_indices.
     To be reused down the pipeline in plot_interactive_transition_graph().
    '''

    # get usages
    group_usage = [usages[j] for j in node_indices if j in usages]

    # get speeds
    group_speed_2d = [scalars['speeds_2d'][j] for j in node_indices if j in scalars['speeds_2d']]
    group_speed_3d = [scalars['speeds_3d'][j] for j in node_indices if j in scalars['speeds_3d']]

    # get average height
    group_height = [scalars['heights'][j] for j in node_indices if j in scalars['heights']]

    # get mean distances to bucket centers
    group_dist = [scalars['dists'][j] for j in node_indices if j in scalars['dists']]

    group_stats = {
        'usage': np.nan_to_num(group_usage),
        'speed_2d': np.nan_to_num(group_speed_2d),
        'speed_3d': np.nan_to_num(group_speed_3d),
        'height': np.nan_to_num(group_height),
        'dist': np.nan_to_num(group_dist)
    }

    return group_stats

def set_node_colors_and_sizes(graph, usages, node_indices, difference_graph=False):
    '''
    Computes the colors and sizes of all the transition nodes. Each individual group
    will always have edges colored black. If difference_graph is true,
     then the nodes will be colored red if the difference is > 0, and blue otherwise.

    Parameters
    ----------
    graph (nx.DiGraph): networkx graph to set the new node attributes to.
    usages (1d list): list of syllable usages corresponding to the node_indices.
    node_indices (1d list): list of plotted node indices as serialized from a networkX.Digraph.nodes array
    difference_graph (bool): indicates whether to color the edges based on the transition difference between two groups.

    Returns
    -------
    '''

    # node colors for difference graphs
    if difference_graph:
        node_color = {s: 'red' if usages[s] > 0 else 'blue' for s in node_indices}
        node_size = {s: max(15., 10 + abs(usages[s] * 500)) for s in node_indices}
    else:
        node_color = {s: 'red' for s in node_indices}
        node_size = {s: max(15., abs(usages[s] * 500)) for s in node_indices}

    # setting node attributes
    nx.set_node_attributes(graph, node_color, "node_color")
    nx.set_node_attributes(graph, node_size, "node_size")

def get_group_node_syllable_info(syll_info, node_indices):
    '''
    Reads the given syllable information dict in the ordering provided by the node_indices
     previously computed via nx.DiGraph.nodes

    Parameters
    ----------
    syll_info (dict): dict of syllable label information to read.
    node_indices (1d list): ordering of syllables to read from the syll info dict

    Returns
    -------
    labels (1d list): 1d list of syllable labels corresponding to each node index
    descs (1d list): 1d list of syllable descriptions corresponding to each node index
    cm_paths (1d list): 1d list of syllable crowd movie relpaths corresponding to each node index
    '''

    # getting hovertool info
    labels, descs, cm_paths = [], [], []

    for n in node_indices:
        labels.append(syll_info[n]['label'])
        descs.append(syll_info[n]['desc'])
        try:
            cm_paths.append(relpath(syll_info[n]['crowd_movie_path']))
        except ValueError:
            # crowd movie path not found
            cm_paths.append('')

    return labels, descs, cm_paths

def setup_graph_hover_renderers(graph_renderer, group_stats, node_indices):
    '''
    Adds all the information enclosed in the group_stats dict to the currently plotted transition graph such that
     they can be viewed in the hover tool.

    Parameters
    ----------
    graph_renderer (bokeh.plotting.GraphRenderer instance): the canvas of the current group's
     transition graph to add the information to.
    group_stats (dict): packed dict of syllable scalars to add to the hover tool.
    node_indices (1d list): ordering of syllables to read from the syll info dict

    Returns
    -------
    graph_renderer (bokeh.plotting.GraphRenderer instance): updated reference of the GraphRenderer instance.
    '''

    # setting common data source to display via HoverTool
    graph_renderer.node_renderer.data_source.add(node_indices, 'number')
    graph_renderer.node_renderer.data_source.add(group_stats['labels'], 'label')
    graph_renderer.node_renderer.data_source.add(group_stats['descs'], 'desc')
    graph_renderer.node_renderer.data_source.add(group_stats['cm_paths'], 'movies')
    graph_renderer.node_renderer.data_source.add(group_stats['prev_states'], 'prev')
    graph_renderer.node_renderer.data_source.add(group_stats['next_states'], 'next')
    graph_renderer.node_renderer.data_source.add(group_stats['usage'], 'usage')
    graph_renderer.node_renderer.data_source.add(group_stats['speed_2d'], 'speed_2d')
    graph_renderer.node_renderer.data_source.add(group_stats['speed_3d'], 'speed_3d')
    graph_renderer.node_renderer.data_source.add(group_stats['height'], 'height')
    graph_renderer.node_renderer.data_source.add(group_stats['dist'], 'dist_to_center_px')
    graph_renderer.node_renderer.data_source.add(group_stats['incoming_transition_entropy'], 'ent_in')
    graph_renderer.node_renderer.data_source.add(group_stats['outgoing_transition_entropy'], 'ent_out')

    return graph_renderer

def setup_node_and_edge_interactions(graph_renderer, group_stats, scalar_color):
    '''
    Adds the interactive functionality to hovering and tapping on the nodes and edges of each transition graph.

    Parameters
    ----------
    graph_renderer (bokeh.plotting.GraphRenderer instance): the canvas of the current group's
     transition graph to add the information to.
    group_stats (dict): packed dict of syllable scalars to add to the hover tool.
    scalar_color (str): name of scalar to color nodes by.

    Returns
    -------
    graph_renderer (bokeh.plotting.GraphRenderer instance): updated reference of the GraphRenderer instance.
    color_bar (bokeh.models.ColorBar instance): a color bar Bokeh glyph to add to the current graph
     if the scalar colors are not all white.
    '''

    data_dict = {
        '2D velocity': {'key': 'speed_2d', 'values': group_stats['speed_2d']},
        '3D velocity': {'key': 'speed_3d', 'values': group_stats['speed_3d']},
        'Height': {'key': 'height', 'values': group_stats['height']},
        'Distance to Center': {'key': 'dist_to_center_px', 'values': group_stats['dist']},
        'Entropy-In': {'key': 'ent_in', 'values': np.nan_to_num(group_stats['incoming_transition_entropy'])},
        'Entropy-Out': {'key': 'ent_out', 'values': np.nan_to_num(group_stats['outgoing_transition_entropy'])},
    }

    fill_color, empty = set_fill_color(scalar_color, data_dict)

    # Get color bar if node colors are not white
    color_bar = None
    if fill_color != 'white' and not empty:
        color_bar = ColorBar(color_mapper=fill_color['transform'])

    # node interactions
    graph_renderer.node_renderer.glyph = Circle(size='node_size', fill_color=fill_color, line_color='node_color')
    graph_renderer.node_renderer.selection_glyph = Circle(size='node_size', line_color='node_color',
                                                          fill_color=fill_color)
    graph_renderer.node_renderer.nonselection_glyph = Circle(size='node_size', line_color='node_color',
                                                             fill_color=fill_color)
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

    return graph_renderer, color_bar

def set_node_labels(x, y, syllable):
    '''
    Given the x and y coordinates of the nodes in the graph, and the syllable numbers they correspond to,
     the function will create a LabelSet instance to render the syllables numbers on each graphed transition node.

    Parameters
    ----------
    x (1d list): list of x coordinates corresponding to each node in the current graph
    y (1d list): list of y coordinates corresponding to each node in the current graph
    syllable (1d list): list of syllable numbers corresponding to each node in the current graph

    Returns
    -------
    labels (bokeh.models.LabelSet instance): glyph to render on the graph such that the nodes are numbered.
    '''

    # Get fill colors
    text_color = 'black'

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

    return labels

def get_node_labels(plots, graph_renderer, rendered_graphs, graph, node_indices):
    '''
    Will read the nx.DiGraph object to use the return node x,y coordinates and syllable/node numbers
     to then create a LabelSet instance to display each group's corresponding syllable numbers of the correct
      corresponding nodes on each plot/for each group.

    Parameters
    ----------
    plots (1d list): list of plots currently generated in the plot_interactive_transition_graph loop.
    graph_renderer (bokeh.plotting.GraphRenderer instance): the canvas of the current group's
     transition graph to add the information to.
    rendered_graphs (1d list): list of GraphRenderers currently generated in the plot_interactive_transition_graph loop.
    graph (nx.DiGraph): networkx graph to read the node coordinates from.
    node_indices (1d list): ordering of syllables to read from the syll info dict

    Returns
    -------
    labels (bokeh.models.LabelSet instance): glyph to render on the graph such that the nodes are numbered.
    '''

    try:
        # get node positions
        if len(plots) == 0:
            x, y = zip(*graph_renderer.layout_provider.graph_layout.values())
            syllable = list(graph.nodes)
        else:
            new_layout = {k: rendered_graphs[0].layout_provider.graph_layout[k] for k in
                          graph_renderer.layout_provider.graph_layout}
            x, y = zip(*new_layout.values())
            syllable = [a if a in node_indices else '' for a in new_layout]
    except:
        # If the graph has been thresholded such that there are missing syllables, or is empty altogether
        # (with or without thresholding) we remove all the node label coordinates.
        x, y = [], []
        syllable = []

    labels = set_node_labels(x, y, syllable)

    return labels

def get_legend_items(plot, edge_width, group_name, difference_graph=False):
    '''
    Will generate two legend items to describe the main group incoming and outgoing transition colors (main_legend),
     and an info_legend containing references to the displayed node widths
     and their corresponding transition probabilities.

    Parameters
    ----------
    plot (bokeh figure): bokeh generated figure to add legends to.
    edge_width (dict): dict of edge tuple(node1, node2) object mapped to float values describing edge widths.
    group_name (str): name of currently plotted group
    difference_graph (bool): indicator for whether the currently plotted graph is a difference graph between two groups.

    Returns
    -------
    main_legend (bokeh Legend instance): Legend containing colors describing incoming and outgoing transitions.
    info_legend (bokeh Legend instance): Legend containing edge widths corresponding
     to min and max transition probabilities.
    '''

    o_line = plot.line(line_color='orange', line_width=4)
    p_line = plot.line(line_color='purple', line_width=4)
    g_line = plot.line(line_color='green', line_width=4)

    min_tp, max_tp = get_minmax_tp(edge_width, diff=False)

    mink_line = plot.line(line_color='black', line_width=min_tp * 200)
    maxk_line = plot.line(line_color='black', line_width=max_tp * 200)

    group_items = [
        LegendItem(label="Incoming Transition", renderers=[o_line]),
        LegendItem(label="Outgoing Transition", renderers=[p_line]),
        LegendItem(label="Bidirectional Transition", renderers=[g_line]),
    ]

    info_items = [
        LegendItem(label=f"Min P(transition): {min_tp:.4f}", renderers=[mink_line]),
        LegendItem(label=f"Max P(transition): {max_tp:.4f}", renderers=[maxk_line]),
    ]

    if difference_graph:
        diff_main_items, info_items = get_difference_legend_items(plot, edge_width, group_name)
        group_items += diff_main_items

    main_legend = Legend(items=group_items,
                         location='top_left',
                         border_line_color="black",
                         background_fill_color='white',
                         background_fill_alpha=0.7)

    info_legend = Legend(items=info_items,
                         location='bottom_left',
                         border_line_color="black",
                         background_fill_color='white',
                         background_fill_alpha=0.7)

    return main_legend, info_legend

def plot_interactive_transition_graph(graphs, pos, group, group_names, usages,
                                      syll_info, incoming_transition_entropy, outgoing_transition_entropy,
                                      scalars, scalar_color='default', plot_vertically=False, legend_loc='above'):
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

    if plot_vertically:
        legend_loc = 'right'

    rendered_graphs, plots = [], []

    for i, graph in enumerate(graphs):

        # get boolean for whether current plot is a group difference graph
        difference_graph = i >= len(group)

        node_indices = [n for n in graph.nodes if n in usages[i]]

        # initialize bokeh plot with title and joined panning/zooming coordinates
        if len(plots) == 0:
            plot = figure(title=f"{group_names[i]}", x_range=(-1.2, 1.2), y_range=(-1.2, 1.2))
        else:
            # Connecting pan-zoom interaction across plots
            plot = figure(title=f"{group_names[i]}", x_range=plots[0].x_range, y_range=plots[0].y_range)

        # format the plot and set up the tooltips
        format_plot(plot)
        setup_trans_graph_tooltips(plot)

        # compute group stats for each node
        group_scalars = {k: v[i] for k, v in scalars.items()}
        group_stats = get_trans_graph_group_stats(node_indices, usages[i], group_scalars)
        group_stats['incoming_transition_entropy'] = incoming_transition_entropy[i]
        group_stats['outgoing_transition_entropy'] = outgoing_transition_entropy[i]

        # get state neighbor lists for each node
        group_stats['prev_states'], group_stats['next_states'], group_stats['neighbor_edge_colors'] = \
            get_neighbors(graph, node_indices, group_names[i])

        # format graph edge colors and widths
        edge_color, edge_width, selected_edge_colors = \
            format_trans_graph_edges(graph, group_stats['neighbor_edge_colors'], difference_graph)

        # format graph node colors and sizes
        set_node_colors_and_sizes(graph, usages[i], node_indices, difference_graph)

        # create bokeh-fied networkx transition graph
        graph_renderer = from_networkx(graph, pos, scale=1, center=(0, 0))

        # get syllable info for each node
        group_stats['labels'], group_stats['descs'], group_stats['cm_paths'] = \
            get_group_node_syllable_info(syll_info, node_indices)

        # setup hover tool information
        graph_renderer = setup_graph_hover_renderers(graph_renderer, group_stats, node_indices)

        graph_renderer, color_bar = setup_node_and_edge_interactions(graph_renderer, group_stats, scalar_color)

        # added rendered graph to plot
        plot.renderers.append(graph_renderer)

        # get node labels and draw their numbers on each node
        labels = get_node_labels(plots, graph_renderer, rendered_graphs, graph, node_indices)

        # render labels
        plot.renderers.append(labels)

        # get plot legends
        main_legend, info_legend = get_legend_items(plot, edge_width, group_names[i], difference_graph)

        # add legends to plot
        plot.renderers.append(main_legend)
        plot.renderers.append(info_legend)

        # set legend layouts for grid plot
        if not plot_vertically:
            plot.add_layout(main_legend, legend_loc)

        # add color bar to graph
        if color_bar is not None:
            plot.renderers.append(color_bar)

        plots.append(plot)
        rendered_graphs.append(graph_renderer)

    # format all the generated bokeh transition graphs
    ncols = 1
    formatted_plots = list(plots)
    plot_height, plot_width = 550, 550
    if not plot_vertically:
        # Format grid of transition graphs
        ncols = None
        plot_width, plot_height = 550, 675
        formatted_plots = format_graphs(plots, group)

    # Create Bokeh grid plot object
    gp = gridplot(formatted_plots, sizing_mode='scale_both', ncols=ncols, plot_width=plot_width, plot_height=plot_height)
    show(gp)