'''
Interactive ROI/Extraction Bokeh visualization functions.
'''
import os
import shutil
import numpy as np
import ipywidgets as widgets
from bokeh.models import Div
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from IPython.display import display, clear_output
from os.path import dirname, join, relpath, exists
from moseq2_extract.io.video import get_video_info


def show_extraction(input_file, video_file, main_output=None):
    '''

    Visualization helper function to display manually triggered extraction.
    Function will facilitate visualization through creating a HTML div to display
    in a jupyter notebook or web page.

    Parameters
    ----------
    input_file (str): session name to display.
    video_file (str): path to video to display

    Returns
    -------
    output (ipywidgets.Output widget): HTML canvas where the video is being displayed.
    '''

    # Copy generated movie to temporary directory
    vid_dir = dirname(video_file)
    tmp_path = join(vid_dir, 'tmp', f'{np.random.randint(0, 99999)}_{os.path.basename(video_file)}')
    tmp_dirname = dirname(tmp_path)

    if not exists(tmp_dirname):
        os.makedirs(tmp_dirname)

    if video_file != tmp_path:
        shutil.copy2(video_file, tmp_path)

    video_dims = get_video_info(tmp_path)['dims']

    video_div = f'''
                    <h2>{input_file}</h2>
                    <video
                        src="{relpath(tmp_path)}"; alt="{tmp_path}"; 
                        height="{video_dims[1]}"; width="{video_dims[0]}"; preload="auto";
                        style="float: center; type: "video/mp4"; margin: 0px 10px 10px 0px;
                        border="2"; autoplay controls loop>
                    </video>
                '''

    div = Div(text=video_div, style={'width': '100%', 'align-items': 'center', 'display': 'contents'})

    if main_output is None:
        main_output = widgets.Output(layout=widgets.Layout(align_items='center', display='inline-block',
                                                  height='100%', width='100%'))
    with main_output:
        clear_output(wait=True)
        show(div)

    return main_output

def bokeh_plot_helper(bk_fig, image):
    '''

    Helper function that creates the Bokeh image gylphs in the
    created canvases/figures.

    Parameters
    ----------
    bk_fig (Bokeh figure): figure canvas to draw image/glyph on
    image (2D np.array): image to draw.

    Returns
    -------
    '''

    bk_fig.x_range.range_padding = bk_fig.y_range.range_padding = 0
    if isinstance(image, dict):
        bk_fig.image(source=image,
                     image='image',
                     x='x',
                     y='y',
                     dw='dw',
                     dh='dh',
                     palette="Viridis256")
    else:
        bk_fig.image(image=[image],
                     x=0,
                     y=0,
                     dw=image.shape[1],
                     dh=image.shape[0],
                     palette="Viridis256")


def plot_roi_results(sessionName, bground_im, roi, overlay, filtered_frames, depth_frames, fn, main_out=None):
    '''
    Main ROI plotting function that uses Bokeh to facilitate 3 interactive plots.
    Plots the background image, and an axis-connected plot of the ROI,
    and an independent plot of the thresholded background subracted segmented image.

    Parameters
    ----------
    sessionName (str): Name of session currently being plotted
    bground_im (2D np.array): Computed session background
    roi (2D np.array): Computed ROI based on given depth ranges
    overlay (2D np.array):
    filtered_frames (2D np.array):
    depth_frames (2D np.array):
    fn (int or ipywidget IntSlider): Current frame number to display

    Returns
    -------
    '''

    # set bokeh tools
    tools = 'pan, box_zoom, wheel_zoom, hover, reset'

    # Plot Background
    bg_fig = figure(title=f"{sessionName} - Background",
                    tools=tools,
                    tooltips=[("(x,y)", "($x{0.1f}, $y{0.1f})"), ("depth", "@image"), ('roi', '@roi')],
                    output_backend="webgl")

    data = dict(image=[bground_im],
                roi=[roi],
                x=[0],
                y=[0],
                dw=[bground_im.shape[1]],
                dh=[bground_im.shape[0]])

    bokeh_plot_helper(bg_fig, data)

    # plot overlayed roi
    overlay_fig = figure(title="Overlayed ROI",
                         x_range=bg_fig.x_range,
                         y_range=bg_fig.y_range,
                         tools=tools,
                         tooltips=[("(x,y)", "($x{0}, $y{0})"), ("depth", "@image")],
                         output_backend="webgl")

    bokeh_plot_helper(overlay_fig, overlay)

    # plot segmented frame
    segmented_fig = figure(title=f"Segmented Frame #{fn}",
                           tools=tools,
                           tooltips=[("(x,y)", "($x{0}, $y{0})"), ("height", "@image")],
                           output_backend="webgl")

    bokeh_plot_helper(segmented_fig, filtered_frames)

    # plot crop rotated frame
    cropped_fig = figure(title=f"Crop-Rotated Frame #{fn}",
                         tools=tools,
                         tooltips=[("(x,y)", "($x{0}, $y{0})"), ("height", "@image")],
                         output_backend="webgl")

    bokeh_plot_helper(cropped_fig, depth_frames)

    # Create 2x2 grid plot
    gp = gridplot([[bg_fig, overlay_fig],
                   [segmented_fig, cropped_fig]],
                  plot_width=350, plot_height=350)

    if main_out is None:
        # Create Output widget object to center grid plot in view
        main_out = widgets.Output(layout=widgets.Layout(align_items='center'))

    with main_out:
        clear_output(wait=True)
        show(gp)

    return main_out
