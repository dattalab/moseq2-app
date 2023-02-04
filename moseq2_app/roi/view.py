"""
Interactive ROI/Extraction Bokeh visualization functions.
"""
import os
import io
import base64
import shutil
import numpy as np
import ipywidgets as widgets
from bokeh.models import Div
from bokeh.layouts import gridplot
from IPython.display import display
from bokeh.plotting import figure, show
from os.path import dirname, join, relpath, exists
from moseq2_extract.io.video import get_video_info


def show_extraction(input_file, video_file):
    """
    display manually triggered extraction.s

    Args:
    input_file (str): session name to display.
    video_file (str): path to video to display

    Returns:
    output (ipywidgets.Output widget): HTML canvas where the video is being displayed.
    """

    # Copy generated movie to temporary directory
    vid_dir = dirname(video_file)
    tmp_path = join(vid_dir, 'tmp', f'{np.random.randint(0, 99999)}_{os.path.basename(video_file)}')
    tmp_dirname = dirname(tmp_path)

    if not exists(tmp_dirname):
        os.makedirs(tmp_dirname)

    if video_file != tmp_path:
        shutil.copy2(video_file, tmp_path)

    video_dims = get_video_info(tmp_path)['dims']

    # Open videos in encoded urls
    # Implementation from: https://github.com/jupyter/notebook/issues/1024#issuecomment-338664139
    vid = io.open(tmp_path, 'r+b').read()
    encoded = base64.b64encode(vid)
    tmp_path = encoded.decode('ascii')

    video_div = f"""
                    <h2>{input_file}</h2>
                    <video
                        src="data:video/mp4;base64, {tmp_path}"; alt="data:video/mp4;base64, {tmp_path}"; 
                        height="{video_dims[1]}"; width="{video_dims[0]}"; preload="auto";
                        style="float: center; type: "video/mp4"; margin: 0px 10px 10px 0px;
                        border="2"; autoplay controls loop>
                    </video>
                """

    div = Div(text=video_div, style={'width': '100%', 'align-items': 'center', 'display': 'contents'})

    output = widgets.Output(layout=widgets.Layout(align_items='center', display='inline-block',
                                                  height='100%', width='100%'))
    with output:
        show(div)

    display(output)

    return output

def bokeh_plot_helper(bk_fig, image):
    """
    create the Bokeh image gylphs in the created canvases/figures.

    Args:
    bk_fig (Bokeh figure): figure canvas to draw image/glyph on
    image (2D np.array): image to draw.
    """

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
