'''
Interactive ROI/Extraction Bokeh visualization functions.
'''
import os
import shutil
import numpy as np
import ipywidgets as widgets
from bokeh.models import Div
from bokeh.layouts import gridplot
from IPython.display import display
from bokeh.plotting import figure, show
from os.path import dirname, join, relpath, exists
from moseq2_extract.io.video import get_video_info
from IPython.display import HTML
try:
    from kora.drive import upload_public
except (ImportError, ModuleNotFoundError) as error:
    print(error)


def show_extraction(input_file, video_file):
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
    '''

    # Copy generated movie to temporary directory
    vid_dir = dirname(video_file)
    tmp_path = join(vid_dir, 'tmp', f'{np.random.randint(0, 99999)}_{os.path.basename(video_file)}')
    tmp_dirname = dirname(tmp_path)

    url = upload_public(input_file)

    if not exists(tmp_dirname):
        os.makedirs(tmp_dirname)

    if video_file != tmp_path:
        shutil.copy2(video_file, tmp_path)

    video_dims = get_video_info(tmp_path)['dims']

    video_div = f'''
                    <h2>{input_file}</h2>
                    <link rel="stylesheet" href="/nbextensions/google.colab/tabbar.css">
                    <video
                        src="{url}"; alt="{url}"; id="preview";                     
                        height="{video_dims[1]}"; width="{video_dims[0]}"; preload="auto";
                        style="float: center; type: "video/mp4"; margin: 0px 10px 10px 0px;
                        border="2"; autoplay controls loop>
                    </video>
                '''

    div = Div(text=video_div, style={'width': '100%', 'align-items': 'center', 'display': 'contents'})

    output = widgets.Output(layout=widgets.Layout(align_items='center', display='inline-block',
                                                  height='100%', width='100%'))
    with output:
        show(div)

    display(output)

    return output


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
