"""
Interactive ROI detection and extraction preview functionalities.
"""

import io
import base64
from bokeh.io import show
import ipywidgets as widgets
from bokeh.models import Div, CustomJS, Slider
from IPython.display import clear_output
from moseq2_app.gui.progress import get_session_paths
from moseq2_extract.io.video import get_video_info


class InteractiveExtractionViewer:

    def __init__(self, data_path, flipped=False):
        """
        initialize the extraction viewer widget.

        Args:
        data_path (str): Path to base directory containing all sessions to test
        flipped (bool): indicates whether to show corrected flip videos
        """

        self.sess_select = widgets.Dropdown(options=get_session_paths(data_path, extracted=True, flipped=flipped),
                                            description='Session:', disabled=False, continuous_update=True)

        self.clear_button = widgets.Button(
            description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.clear_button.on_click(self.clear_on_click)

    def clear_on_click(self, b=None):
        """
        Clear the cell output

        Args:
        b (button click)
        """

        clear_output()

    def get_extraction(self, input_file):
        """
        Returns a div containing a video object to display.

        Args:
        input_file (str): Path to session extraction video to view.
        """

        video_dims = get_video_info(input_file)['dims']
        # input_file goes through encode and decode so it won't carry semantic meanings anymore
        file_name = input_file

        # Open videos in encoded urls
        # Implementation from: https://github.com/jupyter/notebook/issues/1024#issuecomment-338664139
        vid = io.open(input_file, 'r+b').read()
        encoded = base64.b64encode(vid)
        input_file = encoded.decode('ascii')

        video_div = f"""
                        <h2>{file_name}</h2>
                        <video
                            src="data:video/mp4;base64, {input_file}"; alt="data:video/mp4;base64, {input_file}"; id="preview";
                            height="{video_dims[1]}"; width="{video_dims[0]}"; preload="auto";
                            style="float: center; type: "video/mp4"; margin: 0px 10px 10px 0px;
                            border="2"; autoplay controls loop>
                        </video>
                        <script>
                            document.querySelector('video').playbackRate = 0.1;
                        </script>
                     """

        div = Div(text=video_div, style={
                  'width': '100%', 'align-items': 'center', 'display': 'contents'})

        slider = Slider(start=0, end=4, value=1, step=0.1,
                        format="0[.]00", title=f"Playback Speed")

        callback = CustomJS(
            args=dict(slider=slider),
            code="""
                    document.querySelector('video').playbackRate = slider.value;
                 """
        )

        slider.js_on_change('value', callback)
        show(slider)
        show(div)
