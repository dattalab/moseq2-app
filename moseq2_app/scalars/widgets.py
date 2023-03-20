"""
Ipywidgets used to facilitate interactive scalar summary viewer tool.
"""
from ipywidgets import VBox
import ipywidgets as widgets
from IPython.display import display, clear_output

class InteractiveScalarWidgets:

    def __init__(self):
        """
        Initialize all the ipywidgets widgets in a new context.
        """

        style = {'description_width': 'initial'}

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.label_layout = widgets.Layout(display='flex-grow', flex_flow='column', align_items='center', width='100%')
        self.layout_hidden = widgets.Layout(visibility='hidden', display='none')
        self.layout_visible = widgets.Layout(visibility='visible', display='inline-flex')
        self.box_layout = widgets.Layout(display='inline-flex',
                                         justify_content='center',
                                         height='100%',
                                         align_items='center')

        # scalar widgets
        self.checked_list = widgets.SelectMultiple(options=[], description='Scalar Columns to Plot', style=style,
                                                   continuous_update=False, disabled=False, layout=self.label_layout)

        self.ui_tools = VBox([self.clear_button, self.checked_list], layout=self.box_layout)

        # initialize event listeners
        self.checked_list.observe(self.on_column_select, names='value')
        self.clear_button.on_click(self.on_clear)

    def on_clear(self, b=None):
        """
        Clear the display and memory.

        Args:
        b (button Event): Ipywidgets.Button click event.

        Returns:
        """

        clear_output()
        del self

    def on_column_select(self, event=None):
        """
        Update the view once the user selects a new set of scalars to plot.

        Args:
        event

        Returns:
        """

        clear_output()
        display(self.ui_tools)
        self.interactive_view()
