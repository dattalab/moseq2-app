'''

Ipywidgets used to facilitate interactive scalar summary viewer tool.

'''
import ipywidgets as widgets
from ipywidgets import VBox

class InteractiveScalarWidgets:
    '''
    Class that contains Ipywidget widgets and layouts to facilitate interactive ROI finding functionality.
    This class is extended by the controller class InteractiveFindRoi.
    '''

    def __init__(self):
        '''
        Initializing all the ipywidgets widgets in a new context.
        '''

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