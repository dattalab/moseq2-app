'''

This module contains the widget components that comprise the group setting table functionality.
These widgets will work in tandem with the qgrid functionality to facilitate the real time updates.

'''

import qgrid
import pandas as pd
import ruamel.yaml as yaml
import ipywidgets as widgets
from IPython.display import clear_output
from moseq2_app.util import index_to_dataframe

class GroupSettingWidgets:

    def __init__(self, index_filepath):

        self.index_filepath = index_filepath
        style = {'description_width': 'initial', 'display': 'flex-grow', 'align_items': 'stretch'}

        self.col_opts = {
            'editable': False,
            'toolTip': "Not editable"
        }

        self.col_defs = {
            'group': {
                'editable': True,
                'toolTip': 'editable'
            }
        }

        self.clear_button = widgets.Button(description='Clear Output', disabled=False, tooltip='Close Cell Output')

        self.group_input = widgets.Text(value='', placeholder='Enter Group Name to Set', style=style,
                                        description='Desired Group Name', continuous_update=False, disabled=False)
        self.save_button = widgets.Button(description='Set Group', style=style,
                                          disabled=False, tooltip='Set Group')
        self.update_index_button = widgets.Button(description='Update Index File', style=style,
                                                  disabled=False, tooltip='Save Parameters')

        self.group_set = widgets.HBox([self.group_input, self.save_button, self.update_index_button])

        self.index_dict, self.df = index_to_dataframe(self.index_filepath)
        self.qgrid_widget = qgrid.show_grid(self.df[['SessionName', 'SubjectName', 'group', 'uuid']],
                                            column_options=self.col_opts,
                                            column_definitions=self.col_defs,
                                            show_toolbar=False)

        qgrid.set_grid_option('forceFitColumns', False)
        qgrid.set_grid_option('enableColumnReorder', True)
        qgrid.set_grid_option('highlightSelectedRow', True)
        qgrid.set_grid_option('highlightSelectedCell', False)

    def update_table(self, b=None):
        '''
        Updates table upon "Set Button" click

        Parameters
        ----------
        b (button click)

        Returns
        -------
        '''

        self.update_index_button.button_style = 'info'
        self.update_index_button.icon = 'none'

        selected_rows = self.qgrid_widget.get_selected_df()
        x = selected_rows.index

        for i in x:
            self.qgrid_widget.edit_cell(i, 'group', self.index_grid.group_input.value)

    def update_clicked(self, b=None):
        '''
        Updates the index file with the current table state upon Save button click.

        Parameters
        ----------
        b (button click)

        Returns
        -------
        '''

        files = self.index_dict['files']
        meta = [f['metadata'] for f in files]
        meta_cols = pd.DataFrame(meta).columns

        latest_df = self.qgrid_widget.get_changed_df()
        self.df.update(latest_df)

        updated_index = {'files': list(self.df.drop(meta_cols, axis=1).to_dict(orient='index').values()),
                         'pca_path': self.index_dict['pca_path']}

        with open(self.index_filepath, 'w+') as f:
            yaml.safe_dump(updated_index, f)

        self.update_index_button.button_style = 'success'
        self.update_index_button.icon = 'check'

    def clear_clicked(self, b=None):
        # Clear the display
        clear_output()