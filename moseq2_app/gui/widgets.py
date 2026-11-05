"""
This module contains the widget components that comprise the group setting table functionality.
"""

# qgrid stopped at ipywidgets 7: it registers its widget with the zero-argument
# `@widgets.register()` form, which raises on import under 8. qgridnext is a
# fork of it that carries the same API, so the calls below are unchanged.
import qgridnext as qgrid
import ipywidgets as widgets
from moseq2_common.bookkeeping import session_index
from moseq2_common.project import Project, read_groups, write_groups
from moseq2_app.util import ClearOutputMixin, index_to_dataframe

class GroupSettingWidgets(ClearOutputMixin):

    def __init__(self, project_folder):
        """
        Initialize all the Group Setting widgets: a table of the project's sessions, their metadata and groups,
        displayed using QGrid. Saving writes each session's group into the project's config.yaml (``groups``).

        Args:
        project_folder (str): the project folder.
        """

        self.project = Project.at(project_folder)
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
                                        description='New Group Name', continuous_update=False, disabled=False)
        self.save_button = widgets.Button(description='Set Group Name', style=style,
                                          disabled=False, tooltip='Set Group')
        self.save_groups_button = widgets.Button(description='Save Groups', style=style,
                                                 disabled=False, tooltip="Save the groups in the project's config.yaml")

        self.group_set = widgets.HBox([self.group_input, self.save_button, self.save_groups_button])

        self.index_dict, self.df = index_to_dataframe(session_index(self.project))
        self.qgrid_widget = qgrid.show_grid(self.df[['SessionName', 'SubjectName', 'group', 'uuid', 'filename']],
                                            column_options=self.col_opts,
                                            column_definitions=self.col_defs,
                                            show_toolbar=False)

        qgrid.set_grid_option('forceFitColumns', False)
        qgrid.set_grid_option('enableColumnReorder', True)
        qgrid.set_grid_option('highlightSelectedRow', True)
        qgrid.set_grid_option('highlightSelectedCell', False)

        # Add callback functions
        self.clear_button.on_click(self.clear_on_click)
        self.save_groups_button.on_click(self.update_clicked)
        self.save_button.on_click(self.update_table)

    def update_table(self, b=None):
        """
        Update table upon "Set Button" click

        Args:
        b (button click)
        """

        self.save_groups_button.button_style = 'info'
        self.save_groups_button.icon = 'none'

        selected_rows = self.qgrid_widget.get_selected_df()
        x = selected_rows.index

        for i in x:
            self.qgrid_widget.edit_cell(i, 'group', self.group_input.value)

    def update_clicked(self, b=None):
        """
        Save each session's group, as the table shows it, in the project's config.yaml upon Save button click.

        Args:
        b (button click)
        """

        latest_df = self.qgrid_widget.get_changed_df()
        self.df = self.df.drop('filename', axis=1, errors='ignore')
        self.df.update(latest_df)

        groups = {**read_groups(self.project), **dict(zip(self.df['key'], self.df['group']))}
        write_groups(self.project, groups)

        self.save_groups_button.button_style = 'success'
        self.save_groups_button.icon = 'check'
