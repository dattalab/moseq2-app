'''

Interactive Scalar Summary Viewing tool. This module contains the functionality to facilitate selecting
 scalar columns to plot the mean and standard deviations for all detected groups.

'''
import plotly.express as px
import plotly.graph_objects as go
from moseq2_viz.util import parse_index
from plotly.subplots import make_subplots
from moseq2_viz.scalars.util import scalars_to_dataframe
from moseq2_app.scalars.widgets import InteractiveScalarWidgets

class InteractiveScalarViewer(InteractiveScalarWidgets):

    def __init__(self, index_filepath):
        '''
        Initialization function that will compute the scalar dataframe, the mean and standard deviation
         in preparation to plot the default interactive plotly scalar summary.

        Parameters
        ----------
        index_filepath (str): Path to index file
        '''

        super().__init__()

        self.index_filepath = index_filepath

        # get scalar dataframe
        _, self.sorted_index = parse_index(index_filepath)
        self.scalar_df = scalars_to_dataframe(self.sorted_index)

        self.mean_df = self.scalar_df.groupby(['uuid', 'SessionName', 'SubjectName', 'group'], as_index=False).mean()
        self.std_df = self.scalar_df.groupby(['uuid', 'SessionName', 'SubjectName', 'group'],
                                        as_index=True).std().reset_index()
        self.colors = px.colors.qualitative.Alphabet

        # populate column selector
        self.checked_list.options = [
            'area_mm', 'height_ave_mm', 'length_mm', 'velocity_2d_mm', 
            'velocity_3d_mm', 'width_mm', 'dist_to_center_px']

        # set default values
        self.checked_list.value = ['area_mm', 'velocity_2d_mm']

    def make_graphs(self):
        '''
        Creates a 2 column plotly figure where the left column corresponds to the mean of the selected scalar,
         and the right column corresponds to the standard deviation of the scalars.

         Each selected scalar will be plotted in a new row containing violin + swarm plots for all the groups
          found in the scalar dataframe. Users can click on the legend items to hide selected groups from the display.

        Returns
        -------
        '''

        selected_cols = self.checked_list.value
        unique_groups = self.scalar_df.group.unique()

        self.fig = make_subplots(rows=len(selected_cols), cols=2)

        for j, c in enumerate(selected_cols):
            j = j + 1
            for i, g in enumerate(unique_groups):
                y = self.mean_df[self.mean_df['group'] == g][c]
                std_y = self.std_df[self.std_df['group'] == g][c]

                session_name = self.mean_df[self.mean_df['group'] == g]['SessionName']
                subject_name = self.mean_df[self.mean_df['group'] == g]['SubjectName']
                uuid = self.mean_df[self.mean_df['group'] == g]['uuid']

                texts = [f'SessionName: {sn}<br>SubjectName: {sj}<br>uuid: {u}' for sn, sj, u in
                         zip(session_name, subject_name, uuid)]

                show = True
                if j > 1:
                    show = False

                v1 = go.Violin(y=y,
                               name=g,
                               jitter=0.5,
                               line_color=self.colors[i],
                               marker=dict(size=5),
                               line=dict(width=1),
                               points='all',
                               text=texts,
                               legendgroup=g,
                               showlegend=show,
                               hovertemplate=f'Mean {c}: ' + "%{y}<br>%{text}"
                               )

                self.fig.add_trace(v1, row=j, col=1)

                self.fig.update_yaxes(title_text=f"{c}", row=j, col=1)

                v2 = go.Violin(y=std_y,
                               name=g,
                               jitter=0.5,
                               line_color=self.colors[i],
                               marker=dict(size=5),
                               line=dict(width=1),
                               points='all',
                               text=texts,
                               legendgroup=g,
                               showlegend=False,
                               hovertemplate=f'STD {c}: ' + "%{y}<br>%{text}"
                               )

                self.fig.add_trace(v2, row=j, col=2)

        self.fig.update_xaxes(title_text=f"Mean", row=len(selected_cols), col=1)
        self.fig.update_xaxes(title_text=f"STD", row=len(selected_cols), col=2)
        self.fig.update_xaxes(tickangle=45)

        self.fig.update_layout(height=300*len(selected_cols), width=1000, title_text="Scalar Summary")
        self.fig.update_traces(box_visible=True, meanline_visible=True)

        return self.fig

    def interactive_view(self):
        '''
        Displays the interactive plotly graph. Regularly called from self.on_column_select().

        Returns
        -------
        '''

        fig = self.make_graphs()
        fig.show()