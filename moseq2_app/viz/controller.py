import os
import pandas as pd
from glob import glob
from copy import deepcopy
from bokeh.io import show
import ruamel.yaml as yaml
from bokeh.models import Div
import ipywidgets as widgets
from bokeh.layouts import column
from moseq2_viz.util import parse_index
from IPython.display import display, clear_output
from moseq2_extract.io.video import get_video_info
from moseq2_viz.model.util import results_to_dataframe
from moseq2_app.viz.widgets import SyllableLabelerWidgets
from moseq2_viz.helpers.wrappers import make_crowd_movies_wrapper
from moseq2_viz.scalars.util import (scalars_to_dataframe, compute_session_centroid_speeds, compute_mean_syll_scalar)

class SyllableLabeler(SyllableLabelerWidgets):
    '''

    Class that contains functionality for previewing syllable crowd movies and
     user interactions with buttons and menus.

    '''

    def __init__(self, model_fit, index_file, max_sylls, save_path):
        '''
        Initializes class context parameters, reads and creates the syllable information dict.

        Parameters
        ----------
        model_fit (dict): Loaded trained model dict.
        index_file (str): Path to saved index file.
        max_sylls (int): Maximum number of syllables to preview and label.
        save_path (str): Path to save syllable label information dictionary.
        '''

        super().__init__()
        self.save_path = save_path
        self.max_sylls = max_sylls

        self.model_fit = model_fit
        self.sorted_index = parse_index(index_file)[1]

        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                self.syll_info = yaml.safe_load(f)

                if len(self.syll_info.keys()) != max_sylls:
                    self.syll_info = {str(i): {'label': '', 'desc': '', 'crowd_movie_path': '', 'group_info': {}} for i
                                      in range(max_sylls)}

                for i in range(max_sylls):
                    if 'group_info' not in self.syll_info[str(i)].keys():
                        self.syll_info[str(i)]['group_info'] = {}
        else:
            self.syll_info = {str(i): {'label': '', 'desc': '', 'crowd_movie_path': '', 'group_info': {}} for i in
                              range(max_sylls)}
            yml = yaml.YAML()
            yml.indent(mapping=3, offset=2)

            # Write to file
            with open(self.save_path, 'w+') as f:
                yml.dump(self.syll_info, f)

        # Initialize button callbacks
        self.next_button.on_click(self.on_next)
        self.prev_button.on_click(self.on_prev)
        self.set_button.on_click(self.on_set)

        # Syllable Info DataFrame path
        output_dir = os.path.dirname(save_path)
        self.df_output_file = os.path.join(output_dir, 'syll_df.parquet')
        self.label_df_output_file = os.path.join(output_dir, 'label_time_df.parquet')

        self.get_mean_syllable_info()

        self.clear_button.on_click(self.clear_on_click)

    def clear_on_click(self, b):
        '''
        Clears the cell output

        Parameters
        ----------
        b

        Returns
        -------
        '''

        clear_output()

    def on_next(self, event):
        '''
        Callback function to trigger an view update when the user clicks the "Next" button.

        Parameters
        ----------
        event (ipywidgets.ButtonClick): User clicks next button.

        Returns
        -------
        '''

        # Updating dict
        self.syll_info[str(self.syll_select.index)]['label'] = self.lbl_name_input.value
        self.syll_info[str(self.syll_select.index)]['desc'] = self.desc_input.value

        # Handle cycling through syllable labels
        if self.syll_select.index != int(list(self.syll_select.options.keys())[-1]):
            # Updating selection to trigger update
            self.syll_select.index += 1
        else:
            self.syll_select.index = 0

        # Updating input values with current dict entries
        self.lbl_name_input.value = self.syll_info[str(self.syll_select.index)]['label']
        self.desc_input.value = self.syll_info[str(self.syll_select.index)]['desc']

    def on_prev(self, event):
        '''
        Callback function to trigger an view update when the user clicks the "Previous" button.

        Parameters
        ----------
        event (ipywidgets.ButtonClick): User clicks 'previous' button.

        Returns
        -------
        '''

        # Update syllable information dict
        self.syll_info[str(self.syll_select.index)]['label'] = self.lbl_name_input.value
        self.syll_info[str(self.syll_select.index)]['desc'] = self.desc_input.value

        # Handle cycling through syllable labels
        if self.syll_select.index != 0:
            # Updating selection to trigger update
            self.syll_select.index -= 1
        else:
            self.syll_select.index = int(list(self.syll_select.options.keys())[-1])

        # Reloading previously inputted text area string values
        self.lbl_name_input.value = self.syll_info[str(self.syll_select.index)]['label']
        self.desc_input.value = self.syll_info[str(self.syll_select.index)]['desc']

    def on_set(self, event):
        '''
        Callback function to save the dict to syllable information file.

        Parameters
        ----------
        event (ipywidgets.ButtonClick): User clicks the 'Save' button.

        Returns
        -------
        '''

        # Update dict
        self.syll_info[str(self.syll_select.index)]['label'] = self.lbl_name_input.value
        self.syll_info[str(self.syll_select.index)]['desc'] = self.desc_input.value

        yml = yaml.YAML()
        yml.indent(mapping=3, offset=2)

        tmp = deepcopy(self.syll_info)
        for syll in range(self.max_sylls):
            del tmp[str(syll)]['group_info']

        # Write to file
        with open(self.save_path, 'w+') as f:
            yml.dump(tmp, f)

        # Update button style
        self.set_button.button_type = 'success'

    def get_mean_group_dict(self, group_df):
        '''

        Parameters
        ----------
        group_df

        Returns
        -------

        '''

        # Get array of grouped syllable info
        group_dicts = []
        for group in self.groups:
            group_dict = {
                group: group_df[group_df['group'] == group].drop('group', axis=1).reset_index(drop=True).to_dict()}
            group_dicts.append(group_dict)

        self.group_syll_info = deepcopy(self.syll_info)
        # Update syllable info dict
        for gd in group_dicts:
            group_name = list(gd.keys())[0]
            for syll in range(self.max_sylls):
                self.group_syll_info[str(syll)]['group_info'][group_name] = {
                    'usage': gd[group_name]['usage'][syll],
                    'centroid speed (mm/s)': gd[group_name]['speed'][syll],
                    '2D velocity (mm/s)': gd[group_name]['velocity_2d_mm'][syll],
                    '3D velocity (mm/s)': gd[group_name]['velocity_3d_mm'][syll],
                    'height (mm)': gd[group_name]['height_ave_mm'][syll],
                    'norm. dist_to_center': gd[group_name]['dist_to_center'][syll],
                }

    def get_mean_syllable_info(self):
        '''
        Populates syllable information dict with usage and scalar information.

        Returns
        -------
        '''

        # Load scalar Dataframe to compute syllable speeds
        scalar_df = scalars_to_dataframe(self.sorted_index)
        scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)

        if not os.path.exists(self.df_output_file):
            # Compute a syllable summary Dataframe containing usage-based
            # sorted/relabeled syllable usage and duration information from [0, max_syllable) inclusive
            df, label_df = results_to_dataframe(self.model_fit, self.sorted_index, count='usage',
                                                max_syllable=self.max_sylls, sort=True, compute_labels=True)

            scalars = ['centroid_speed_mm', 'velocity_2d_mm', 'velocity_3d_mm', 'height_ave_mm', 'dist_to_center_px']
            for scalar in scalars:
                df = compute_mean_syll_scalar(df, scalar_df, label_df, scalar=scalar, max_sylls=self.max_sylls)

            print('Writing main syllable info to parquet')
            df.to_parquet(self.df_output_file, engine='fastparquet', compression='gzip')
            tmp = label_df.copy()
            tmp.columns = tmp.columns.astype(str)
            print('Writing session syllable labels to parquet')
            tmp.to_parquet(self.label_df_output_file, engine='fastparquet', compression='gzip')
        else:
            print('Loading parquet files')
            df = pd.read_parquet(self.df_output_file, engine='fastparquet')
            label_df = pd.read_parquet(self.label_df_output_file, engine='fastparquet')
            label_df.columns = label_df.columns.astype(int)

        # Get all unique groups in df
        self.groups = df.group.unique()

        # Get grouped DataFrame
        group_df = df.groupby(['group', 'syllable'], as_index=False).mean()

        # Get self.group_info
        self.get_mean_group_dict(group_df)

    def set_group_info_widgets(self, group_info):
        '''
        Display function that reads the syllable information into a pandas DataFrame, converts it
        to an HTML table and displays it in a Bokeh Div facilitated via the Output() widget.

        Parameters
        ----------
        group_info (dict): Dictionary of grouped current syllable information

        Returns
        -------
        '''

        output_table = Div(text=pd.DataFrame(group_info).to_html())

        ipy_output = widgets.Output()
        with ipy_output:
            show(output_table)

        self.info_boxes.children = [self.syll_info_lbl, ipy_output, ]

    def interactive_syllable_labeler(self, syllables):
        '''
        Helper function that facilitates the interactive view. Function will create a Bokeh Div object
        that will display the current video path.

        Parameters
        ----------
        syllables (int or ipywidgets.DropDownMenu): Current syllable to label

        Returns
        -------
        '''

        self.set_button.button_type = 'primary'

        # Set current widget values
        if len(syllables['label']) > 0:
            self.lbl_name_input.value = syllables['label']

        if len(syllables['desc']) > 0:
            self.desc_input.value = syllables['desc']

        # Update label
        self.cm_lbl.text = f'Crowd Movie {self.syll_select.index + 1}/{len(self.syll_select.options)}'

        # Update scalar values
        self.set_group_info_widgets(self.group_syll_info[str(self.syll_select.index)]['group_info'])

        # Get current movie path
        cm_path = syllables['crowd_movie_path']

        video_dims = get_video_info(cm_path)['dims']

        # Create syllable crowd movie HTML div to embed
        video_div = f'''
                        <h2>{self.syll_select.index}: {syllables['label']}</h2>
                        <video
                            src="{cm_path}"; alt="{cm_path}"; height="{video_dims[1]}"; width="{video_dims[0]}"; preload="true";
                            style="float: left; type: "video/mp4"; margin: 0px 10px 10px 0px;
                            border="2"; autoplay controls loop>
                        </video>
                    '''

        # Create embedded HTML Div and view layout
        div = Div(text=video_div, style={'width': '100%'})
        layout = column([div, self.cm_lbl])

        # Insert Bokeh div into ipywidgets Output widget to display
        vid_out = widgets.Output(layout=widgets.Layout(display='inline-block'))
        with vid_out:
            show(layout)

        # Create grid layout to display all the widgets
        grid = widgets.AppLayout(left_sidebar=vid_out,
                                 right_sidebar=self.data_box,
                                 pane_widths=[3, 0, 2.5])

        # Display all widgets
        display(grid, self.button_box)

    def set_default_cm_parameters(self, config_data):

        config_data['separate_by'] = None
        config_data['specific_syllable'] = None
        config_data['max_syllable'] = self.max_sylls
        config_data['max_examples'] = 20

        config_data['gaussfilter_space'] = [0, 0]
        config_data['medfilter_space'] = [0]
        config_data['sort'] = True
        config_data['dur_clip'] = 300
        config_data['raw_size'] = (512, 424)
        config_data['scale'] = 1
        config_data['legacy_jitter_fix'] = False
        config_data['cmap'] = 'jet'
        config_data['count'] = 'usage'

        return config_data

    def get_crowd_movie_paths(self, index_path, model_path, config_data, crowd_movie_dir):
        '''
        Populates the syllable information dict with the respective crowd movie paths.

        Parameters
        ----------
        crowd_movie_dir (str): Path to directory containing all the generated crowd movies

        Returns
        -------
        '''

        if not os.path.exists(crowd_movie_dir):
            print('Crowd movies not found. Generating movies...')
            config_data = self.set_default_cm_parameters(config_data)

            # Generate movies if directory does not exist
            _ = make_crowd_movies_wrapper(index_path, model_path, config_data, crowd_movie_dir)

        # Get movie paths
        crowd_movie_paths = [f for f in glob(crowd_movie_dir + '*') if '.mp4' in f]

        if len(crowd_movie_paths) < self.max_sylls:
            print('Crowd movie list is incomplete. Generating movies...')
            config_data = self.set_default_cm_parameters(config_data)

            # Generate movies if directory does not exist
            _ = make_crowd_movies_wrapper(index_path, model_path, config_data, crowd_movie_dir)

        crowd_movie_paths = [f for f in glob(crowd_movie_dir + '*') if '.mp4' in f]

        for cm in crowd_movie_paths:
            # Parse paths to get corresponding syllable number
            syll_num = cm.split('sorted-id-')[1].split()[0]
            if syll_num in self.syll_info.keys():
                self.syll_info[syll_num]['crowd_movie_path'] = cm