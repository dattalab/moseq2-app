'''

Main syllable crowd movie viewing, comparing, and labeling functionality.

'''

import re
import os
import shutil
import numpy as np
import pandas as pd
from glob import glob
from copy import deepcopy
from bokeh.io import show
import ruamel.yaml as yaml
import ipywidgets as widgets
from os.path import basename
from bokeh.models import Div
from bokeh.layouts import column
from bokeh.plotting import figure
from moseq2_viz.util import get_sorted_index
from moseq2_extract.util import read_yaml
from IPython.display import display, clear_output
from moseq2_extract.io.video import get_video_info
from moseq2_app.viz.view import display_crowd_movies
from moseq2_app.util import merge_labels_with_scalars
from moseq2_viz.model.util import parse_model_results
from moseq2_app.viz.widgets import SyllableLabelerWidgets, CrowdMovieCompareWidgets
from moseq2_viz.helpers.wrappers import make_crowd_movies_wrapper, init_wrapper_function
from moseq2_viz.scalars.util import (scalars_to_dataframe, compute_syllable_position_heatmaps, get_syllable_pdfs)

yml = yaml.YAML()
yml.indent(mapping=3, offset=2)

class SyllableLabeler(SyllableLabelerWidgets):
    '''

    Class that contains functionality for previewing syllable crowd movies and
     user interactions with buttons and menus.

    '''

    def __init__(self, model_fit, model_path, index_file, max_sylls, save_path):
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
        self.model_path = model_path
        self.sorted_index = get_sorted_index(index_file)

        # Syllable Info DataFrame path
        output_dir = os.path.dirname(save_path)
        self.df_output_file = os.path.join(output_dir, 'syll_df.parquet')
        self.scalar_df_output = os.path.join(output_dir, 'moseq_dataframe.parquet')

        index_uuids = sorted(self.sorted_index['files'])
        model_uuids = sorted(set(self.model_fit['metadata']['uuids']))

        if index_uuids != model_uuids:
            print('Error: Index file UUIDs do not match model UUIDs.')

        if os.path.exists(save_path):
            self.syll_info = read_yaml(save_path)
            if len(self.syll_info) != max_sylls:
                # Delete previously saved parquet
                if os.path.exists(self.df_output_file):
                    os.remove(self.df_output_file)

                self.syll_info = {i: {'label': '', 'desc': '', 'crowd_movie_path': '', 'group_info': {}} for i
                                    in range(max_sylls)}

            for i in range(max_sylls):
                if 'group_info' not in self.syll_info[i]:
                    self.syll_info[i]['group_info'] = {}
        else:
            # Delete previously saved parquet
            if os.path.exists(self.df_output_file):
                os.remove(self.df_output_file)

            self.syll_info = {i: {'label': '', 'desc': '', 'crowd_movie_path': '', 'group_info': {}} for i in
                              range(max_sylls)}

            # Write to file
            with open(self.save_path, 'w+') as f:
                yml.dump(self.syll_info, f)

        # Initialize button callbacks
        self.next_button.on_click(self.on_next)
        self.prev_button.on_click(self.on_prev)
        self.set_button.on_click(self.on_set)

        self.get_mean_syllable_info()

        self.clear_button.on_click(self.clear_on_click)

    def clear_on_click(self, b):
        '''
        Clears the cell output

        Parameters
        ----------
        b (button click)

        Returns
        -------
        '''

        clear_output()

    def write_syll_info(self):
        '''
        Writes current syllable info data to a YAML file.

        Returns
        -------
        '''

        tmp = deepcopy(self.syll_info)
        for syll in range(self.max_sylls):
            tmp[syll].pop('group_info', None)

        # Write to file
        with open(self.save_path, 'w+') as f:
            yml.dump(tmp, f)

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
        self.syll_info[self.syll_select.index]['label'] = self.lbl_name_input.value
        self.syll_info[self.syll_select.index]['desc'] = self.desc_input.value

        # Handle cycling through syllable labels
        if self.syll_select.index < len(self.syll_select.options) - 1:
            # Updating selection to trigger update
            self.syll_select.index += 1
        else:
            self.syll_select.index = 0

        # Updating input values with current dict entries
        self.lbl_name_input.value = self.syll_info[self.syll_select.index]['label']
        self.desc_input.value = self.syll_info[self.syll_select.index]['desc']

        self.write_syll_info()

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
        self.syll_info[self.syll_select.index]['label'] = self.lbl_name_input.value
        self.syll_info[self.syll_select.index]['desc'] = self.desc_input.value

        # Handle cycling through syllable labels
        if self.syll_select.index != 0:
            # Updating selection to trigger update
            self.syll_select.index -= 1
        else:
            self.syll_select.index = len(self.syll_select.options) - 1

        # Reloading previously inputted text area string values
        self.lbl_name_input.value = self.syll_info[self.syll_select.index]['label']
        self.desc_input.value = self.syll_info[self.syll_select.index]['desc']

        self.write_syll_info()

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
        self.syll_info[self.syll_select.index]['label'] = self.lbl_name_input.value
        self.syll_info[self.syll_select.index]['desc'] = self.desc_input.value

        self.write_syll_info()

        # Update button style
        self.set_button.button_style = 'success'

    def get_mean_group_dict(self, group_df):
        '''
        Creates a dict object to convert to a displayed table containing syllable scalars.

        Parameters
        ----------
        group_df (pd.DataFrame): DataFrame containing mean syllable scalar data for each session and their groups

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
            group_name = list(gd)[0]
            for syll in range(self.max_sylls):
                self.group_syll_info[syll]['group_info'][group_name] = {
                    'usage': gd[group_name]['usage'][syll],
                    '2D velocity (mm/s)': gd[group_name]['velocity_2d_mm'][syll],
                    '3D velocity (mm/s)': gd[group_name]['velocity_3d_mm'][syll],
                    'height (mm)': gd[group_name]['height_ave_mm'][syll],
                    'dist_to_center_px': gd[group_name]['dist_to_center_px'][syll],
                }

    def get_mean_syllable_info(self):
        '''
        Populates syllable information dict with usage and scalar information.

        Returns
        -------
        '''

        if not os.path.exists(self.df_output_file):
            # Compute a syllable summary Dataframe containing usage-based
            # sorted/relabeled syllable usage and duration information from [0, max_syllable) inclusive
            df, scalar_df = merge_labels_with_scalars(self.sorted_index, self.model_path)
            print('Writing main syllable info to parquet')
            df.to_parquet(self.df_output_file, engine='fastparquet', compression='gzip')
            scalar_df.to_parquet(self.scalar_df_output, compression='gzip')
        else:
            print('Loading parquet files')
            df = pd.read_parquet(self.df_output_file, engine='fastparquet')

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

        full_df = pd.DataFrame(group_info)
        columns = full_df.columns

        output_tables = []
        if len(self.groups) < 4:
            # if there are less than 4 groups, plot the table in one row
            output_tables = [Div(text=full_df.to_html())]
        else:
            # plot 4 groups per row to avoid table being cut off by movie
            n_rows = int(len(columns) / 4)
            row_cols = np.split(columns, n_rows)

            for i in range(len(row_cols)):
                row_df = full_df[row_cols[i]]
                output_tables += [Div(text=row_df.to_html())]

        ipy_output = widgets.Output()
        with ipy_output:
            for ot in output_tables:
                show(ot)

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

        self.set_button.button_style = 'primary'

        # Set current widget values
        if len(syllables['label']) > 0:
            self.lbl_name_input.value = syllables['label']

        if len(syllables['desc']) > 0:
            self.desc_input.value = syllables['desc']

        # Update label
        self.cm_lbl.text = f'Crowd Movie {self.syll_select.index + 1}/{len(self.syll_select.options)}'

        # Update scalar values
        self.set_group_info_widgets(self.group_syll_info[self.syll_select.index]['group_info'])

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
                                 pane_widths=[3, 0, 3])

        # Display all widgets
        display(grid, self.button_box)

    def set_default_cm_parameters(self, config_data):
        '''
        Sets default crowd movie generation parameters that may be manually updated.

        Parameters
        ----------
        config_data (dict): Dict of main moseq configuration parameters.

        Returns
        -------
        config_data (dict): Updated dict of main moseq configuration parameters
         with default crowd movie generation parameters.
        '''

        config_data['separate_by'] = ''
        config_data['specific_syllable'] = None
        config_data['max_syllable'] = self.max_sylls
        config_data['max_examples'] = 20

        config_data['gaussfilter_space'] = [0, 0]
        config_data['medfilter_space'] = [0]
        config_data['sort'] = True
        config_data['pad'] = 10
        config_data['min_dur'] = 3
        config_data['max_dur'] = 60
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
            crowd_movie_paths = make_crowd_movies_wrapper(index_path, model_path, config_data, crowd_movie_dir)['all']
        else:
            # get existing crowd movie paths
            crowd_movie_paths = [f for f in glob(crowd_movie_dir + '*') if '.mp4' in f]

        if len(crowd_movie_paths) < self.max_sylls:
            print('Crowd movie list is incomplete. Generating movies...')
            config_data = self.set_default_cm_parameters(config_data)

            # Generate movies if directory does not exist
            crowd_movie_paths = make_crowd_movies_wrapper(index_path, model_path, config_data, crowd_movie_dir)['all']

        # Get syll_info paths
        info_cm_paths = [s['crowd_movie_path'] for s in self.syll_info.values()]

        if set(crowd_movie_paths) != set(info_cm_paths):
            exp = re.compile(r'.*sorted-id-(?P<sorted_id>\d{1,3}).\((?P<sort_type>\w+)\)_original-id-(?P<original_id>\d{1,3})')
            for cm in crowd_movie_paths:
                # Parse paths to get corresponding syllable number
                match_groups = exp.search(cm).groupdict()
                match_groups = {k: int(v) if v.isdigit() else v for k, v in match_groups.items()}
                sorted_num = match_groups['sorted_id']
                if sorted_num in self.syll_info:
                    sd = self.syll_info[sorted_num]
                    sd['crowd_movie_path'] = cm
                    self.syll_info[sorted_num] = {**sd, **match_groups}

        # Write to file
        with open(self.save_path, 'w+') as f:
            yml.dump(self.syll_info, f)

class CrowdMovieComparison(CrowdMovieCompareWidgets):
    '''
    Crowd Movie Comparison application class. Contains all the user inputted parameters
    within its context.

    '''

    def __init__(self, config_data, index_path, df_path, model_path, syll_info, output_dir, get_pdfs, load_parquet):
        '''
        Initializes class object context parameters.

        Parameters
        ----------
        config_data (dict): Configuration parameters for creating crowd movies.
        index_path (str): Path to loaded index file.
        df_path (str): Path to pre-computed parquet file with syllable df info.
        model_path (str): Path to loaded model.
        syll_info (dict): Dict object containing labeled syllable information.
        output_dir (str): Path to directory to store crowd movies.
        get_pdfs (bool): Generate position heatmaps for the corresponding crowd movie grouping
        load_parquet (bool): Indicates to load previously saved syllable data.
        '''

        super().__init__()

        self.config_data = config_data
        self.index_path = index_path
        self.model_path = model_path
        self.df_path = df_path
        self.scalar_df_path = os.path.join(os.path.dirname(df_path), 'moseq_dataframe.parquet')

        self.get_pdfs = get_pdfs
        self.syll_info = syll_info
        self.output_dir = output_dir
        self.max_sylls = len(syll_info)

        # Set Syllable select widget options
        self.cm_syll_select.options = list(range(self.max_sylls))

        if load_parquet:
            if df_path is not None and not os.path.exists(df_path):
                self.df_path = None
        else:
            self.df_path = None

        # Prepare current context's base session syllable info dict
        self.session_dict = {i: {'session_info': {}} for i in range(self.max_sylls)}

        _, self.sorted_index = init_wrapper_function(index_file=index_path, output_dir=output_dir)
        self.model_fit = parse_model_results(model_path)

        # Set Session MultipleSelect widget options
        self.sessions = sorted(set(self.model_fit['metadata']['uuids']))

        index_uuids = sorted(self.sorted_index['files'])
        if index_uuids != self.sessions:
            print('Error: Index file UUIDs do not match model UUIDs.')

        options = list(set([self.sorted_index['files'][s]['metadata']['SessionName'] for s in self.sessions]))

        self.cm_session_sel.options = sorted(options)

        self.get_session_mean_syllable_info_df()

        # Set widget callbacks
        self.cm_session_sel.observe(self.select_session)
        self.cm_sources_dropdown.observe(self.show_session_select)
        self.cm_trigger_button.on_click(self.on_click_trigger_button)

        self.set_default_cm_parameters()

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

    def set_default_cm_parameters(self):

        self.config_data['max_syllable'] = self.max_sylls
        self.config_data['separate_by'] = 'groups'

        self.config_data['gaussfilter_space'] = [0, 0]
        self.config_data['medfilter_space'] = [0]
        self.config_data['sort'] = True
        self.config_data['pad'] = 10
        self.config_data['min_dur'] = 4
        self.config_data['max_dur'] = 60
        self.config_data['raw_size'] = (512, 424)
        self.config_data['scale'] = 1
        self.config_data['legacy_jitter_fix'] = False
        self.config_data['cmap'] = 'jet'
        self.config_data['count'] = 'usage'

    def show_session_select(self, change):
        '''
        Callback function to change current view to show session selector when user switches
        DropDownMenu selection to 'SessionName', and hides it if the user
        selects 'groups'.

        Parameters
        ----------
        change (event): User switches their DropDownMenu selection

        Returns
        -------
        '''

        # Handle display syllable selection and update config_data crowd movie generation
        # source selector.
        options = [self.sorted_index['files'][s]['metadata'] for s in self.sessions]

        if change.new == 'SessionName':
            # Show session selector
            self.cm_session_sel.options = sorted([o['SessionName'] for o in options])
            self.cm_session_sel.layout = self.layout_visible
            self.cm_trigger_button.layout.display = 'block'
            self.config_data['separate_by'] = 'sessions'
        elif change.new == 'SubjectName':
            self.cm_session_sel.options = sorted([o['SubjectName'] for o in options])
            self.cm_session_sel.layout = self.layout_visible
            self.cm_trigger_button.layout.display = 'block'
            self.config_data['separate_by'] = 'subjects'
        elif change.new == 'group':
            # Hide session selector
            self.cm_session_sel.layout = self.layout_hidden
            self.cm_trigger_button.layout.display = 'none'
            self.config_data['separate_by'] = 'groups'

    def select_session(self, event):
        '''
        Callback function to save the list of selected sessions to config_data,
         and get session syllable info to pass to crowd_movie_wrapper and create the
         accompanying syllable scalar metadata table.

        Parameters
        ----------
        event (event): User clicks on multiple sessions in the SelectMultiple widget

        Returns
        -------
        '''

        # Set currently selected sessions
        self.config_data['session_names'] = list(self.cm_session_sel.value)

        # Update session_syllable info dict
        self.get_selected_session_syllable_info(self.config_data['session_names'])

    def get_mean_group_dict(self, group_df):
        '''
        Creates a dict object to convert to a displayed table containing syllable scalars.

        Parameters
        ----------
        group_df (pd.DataFrame): DataFrame containing mean syllable scalar data for each session and their groups

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

        for i in range(self.max_sylls):
            if 'group_info' not in self.group_syll_info[i].keys():
                self.group_syll_info[i]['group_info'] = {}

        # Update syllable info dict
        for gd in group_dicts:
            group_name = list(gd.keys())[0]
            for syll in range(self.max_sylls):
                self.group_syll_info[syll]['group_info'][group_name] = {
                    'usage': gd[group_name]['usage'][syll],
                    '2D velocity (mm/s)': gd[group_name]['velocity_2d_mm'][syll],
                    '3D velocity (mm/s)': gd[group_name]['velocity_3d_mm'][syll],
                    'height (mm)': gd[group_name]['height_ave_mm'][syll],
                    'dist_to_center_px': gd[group_name]['dist_to_center_px'][syll],
                }

    def get_session_mean_syllable_info_df(self):
        '''

        Populates session-based syllable information dict with usage and scalar information.

        Parameters
        ----------

        Returns
        -------
        '''
        if self.df_path is not None and os.path.exists(self.df_path):
            print('Loading parquet files')
            df = pd.read_parquet(self.df_path, engine='fastparquet')
            if not os.path.exists(self.scalar_df_path):
                self.scalar_df = scalars_to_dataframe(self.sorted_index, model_path=self.model_path)
                self.scalar_df.to_parquet(self.scalar_df_path, compression='gzip')
            else:
                self.scalar_df = pd.read_parquet(self.scalar_df_path)
        else:
            print('Syllable DataFrame not found. Computing and saving syllable statistics...')
            df, self.scalar_df = merge_labels_with_scalars(self.sorted_index, self.model_path)
            df.to_parquet(self.df_path, compression='gzip')
            self.scalar_df.to_parquet(self.scalar_df_path, compression='gzip')

        if self.get_pdfs:
            # Compute syllable position PDFs
            hists = compute_syllable_position_heatmaps(self.scalar_df,
                                                       syllables=range(self.max_sylls),
                                                       normalize=True).reset_index().rename(columns={'labels (usage sort)': 'syllable', 0: 'pdf'})
            self.df = pd.merge(df, hists, on=['group', 'SessionName', 'SubjectName', 'uuid', 'syllable'])

        # Get grouped DataFrame
        self.session_df = df.groupby(['SessionName', 'syllable'], as_index=False).mean()
        self.subject_df = df.groupby(['SubjectName', 'syllable'], as_index=False).mean()

        self.groups = list(df.group.unique())

        # Get group DataFrame
        self.group_df = df.groupby(['group', 'syllable'], as_index=False).mean()

        self.get_mean_group_dict(self.group_df)

    def get_selected_session_syllable_info(self, sel_sessions):
        '''
        Prepares dict of session-based syllable information to display.

        Parameters
        ----------
        sel_sessions (list): list of selected session names.

        Returns
        -------
        '''

        if self.cm_sources_dropdown.value == 'SubjectName':
            use_df = self.subject_df
        elif self.cm_sources_dropdown.value == 'SessionName':
            use_df = self.session_df

        # Get array of grouped syllable info
        session_dicts = []
        for sess in sel_sessions:
            session_dict = {
                sess: use_df[use_df[self.cm_sources_dropdown.value] == sess].drop(self.cm_sources_dropdown.value, axis=1).reset_index(
                    drop=True).to_dict()}
            session_dicts.append(session_dict)

        # Update syllable data with session info
        for sd in session_dicts:
            session_name = list(sd.keys())[0]
            for syll in range(self.max_sylls):
                self.session_dict[syll]['session_info'][session_name] = {
                    'usage': sd[session_name]['usage'][syll],
                    '2D velocity (mm/s)': sd[session_name]['velocity_2d_mm'][syll],
                    '3D velocity (mm/s)': sd[session_name]['velocity_3d_mm'][syll],
                    'height (mm)': sd[session_name]['height_ave_mm'][syll],
                    'duration': sd[session_name]['duration'][syll],
                    'dist_to_center_px': sd[session_name]['dist_to_center_px'][syll],
                }

    def get_pdf_plot(self, group_syllable_pdf, group_name):
        '''
        Helper function that creates a bokeh plot with the given PDF heatmap and figure title.

        Parameters
        ----------
        group_syllable_pdf (2D np.ndarray): Mean syllable position PDF heatmap.
        group_name (str): Name of group for generated syllable pdf

        Returns
        -------
        pdf_fig (bokeh.figure): Create bokeh figure.
        '''

        pdf_fig = figure(height=350, width=350, title=f'{group_name}')
        pdf_fig.x_range.range_padding = pdf_fig.y_range.range_padding = 0
        pdf_fig.image(image=[group_syllable_pdf],
                      x=0,
                      y=0,
                      dw=group_syllable_pdf.shape[1],
                      dh=group_syllable_pdf.shape[0],
                      palette="Viridis256")

        return pdf_fig

    def generate_crowd_movie_divs(self):
        '''
        Generates HTML divs containing crowd movies and syllable metadata tables
         from the given syllable dict file.

        Returns
        -------
        divs (list of Bokeh.models.Div): Divs of HTML videos and metadata tables.
        bk_plots (list): list of corresponding position heatmap figures.
        '''

        # Compute paths to crowd movies
        path_dict = make_crowd_movies_wrapper(self.index_path, self.model_path, self.output_dir, self.config_data)

        if self.cm_sources_dropdown.value == 'group':
            g_iter = self.groups
        else:
            g_iter = self.cm_session_sel.value

        if self.get_pdfs:
            # Get corresponding syllable position PDF
            group_syll_pdfs = get_syllable_pdfs(self.df,
                                                syllables=[self.cm_syll_select.value],
                                                groupby=self.cm_sources_dropdown.value,
                                                syllable_key='syllable')[0].drop('syllable', axis=1).reset_index()
            for group in g_iter:
                self.grouped_syll_dict[group]['pdf'] = group_syll_pdfs[group_syll_pdfs['group'] == group]['pdf']

        # Remove previously displayed data
        clear_output()

        # Get each group's syllable info to display; formatting keys.
        curr_grouped_syll_dict = {}
        for group in self.grouped_syll_dict.keys():
            curr_grouped_syll_dict[group] = {}
            for key in self.grouped_syll_dict[group].keys():
                if key == 'velocity_2d_mm':
                    new_key = '2D velocity (mm/s)'
                    curr_grouped_syll_dict[group][new_key] = self.grouped_syll_dict[group][key]
                else:
                    curr_grouped_syll_dict[group][key] = self.grouped_syll_dict[group][key]

        # Create syllable info DataFrame
        syll_info_df = pd.DataFrame(curr_grouped_syll_dict)

        # Get currently selected syllable name info
        self.curr_label = self.syll_info[self.cm_syll_select.value]['label']
        self.curr_desc = self.syll_info[self.cm_syll_select.value]['desc']

        # Create video divs including syllable metadata
        divs = []
        bk_plots = []
        for group_name, cm_path in path_dict.items():
            # Convert crowd movie metadata to HTML table
            if self.get_pdfs:
                group_info = pd.DataFrame(syll_info_df.drop('pdf', axis=0)[group_name]).to_html()
                group_syllable_pdf = syll_info_df[group_name]['pdf'].iloc[self.cm_syll_select.index]
                pdf_fig = self.get_pdf_plot(group_syllable_pdf, group_name)

                bk_plots.append(pdf_fig)
            else:
                group_info = pd.DataFrame(syll_info_df[group_name]).to_html()

            # Copy generated movie to temporary directory
            cm_dir = os.path.dirname(cm_path[0])
            tmp_path = os.path.join(cm_dir, 'tmp', f'{np.random.randint(0, 99999)}_{os.path.basename(cm_path[0])}')
            tmp_dirname = os.path.dirname(tmp_path)

            self.base_tmpdir = os.path.join(cm_dir, 'tmp')

            os.makedirs(tmp_dirname, exist_ok=True)
            shutil.copy2(cm_path[0], tmp_path)

            video_dims = get_video_info(tmp_path)['dims']

            # Insert paths and table into HTML div
            group_txt = '''
                {group_info}
                <video
                    src="{src}"; alt="{alt}"; height="{height}"; width="{width}"; preload="auto";
                    style="float: center; type: "video/mp4"; margin: 0px 10px 10px 0px;
                    border="2"; autoplay controls loop>
                </video>
            '''.format(group_info=group_info, src=tmp_path, alt=tmp_path, height=int(video_dims[1] * 0.8),
                       width=int(video_dims[0] * 0.8))

            divs.append(group_txt)

        return divs, bk_plots

    def on_click_trigger_button(self, b):
        '''
        Generates crowd movies and displays them when the user clicks the trigger button

        Parameters
        ----------
        b (ipywidgets.Button click event): User clicks "Generate Movies" button

        Returns
        -------
        '''

        # Compute current selected syllable's session dict.
        self.grouped_syll_dict = self.session_dict[self.cm_syll_select.value]['session_info']

        # Get Crowd Movie Divs
        divs, self.bk_plots = self.generate_crowd_movie_divs()

        # Display generated movies
        display_crowd_movies(self.widget_box, self.curr_label, self.curr_desc, divs, self.bk_plots)

    def crowd_movie_preview(self, syllable, groupby, nexamples):
        '''
        Helper function that triggers the crowd_movie_wrapper function and creates the HTML
        divs containing the generated crowd movies.
        Function is triggered whenever any of the widget function inputs are changed.

        Parameters
        ----------
        syllable (int or ipywidgets.DropDownMenu): Currently displayed syllable.
        nexamples (int or ipywidgets.IntSlider): Number of mice to display per crowd movie.

        Returns
        -------
        '''

        # Update current config data with widget values
        self.config_data['specific_syllable'] = int(syllable)
        self.config_data['max_examples'] = nexamples

        # Get group info based on selected DropDownMenu item
        if groupby == 'group':
            self.grouped_syll_dict = self.group_syll_info[syllable]['group_info']
            for k in self.grouped_syll_dict:
                self.grouped_syll_dict[k]['pdf'] = None

            # Get Crowd Movie Divs
            divs, self.bk_plots = self.generate_crowd_movie_divs()

            # Display generated movies
            display_crowd_movies(self.widget_box, self.curr_label, self.curr_desc, divs, self.bk_plots)
        else:
            # Display widget box until user clicks button to generate session-based crowd movies
            display(self.widget_box)