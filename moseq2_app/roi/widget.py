"""
Widget to identify the arena floor and to validate extractions performed on a small chunk of data. 
"""

import param
import numpy as np
import panel as pn
import holoviews as hv
from operator import add
from copy import deepcopy
from tqdm.auto import tqdm
from functools import reduce
from panel.viewable import Viewer
from bokeh.models import HoverTool
from os.path import exists, basename, dirname
from moseq2_app.gui.progress import get_sessions
from moseq2_extract.io.video import load_movie_data
from moseq2_extract.util import select_strel, get_strels
from moseq2_extract.extract.extract import extract_chunk
from moseq2_extract.util import detect_and_set_camera_parameters
from moseq2_app.util import write_yaml, read_yaml, read_and_clean_config, update_config
from moseq2_extract.extract.proc import get_bground_im_file, get_roi, apply_roi, threshold_chunk



# contains display parameters for the bokeh hover tools
_hover_dict = {
    'Background': HoverTool(tooltips=[('Distance from camera (mm)', '@image')]),
    'Arena mask': HoverTool(tooltips=[('Mask', '@image')]),
    'Extracted mouse': HoverTool(tooltips=[('Height from floor (mm)', '@image')]),
    'Frame (background subtracted)': HoverTool(tooltips=[('Height from floor (mm)', '@image')]),
}


class ArenaMaskWidget:
    
    def __init__(self, data_dir, config_file, session_config_path, skip_extracted=False) -> None:
        """initialize arena mask widget

        Args:
            data_dir (str): project base directory.
            config_file (str): path to config.yaml
            session_config_path (str): path to session_config.yaml.
            skip_extracted (bool, optional): boolean flag that indicates whether to skip extracted sessions. Defaults to False.
        """
        self.backgrounds = {}
        self.extracts = {}
        self.data_dir = data_dir
        self.session_config_path = session_config_path

        self.config_data = read_and_clean_config(config_file)

        sessions = get_sessions(data_dir, skip_extracted=skip_extracted)
        if len(sessions) == 0:
            self.view = None
            print('No sessions to show. There are either no sessions present or they are all extracted.')
            return

        # creates session-specific configurations
        if exists(session_config_path):
            session_parameters = read_yaml(session_config_path)
            new_sessions = set(map(lambda f: basename(dirname(f)), sessions)) - set(session_parameters)
            if len(new_sessions) > 0:
                for _session in tqdm(new_sessions, desc="Setting camera parameters", leave=False):
                    full_path = [x for x in sessions if _session in x][0]
                    session_parameters[_session] = detect_and_set_camera_parameters(
                        deepcopy(self.config_data), full_path)
                # write session config with default parameters for new sessions
                write_yaml(session_parameters, self.session_config_path)

        else:
            session_parameters = {basename(dirname(f)): detect_and_set_camera_parameters(
                deepcopy(self.config_data), f) for f in tqdm(sessions, desc="Setting camera parameters", leave=False)}
            write_yaml(session_parameters, self.session_config_path)

        # add session_config path to config.yaml

        with update_config(config_file) as temp_config_data:
            temp_config_data['session_config_path'] = session_config_path

        self.session_config = session_parameters
        self.sessions = {basename(dirname(f)): f for f in sessions}

        # instantiate ArenaMaskData with the first session in this list
        self.session_data = ArenaMaskData(path=list(self.sessions)[0], controller=self)
        self.session_data.param.path.objects = list(self.sessions)  # generates object selector in gui

        self.view = ArenaMaskView(self.session_data)

    def _repr_mimebundle_(self, include=None, exclude=None):
        if self.view is not None:
            return self.view._repr_mimebundle_(include, exclude)


    def set_session_config_vars(self):
        """
        get session default config variable.
        """
        folder = self.session_data.path
        session = self.session_config[folder]

        session['dilate_iterations'] = self.session_data.mask_dilations
        session['min_height'], session['max_height'] = self.session_data.mouse_height
        session['manual_set_depth_range'] = True
        session['bg_roi_depth_range'] = self.session_data.depth_range
        session['bg_roi_dilate'] = (self.session_data.dilation_kernel, ) * 2
        session['bg_roi_index'] = self.session_data.mask_index
        session['bg_roi_shape'] = self.session_data.mask_shape
        session['bg_roi_weights'] = (self.session_data.mask_weight1, self.session_data.mask_weight2, self.session_data.mask_weight3)

        session['cable_filter_iters'] = self.session_data.cable_filters
        session['cable_filter_shape'] = self.session_data.cable_filter_shape
        session['cable_filter_size'] = (self.session_data.cable_filter_size, ) * 2

        session['chunk_overlap'] = self.session_data.chunk_overlap
        session['compress'] = self.session_data.compress
        session['crop_size'] = (self.session_data.crop_size, ) * 2
        session['flip_classifier_smoothing'] = self.session_data.flip_classifier_smoothing
        session['frame_dtype'] = self.session_data.frame_dtype
        session['movie_dtype'] = self.session_data.movie_dtype
        session['noise_tolerance'] = self.session_data.noise_tolerance
        session['pixel_format'] = self.session_data.pixel_format
        # some simple cleanup for the spatial filter. Maybe should be more sophisticated? or move to the data class?
        spatial_filter = eval(self.session_data.spatial_filter)
        if isinstance(spatial_filter, (int, float)):
            spatial_filter = [int(spatial_filter)]
        elif not isinstance(spatial_filter, (list, tuple)):
            print('Incorrect spatial filter format. Spatial filter should be an integer or a list of integers. Setting to [3]', end='\r')
            spatial_filter = [3]
        session['spatial_filter_size'] = spatial_filter

        temporal_filter = eval(self.session_data.temporal_filter)
        if isinstance(temporal_filter, (int, float)):
            temporal_filter = [int(temporal_filter)]
        elif not isinstance(temporal_filter, (list, tuple)):
            print('Incorrect temporal filter format. Temporal filter should be an integer or a list of integers. Setting to [0]', end='\r')
            temporal_filter = [0]
        session['temporal_filter_size'] = temporal_filter

        session['tail_filter_iters'] = self.session_data.tail_filters
        session['tail_filter_shape'] = self.session_data.tail_filter_shape
        session['tail_filter_size'] = (self.session_data.tail_filter_size, ) * 2

        session['use_tracking_model'] = self.session_data.tracking_model_flag
        session['tracking_model_mask_threshold'] = self.session_data.tracking_model_mask_thresh


    def save_session_parameters(self):
        """save session paramter to session config file.
        """
        self.set_session_config_vars()

        write_yaml(self.session_config, self.session_config_path)

    def compute_arena_mask(self):
        """compute the arena mask using the paramters in the session config file.

        Returns:
            background (numpy.ndarray): background of the session
            rois(numpy.ndarray): roi for arena
    
        """
        self.set_session_config_vars()

        folder = self.session_data.path
        background = self.get_background(folder)

        session_config = self.session_config[folder]

        strel_dilate = select_strel(session_config['bg_roi_shape'], tuple(session_config['bg_roi_dilate']))
        strel_erode = select_strel(session_config['bg_roi_shape'], tuple(session_config['bg_roi_erode']))

        rois, _, = get_roi(background,
                            **session_config,
                            strel_dilate=strel_dilate,
                            strel_erode=strel_erode,
                            get_all_data=False
                            )
        # add to view
        return background, rois

    def compute_extraction(self):
        """compute extraction of the given frames

        Returns:
            (numpy.ndarray): extracted frames
            frames (numpy.ndarray): raw frames after background subtraction and roi application
        """
        self.set_session_config_vars()

        folder = self.session_data.path
        background = self.get_background(folder)
        session_config = self.session_config[folder]

        # TODO: compute mask only if not already done
        _, rois = self.compute_arena_mask()
        mask = rois[self.session_data.mask_index]

        # get segmented frame
        raw_frames = load_movie_data(self.sessions[folder],
                                     range(1, self.session_data.frames_to_extract + 1),
                                     **session_config,
                                     frame_size=background.shape[::-1])

        # subtract background
        frames = (background - raw_frames)

        # filter out regions outside of ROI
        frames = apply_roi(frames, mask)

        # filter for included mouse height range
        frames = threshold_chunk(frames, session_config['min_height'], session_config['max_height'])

        struct_elements = get_strels(session_config)
        extraction = extract_chunk(raw_frames.copy(),
                                   roi=mask,
                                   bground=background,
                                   **session_config,
                                   **struct_elements,
                                   )
        return extraction['depth_frames'], frames

    def get_background(self, folder=None):
        """Assuming this will be called by an event-triggered function"""
        if folder is None:
            folder = self.session_data.path
        if folder not in self.backgrounds:
            # get full path
            file = self.sessions[folder]
            # compute background
            bground = get_bground_im_file(file, frame_stride=1000)
            # save for recall later
            self.backgrounds[folder] = bground

        return self.backgrounds[folder]


# define data class first
class ArenaMaskData(param.Parameterized):
    ### data ###
    path = param.ObjectSelector()  # stores the currently selected path and all others
    # defines the range of depth values needed to compute mask for floor
    depth_range = param.Range(default=(650, 750), bounds=(400, 1000), label="Floor depth range (mm)")
    # defines how many iterations cv2.dilate should be run on mask computed for floor
    mask_dilations = param.Integer(default=3, bounds=(0, 10), label="Floor mask dilation iterations")
    # defines thresholds used to locate mouse for extractions
    mouse_height = param.Range(default=(10, 110), bounds=(0, 175), label="Mouse height clip (mm)")
    # stores the extracted frame number to display and how many to extract in test
    frame_num = param.Integer(default=0, bounds=(0, 99), label="Display frame (index)")
    frames_to_extract = param.Integer(default=100, bounds=(1, None), label="Number of test frames to extract")
    # stores images of the arena and extractions
    images = param.Dict(default={'Background': None, 'Arena mask': None,
                        'Extracted mouse': None, 'Frame (background subtracted)': None})
    # stores class object that holds the underlying data
    controller: ArenaMaskWidget = param.Parameter()

    ### advanced arena mask parameters ###
    adv_arena_msk_flag = param.Boolean(label="Show advanced arena mask parameters")
    mask_shape = param.Selector(objects=['ellipse', 'rectangle'], default='ellipse', label='Dilation shape')
    dilation_kernel = param.Integer(default=10, bounds=(3, 40), label="Diameter of mask dilation kernel")

    mask_weight1 = param.Number(default=1.0, bounds=(0, None), label="Mask area")
    mask_weight2 = param.Number(default=0.1, bounds=(0, None), label="Mask extent")
    mask_weight3 = param.Number(default=1.0, bounds=(0, None), label="Mask distance from center")

    mask_index = param.Integer(default=0, bounds=(0, None), label="Mask index")

    noise_tolerance = param.Number(default=30.0, bounds=(0, None), label="RANSAC noise tolerance")

    ### advanced extraction parameters ###
    adv_extraction_flag = param.Boolean(label="Show advanced extraction parameters")
    crop_size = param.Integer(default=80, bounds=(1, None), label="Crop size (width and height; pixels)")
    flip_classifier = param.Selector(label='Flip classifier')  # TODO: fill this one out better
    flip_classifier_smoothing = param.Integer(default=51, bounds=(1, None), label="Flip classifier smoothing (frames)")

    tracking_model_flag = param.Boolean(default=False, label="Use tracking model")
    tracking_model_mask_thresh = param.Number(default=-16, bounds=(None, 0), label="Tracking model mouse likelihood threshold")

    cable_filters = param.Integer(default=0, bounds=(0, None), label="Cable filter iterations")
    cable_filter_shape = param.Selector(objects=['rectangle', 'ellipse'], default='rectangle', label="Cable filter shape")
    cable_filter_size = param.Integer(default=5, bounds=(3, None), step=2, label="Cable filter size")

    tail_filters = param.Integer(default=1, bounds=(0, None), label="Tail filter iterations")
    tail_filter_shape = param.Selector(objects=['rectangle', 'ellipse'], default='ellipse', label="Tail filter shape")
    tail_filter_size = param.Integer(default=9, bounds=(3, None), step=2, label="Tail filter size")

    spatial_filter = param.String(default="[3]", label="Spatial filter(s)")
    temporal_filter = param.String(default="[0]", label="Temporal filter(s)")

    chunk_overlap = param.Integer(default=0, bounds=(0, None), label="Extraction chunk overlap (useful for tracking model)")
    frame_dtype = param.Selector(default="uint8", objects=["uint8", "uint16"], label="Output extraction datatype")
    # NOTE: in the past this has been <i2, so might change some people's analysis
    movie_dtype = param.String(default="<u2", label="Raw depth video datatype (for .dat files)")
    pixel_format = param.Selector(default="gray16le", objects=["gray16le", "gray16be"], label="Raw depth video datatype (for .avi or .mkv files)")
    compress = param.Boolean(default=False, label="Compress raw data after extraction")

    ### flags and actions ###
    computing_arena = param.Boolean(default=False)  # used to indicate a computation is being performed
    computing_extraction = param.Boolean(default=False)  # used to indicate a computation is being performed
    # used to trigger the computation of the arena mask via a button
    compute_arena_mask = param.Action(lambda x: x.param.trigger('compute_arena_mask'), label="Compute arena mask")
    # used to trigger the computation of a small extraction via a button
    compute_extraction = param.Action(lambda x: x.param.trigger('compute_extraction'), label="Compute extraction")
    save_session_and_move_btn = param.Action(lambda x: x.param.trigger('save_session_and_move_btn'), label="Save session parameters and move to next")
    save_session_btn = param.Action(lambda x: x.param.trigger('save_session_btn'), label="Save session parameters")

    @param.depends('save_session_btn', 'save_session_and_move_btn', watch=True)
    def save_session(self):
        """save session parameters
        """
        self.controller.save_session_parameters()

    @param.depends('save_session_and_move_btn', watch=True)
    def next_session(self):
        """go to next session
        """
        cur_path = self.path
        paths = self.param.path.objects
        try:
            next_path_index = (paths.index(cur_path) + 1) % len(paths)
            self.path = paths[next_path_index]
        except ValueError:
            self.path = paths[0]

    @param.depends('compute_arena_mask', 'next_session', watch=True)
    def get_arena_mask(self):
        """render the arena mask
        """
        self.computing_arena = True
        background, mask = self.controller.compute_arena_mask()
        self.images['Arena mask'] = mask
        self.images['Background'] = background
        self.param.mask_index.bounds = (0, len(mask) - 1)

        self.computing_arena = False

    @param.depends('compute_extraction', 'next_session', watch=True)
    def get_extraction(self):
        """get the extraction results
        """
        self.computing_extraction = True
        mouse, frame = self.controller.compute_extraction()
        self.images['Extracted mouse'] = mouse
        self.images['Frame (background subtracted)'] = frame

        self.computing_extraction = False

    @param.depends('get_extraction', watch=True)
    def change_frame_slider(self):
        """change the randge for the number of frames slider
        """
        self.param.frame_num.bounds = (0, min(self.frames_to_extract - 1, len(self.images['Extracted mouse']) - 1))

    @param.depends('get_arena_mask', 'get_extraction', 'frame_num', 'mask_index')
    def display(self):
        """show interactive arena mask widget

        Returns:
            panels (panel object): interactive widget for arena mask
        """
        panels = []
        for k, v in self.images.items():
            if not isinstance(v, (np.ndarray, list)):
                im = hv.Image([], label=k)
            elif k in ('Extracted mouse', 'Frame (background subtracted)'):
                _img = v[min(len(v) - 1, self.frame_num)]
                im = hv.Image(_img, label=k, bounds=(0, 0, v.shape[2], v.shape[1])
                              ).opts(xlim=(0, v.shape[2]), ylim=(0, v.shape[1]), framewise=True)
            elif k == 'Arena mask':
                _img = v[self.mask_index]
                im = hv.Image(_img, label=k, bounds=(0, 0, _img.shape[1], _img.shape[0])
                              ).opts(xlim=(0, _img.shape[1]), ylim=(0, _img.shape[0]), framewise=True)
            else:
                im = hv.Image(v, label=k, bounds=(
                    0, 0, *v.shape[::-1])).opts(xlim=(0, v.shape[1]), ylim=(0, v.shape[0]), framewise=True)
            panels.append(im.opts(
                hv.opts.Image(
                    tools=[_hover_dict[k]],
                    cmap='cubehelix',
                    xlabel="Width (pixels)",
                    ylabel="Height (pixels)",
                ),
            ))
        panels = reduce(add, panels).opts(hv.opts.Image()).cols(2)
        return panels.opts(shared_axes=False, framewise=True)


class ArenaMaskView(Viewer):

    def __init__(self, session_data: ArenaMaskData, **params):
        """view the arena mask tool
        """
        super().__init__(**params)

        def _link_data(widget, param, **kwargs):
            return widget.from_param(getattr(session_data.param, param), **kwargs)

        # define widget column

        ### subsection: session selector ###
        session_selector = _link_data(pn.widgets.Select, "path", size=4, name="")

        ### subsection: arena mask parameters ###
        arena_depth_range = _link_data(pn.widgets.IntRangeSlider, "depth_range", step=5)
        mask_dilate_iters = _link_data(pn.widgets.IntSlider, "mask_dilations", step=1)
        compute_mask_btn = _link_data(pn.widgets.Button, "compute_arena_mask")
        show_adv_arena_msk = _link_data(pn.widgets.Checkbox, "adv_arena_msk_flag")

        ### subsection: optional advanced arena mask parameters ###

        mask_shape = _link_data(pn.widgets.Select, "mask_shape", height=40)
        dilation_kernel = _link_data(pn.widgets.IntSlider, "dilation_kernel", step=1, height=40)

        mask_weight1 = _link_data(pn.widgets.NumberInput, "mask_weight1", height=45)
        mask_weight2 = _link_data(pn.widgets.NumberInput, "mask_weight2", height=45)
        mask_weight3 = _link_data(pn.widgets.NumberInput, "mask_weight3", height=45)

        mask_index = _link_data(pn.widgets.IntInput, "mask_index", height=45)
        noise_tolerance = _link_data(pn.widgets.NumberInput, "noise_tolerance", height=45)

        adv_arena_mask = pn.Column(
            mask_index,
            mask_shape,
            dilation_kernel,
            noise_tolerance,
            pn.pane.Markdown("#### Mask weightings", height=20, width=140),
            mask_weight1,
            mask_weight2,
            mask_weight3,
            visible=False,
            scroll=True,
            height=175
        )
        show_adv_arena_msk.link(adv_arena_mask, value='visible')

        ### subsection: extraction parameters ###
        clip_mouse_height = _link_data(pn.widgets.IntRangeSlider, "mouse_height")
        display_frame_num = _link_data(pn.widgets.IntSlider, "frame_num")
        extraction_frame_num = _link_data(pn.widgets.IntInput, "frames_to_extract")
        compute_extraction_btn = _link_data(pn.widgets.Button, "compute_extraction")
        show_adv_extraction = _link_data(pn.widgets.Checkbox, "adv_extraction_flag")

        ### subsection: optional advanced extraction parameters ###
        crop_size = _link_data(pn.widgets.NumberInput, "crop_size", height=40)
        # TODO: add flip classifier
        flip_classifier_smoothing = _link_data(pn.widgets.NumberInput, "flip_classifier_smoothing", height=40)
        tracking_model_flag = _link_data(pn.widgets.Checkbox, "tracking_model_flag", height=20)
        tracking_model_mask_thresh = _link_data(pn.widgets.NumberInput, "tracking_model_mask_thresh", height=40)

        cable_filters = _link_data(pn.widgets.IntInput, "cable_filters", height=40)
        cable_filter_shape = _link_data(pn.widgets.Select, "cable_filter_shape", height=40)
        cable_filter_size = _link_data(pn.widgets.IntInput, "cable_filter_size", height=50)

        tail_filters = _link_data(pn.widgets.IntInput, "tail_filters", height=40)
        tail_filter_shape = _link_data(pn.widgets.Select, "tail_filter_shape", height=40)
        tail_filter_size = _link_data(pn.widgets.IntInput, "tail_filter_size", height=40)

        spatial_filter = _link_data(pn.widgets.TextInput, "spatial_filter", height=40)
        temporal_filter = _link_data(pn.widgets.TextInput, "temporal_filter", height=40)

        chunk_overlap = _link_data(pn.widgets.IntInput, "chunk_overlap", height=40)
        frame_dtype = _link_data(pn.widgets.Select, "frame_dtype", height=40)
        movie_dtype = _link_data(pn.widgets.TextInput, "movie_dtype", height=40)
        pixel_format = _link_data(pn.widgets.Select, "pixel_format", height=40)
        compress = _link_data(pn.widgets.Checkbox, "compress", height=15)

        adv_extraction = pn.Column(
            crop_size,
            flip_classifier_smoothing,
            pn.pane.Markdown("#### Tail filtering parameters", height=20, width=300),
            tail_filters,
            tail_filter_shape,
            tail_filter_size,
            pn.pane.Markdown("#### Pose filtering parameters", height=20, width=300),
            spatial_filter,
            temporal_filter,
            pn.pane.Markdown("#### Video parameters", height=20, width=300),
            chunk_overlap,
            frame_dtype,
            movie_dtype,
            pixel_format,
            compress,
            pn.pane.Markdown("#### Tracking model parameters", height=20, width=300),
            tracking_model_flag,
            tracking_model_mask_thresh,
            pn.pane.Markdown("#### Cable filtering parameters", height=20, width=300),
            cable_filters,
            cable_filter_shape,
            cable_filter_size,
            visible=False,
            scroll=True,
            height=175
        )
        show_adv_extraction.link(adv_extraction, value='visible')

        ### subsection: saving and indicator ###
        save_session_and_move_btn = _link_data(pn.widgets.Button, "save_session_and_move_btn")
        save_session_btn = _link_data(pn.widgets.Button, "save_session_btn")

        # computing arena indicator
        indicator = pn.Row(
            pn.pane.Markdown('Finding arena...', width=100),
            pn.widgets.Progress(active=True, bar_color='warning', width=160),
            visible=False,
            height=45,
        )
        computing_check = _link_data(pn.widgets.Checkbox, "computing_arena", value=False, visible=False)
        computing_check.link(indicator, value='visible')
        def _link_button_visibility(target, event):
            target.visible = not event.new
        computing_check.link(compute_mask_btn, callbacks={'value': _link_button_visibility})

        # computing extraction
        indicator2 = pn.Row(
            pn.pane.Markdown('Extracting...', width=100),
            pn.widgets.Progress(active=True, bar_color='info', width=160),
            visible=False,
            height=45,
        )
        computing_check2 = _link_data(pn.widgets.Checkbox, "computing_extraction", value=False, visible=False)
        computing_check2.link(indicator2, value='visible')
        computing_check2.link(compute_extraction_btn, callbacks={'value': _link_button_visibility})

        ### link all subsections into GUI layout ###

        # combine widgets
        self.gui_col = pn.Column(
            '### Sessions',
            session_selector,
            '### Arena floor mask parameters',
            arena_depth_range,
            mask_dilate_iters,
            compute_mask_btn,
            indicator,
            show_adv_arena_msk,
            adv_arena_mask,
            '### Extraction parameters',
            clip_mouse_height,
            extraction_frame_num,
            display_frame_num,
            compute_extraction_btn,
            indicator2,
            show_adv_extraction,
            adv_extraction,
            pn.pane.Markdown('### Save', height=40),
            save_session_btn,
            save_session_and_move_btn,
        )

        # define widget containing plots of arena and extraction
        self.plotting_col = hv.DynamicMap(session_data.display).opts(framewise=True)

        self._layout = pn.Row(self.gui_col, self.plotting_col)

    def __panel__(self):
        return self._layout
