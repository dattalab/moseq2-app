'''
Constructs a jupyter notebook viewable widget users can use to identify the arena floor
and to validate extractions performed on a small chunk of data. 
'''
import time
import param
import numpy as np
import panel as pn
import holoviews as hv
from operator import add
from copy import deepcopy
from tqdm.auto import tqdm
from functools import reduce
from panel.viewable import Viewer
from os.path import exists, basename, dirname
from moseq2_extract.util import select_strel, get_strels
from moseq2_app.util import write_yaml, read_yaml
from moseq2_app.gui.progress import get_sessions
from moseq2_extract.io.video import load_movie_data
from moseq2_app.roi.controller import read_and_clean_config
from moseq2_extract.util import detect_and_set_camera_parameters
from moseq2_extract.extract.proc import get_bground_im_file, get_roi, apply_roi, threshold_chunk
from moseq2_extract.extract.extract import extract_chunk


class ArenaMaskWidget:
    def __init__(self, data_dir, config_file, session_config_path) -> None:
        self.backgrounds = {}
        self.masks = {}
        self.extracts = {}
        self.data_dir = data_dir
        self.session_config_path = session_config_path

        self.config_data = read_and_clean_config(config_file)

        sessions = get_sessions(data_dir)

        # creates session-specific configurations
        if exists(session_config_path):
            session_parameters = read_yaml(session_config_path)
            for _session in tqdm(set(map(lambda f: basename(dirname(f)), sessions)) - set(session_parameters), desc="Setting camera parameters", leave=False):
                full_path = [x for x in sessions if _session in x][0]
                session_parameters[_session] = detect_and_set_camera_parameters(deepcopy(self.config_data), full_path)
        else:
            session_parameters = {basename(dirname(f)): detect_and_set_camera_parameters(deepcopy(self.config_data), f) for f in tqdm(sessions, desc="Setting camera parameters", leave=False)}
            write_yaml(session_parameters, self.session_config_path)
        
        # instantiate ArenaMaskData with the first session in this list
        self.session_data = ArenaMaskData(path=list(session_parameters)[0], controller=self)
        self.session_data.param.path.objects = list(session_parameters)  # generates object selector in gui

        self.session_config = session_parameters
        self.sessions = {basename(dirname(f)): f for f in sessions}

        self.view = ArenaMaskView(self.session_data, self)

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.view._repr_mimebundle_(include, exclude)

    def set_session_config_vars(self):
        folder = self.session_data.path
        session = self.session_config[folder]

        session['dilate_iterations'] = self.session_data.mask_dilations
        session['min_height'], session['max_height'] = self.session_data.mouse_height
        session['manual_set_depth_range'] = not self.session_data.auto_depth
        if not self.session_data.auto_depth:
            session['bg_roi_depth_range'] = self.session_data.depth_range

    def save_session_parameters(self, event=None):
        self.set_session_config_vars()

        write_yaml(self.session_config, self.session_config_path)

        self.session_data.next_session()

    def compute_arena_mask(self):
        self.set_session_config_vars()

        folder = self.session_data.path
        background = self.get_background(folder)

        session_config = self.session_config[folder]

        strel_dilate = select_strel(session_config['bg_roi_shape'], tuple(session_config['bg_roi_dilate']))
        strel_erode = select_strel(session_config['bg_roi_shape'], tuple(session_config['bg_roi_erode']))

        rois, plane, bboxes, _, _, _ = get_roi(background,
                                                **session_config,
                                                strel_dilate=strel_dilate,
                                                strel_erode=strel_erode,
                                                get_all_data=True
                                                )
        self.masks[folder] = rois[0]
        # add to view
        return background, rois[0]

    def compute_extraction(self):
        self.set_session_config_vars()

        folder = self.session_data.path
        background = self.get_background(folder)
        session_config = self.session_config[folder]

        # compute background if not already done
        if folder not in self.masks:
            self.compute_arena_mask()
        mask = self.masks[folder]

        # get segmented frame
        raw_frames = load_movie_data(self.sessions[folder],
                                     range(0, 300),
                                     **session_config,
                                     frame_size=background.shape[::-1])

        # subtract background
        frames = (background - raw_frames)

        # filter out regions outside of ROI
        frames = apply_roi(frames, mask)#.astype(session_config['frame_dtype'])

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
        '''Assuming this will be called by an event-triggered function'''
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
    '''The link between the widget view and the underlying data'''
    ### data ###
    path = param.ObjectSelector()  # stores the currently selected path and all others
    # indicates if users want floor depth to be computed automatically
    auto_depth = param.Boolean(default=True, label="Auto-detect floor depth")
    # defines the range of depth values needed to compute mask for floor
    depth_range = param.Range(default=(650, 750), bounds=(400, 1000), label="Floor depth range (mm)")
    # defines how many iterations cv2.dilate should be run on mask computed for floor
    mask_dilations = param.Integer(default=1, bounds=(0, 10), label="Floor mask dilation iterations")
    # defines thresholds used to locate mouse for extractions
    mouse_height = param.Range(default=(10, 100), bounds=(0, 175), label="Mouse height clip (mm)")
    # stores the extracted frame number to display
    frame_num = param.Integer(default=0, bounds=(0, 300), label="Display frame (index)")
    # stores images of the arena and extractions
    images = param.Dict(default={'Background': None, 'Arena mask': None, 'Extracted mouse': None, 'Frame (background removed)': None})
    # stores class object that holds the underlying data
    controller: ArenaMaskWidget = param.Parameter()

    ### flags and actions ###
    computing = param.Boolean(default=False)  # used to indicate a computation is being performed
    # used to trigger the computation of the arena mask via a button
    compute_arena_mask = param.Action(lambda x: x.param.trigger('compute_arena_mask'), label="Compute arena mask")
    # used to trigger the computation of a small extraction via a button
    compute_extraction = param.Action(lambda x: x.param.trigger('compute_extraction'), label="Compute extraction")

    def next_session(self):
        cur_path = self.path
        paths = self.param.path.objects
        try:
            next_path_index = (paths.index(cur_path) + 1) % len(paths)
            self.path = paths[next_path_index]
        except ValueError:
            self.path = paths[0]
        self.images['Background'] = self.controller.get_background()

    @param.depends('compute_arena_mask', watch=True)
    def get_arena_mask(self):
        self.computing = True
        background, mask = self.controller.compute_arena_mask()
        self.images['Arena mask'] = mask
        self.images['Background'] = background

        self.computing = False

    @param.depends('compute_extraction', watch=True)
    def get_extraction(self):
        self.computing = True
        mouse, frame = self.controller.compute_extraction()
        self.images['Extracted mouse'] = mouse
        self.images['Frame (background subtracted)'] = frame

        self.computing = False

    # @param.depends('images', watch=True)
    @param.depends('compute_arena_mask', 'compute_extraction', 'frame_num')
    def display(self):
        panels = []
        for k, v in self.images.items():
            if not isinstance(v, np.ndarray):
                im = hv.Image([], label=k)
            elif k in ('Extracted mouse', 'Frame (background subtracted)'):
                im = hv.Image(v[self.frame_num], label=k)
            else:
                im = hv.Image(v, label=k)
            panels.append(im)
        panels = reduce(add, panels).cols(2)
        return panels.opts(shared_axes=False)

class ArenaMaskView(Viewer):
    
    def __init__(self, session_data: ArenaMaskData, controller: ArenaMaskWidget, **params):
        super().__init__(**params)

        # define widget column

        ### subsection: session selector ###
        session_selector = pn.widgets.Select.from_param(session_data.param.path, size=4, name="")

        ### subsection: arena mask parameters ###
        auto_depth = pn.widgets.Checkbox.from_param(session_data.param.auto_depth)
        arena_depth_range = pn.widgets.IntRangeSlider.from_param(session_data.param.depth_range, step=5, disabled=auto_depth.value)
        auto_depth.link(arena_depth_range, value='disabled')
        mask_dilate_iters = pn.widgets.IntSlider.from_param(session_data.param.mask_dilations, step=1)
        compute_mask_btn = pn.widgets.Button.from_param(session_data.param.compute_arena_mask)

        ### subsection: extraction parameters ###
        clip_mouse_height = pn.widgets.IntRangeSlider.from_param(session_data.param.mouse_height)
        display_frame_num = pn.widgets.IntSlider.from_param(session_data.param.frame_num, step=5)
        compute_extraction_btn = pn.widgets.Button.from_param(session_data.param.compute_extraction)

        ### subsection: saving and indicator ###
        save_one_session_btn = pn.widgets.Button(name="Save parameters for this session and move to next")
        save_one_session_btn.on_click(controller.save_session_parameters)

        indicator = pn.widgets.Progress(active=True, bar_color='warning', visible=False, name='Computing...', sizing_mode='stretch_width')
        computing_check = pn.widgets.Checkbox.from_param(session_data.param.computing, value=False, visible=False)
        computing_check.link(indicator, value='visible')

        ### link all subsections into GUI layout ###

        # combine widgets
        self.gui_col = pn.Column(
            '### Sessions',
            session_selector,
            '### Arena floor mask parameters',
            auto_depth,
            arena_depth_range,
            mask_dilate_iters,
            compute_mask_btn,
            '### Extraction parameters',
            clip_mouse_height,
            display_frame_num,
            compute_extraction_btn,
            '### Save',
            save_one_session_btn,
            indicator,
        )

        # define widget containing plots of arena and extraction
        self.plotting_col = hv.DynamicMap(session_data.display)

        self._layout = pn.Row(self.gui_col, self.plotting_col)

    def __panel__(self):
        return self._layout

