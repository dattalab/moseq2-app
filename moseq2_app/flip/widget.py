import h5py
import pickle
import warnings
import panel as pn
import holoviews as hv
from tqdm import tqdm
import numpy as np
from pathlib import Path
import ruamel.yaml as yaml
from collections import defaultdict
from moseq2_extract.extract.proc import get_flips
from moseq2_extract.util import gen_batch_sequence
from moseq2_extract.io.video import write_frames_preview

def _extraction_complete(file_path: Path):
    config = yaml.safe_load(file_path.read_text())
    return config['complete']


def _find_extractions(data_path: str):
    files = Path(data_path).glob('**/*.h5')
    files = sorted(f for f in files if _extraction_complete(f.with_suffix('.yaml')))
    if len(set([f.name for f in files])) < len(files):
        files = {f.parent.name + '/' + f.name: f for f in files}
    else:
        files = {f.name: f for f in files}
    return files


class FlipClassifierWidget:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.sessions = _find_extractions(data_path)
        # self.selected_frame_ranges_dict = {k: [] for k in self.path_dict}
        self.selected_frame_ranges_dict = defaultdict(list)
        self.curr_total_selected_frames = 0

        self.session_select_dropdown = pn.widgets.Select(options=list(self.sessions), name='Session', value=list(self.sessions)[1])
        self.frame_num_slider = pn.widgets.IntSlider(name='Current Frame', start=0, end=1000, step=1, value=1)
        self.start_button = pn.widgets.Button(name='Start Range', button_type='primary')
        self.face_left_button = pn.widgets.Button(name='Facing Left', button_type='success', width=140, visible=False)
        self.face_right_button = pn.widgets.Button(name='Facing Right', button_type='success', width=140, visible=False)
        self.selected_ranges = pn.widgets.MultiSelect(name='Selected Ranges', options=[])
        self.delete_selection_button = pn.widgets.Button(name='Delete Selection', button_type='danger')
        self.curr_total_label = pn.pane.Markdown(f'Current Total Selected Frames: {self.curr_total_selected_frames}')

        self.facing_info = pn.pane.Markdown('To finish selection, click on direction the animal is facing', visible=False)
        self.facing_row = pn.Row(self.face_left_button, self.face_right_button)
        self.range_box = pn.Column(
            self.curr_total_label,
            pn.pane.Markdown('#### Selected Correct Frame Ranges'),
            self.selected_ranges,
            self.delete_selection_button
        )

        self.forward_button = pn.widgets.Button(name='Forward', button_type='primary', width=142)
        self.backward_button = pn.widgets.Button(name='Backward', button_type='primary', width=142)

        self.frame_advancer_row = pn.Row(self.backward_button, self.forward_button)

        self.widgets = pn.Column(
            self.session_select_dropdown,
            self.frame_num_slider,
            self.start_button,
            self.frame_advancer_row,
            self.facing_info,
            self.facing_row,
            self.range_box,
            width=325
        )

        self.frame_display = hv.DynamicMap(self.display_frame, streams=[self.frame_num_slider.param.value]).opts(
            frame_width=400, frame_height=400, aspect='equal', xlim=(0, 1), ylim=(0, 1)
        )

        self.start_button.on_click(self.start_stop_frame_range)
        self.face_left_button.on_click(lambda event: self.facing_range_callback(event, True))
        self.face_right_button.on_click(lambda event: self.facing_range_callback(event, False))
        self.delete_selection_button.on_click(self.on_delete_selection_clicked)

        self.forward_button.on_click(self.advance_frame)
        self.backward_button.on_click(self.rewind_frame)

        self.session_select_dropdown.param.watch(self.changed_selected_session, 'value')
        self.session_select_dropdown.value = list(self.sessions)[0]

    def display_frame(self, value):
        if hasattr(self, 'frames'):
            frame = self.frames[value]
        else:
            frame = None
        # set bounds for the image
        return hv.Image(frame, bounds=(0, 0, 1, 1)).opts(cmap="cubehelix")

    def advance_frame(self, event):
        if self.frame_num_slider.value < self.frame_num_slider.end:
            self.frame_num_slider.value += 1
    
    def rewind_frame(self, event):
        if self.frame_num_slider.value > 0:
            self.frame_num_slider.value -= 1

    def start_stop_frame_range(self, event):
        if self.start_button.name == 'Start Range':
            self.start = self.frame_num_slider.value
            self.start_button.name = 'Cancel Select'
            self.start_button.button_type = 'danger'
            self.face_left_button.visible = True
            self.face_right_button.visible = True
            self.facing_info.visible = True
            self.facing_info.object = f'To finish selection, click on direction the animal is facing. Start: {self.start}'
        else:
            self.start_button.name = 'Start Range'
            self.start_button.button_type = 'primary'
            self.face_left_button.visible = False
            self.face_right_button.visible = False
            self.facing_info.visible = False

    def facing_range_callback(self, event, left):
        self.stop = self.frame_num_slider.value
        if self.stop > self.start:
            self.update_state_on_selected_range(left)
            self.face_left_button.visible = False
            self.face_right_button.visible = False
            self.facing_info.visible = False
            self.start_button.name = 'Start Range'
            self.start_button.button_type = 'primary'

    def update_state_on_selected_range(self, left):
        selected_range = range(self.start, self.stop)

        beginning = 'L' if left else 'R'
        display_selected_range = f'{beginning} - {selected_range} - {self.session_select_dropdown.value}'

        self.curr_total_selected_frames += len(selected_range)
        self.curr_total_label.object = f'Current Total Selected Frames: {self.curr_total_selected_frames}'

        self.selected_frame_ranges_dict[self.session_select_dropdown.value].append((left, selected_range))

        self.selected_ranges.options = self.selected_ranges.options + [display_selected_range]
        self.selected_ranges.value = []

        self.save_frame_ranges()

    def on_delete_selection_clicked(self, event):
        selected_range = self.selected_ranges.value
        if selected_range:
            vals = selected_range[0].split(' - ')
            delete_key = vals[2]
            direction = vals[0] == 'L'
            range_to_delete = eval(vals[1])

            to_drop = (direction, range_to_delete)
            self.selected_frame_ranges_dict[delete_key].remove(to_drop)

            self.curr_total_selected_frames -= len(range_to_delete)
            self.curr_total_label.object = f'Current Total Selected Frames: {self.curr_total_selected_frames}'

            l = list(self.selected_ranges.options)
            l.remove(selected_range[0])

            self.selected_ranges.options = l
            self.selected_ranges.value = []

    def changed_selected_session(self, event):
        if self.start_button.name == 'Cancel Select':
            self.start_button.name = 'Start Range'
            self.start_button.button_type = 'primary'

        self.frame_num_slider.value = 1
        self.widgets.loading = True

        with h5py.File(self.sessions[event.new], mode='r') as f:
            self.frame_num_slider.end = f['frames'].shape[0] - 1
            self.frames = f['frames'][()]

        self.widgets.loading = False

        self.frame_num_slider.value = 0

    def show(self):
        return pn.Row(self.widgets, self.frame_display)

    def save_frame_ranges(self):
        with open(self.training_data_path, 'wb') as f:
            pickle.dump((self.sessions, dict(self.selected_frame_ranges_dict)), f)
            
    def apply_flip_classifier(self, clf_path, chunk_size=4000, chunk_overlap=0,
                              smoothing=51, frame_path='frames', fps=30,
                              write_movie=False, verbose=True):
        """
        Apply a trained flip classifier on previously extracted data to flip the mice to the correct orientation.
    
        Args:
        chunk_size (int): size of frame chunks to process in batches.
        chunk_overlap (int): number of frames to overlap between chunks to improve classification precision between chunks.
        smoothing (int): kernel size of the applied median filter on the flip classifier results
        verbose (bool): displays the tqdm progress bars for each session.
        """
        video_pipe = None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            for key, path in tqdm(self.sessions.items(), desc='Flipping extracted sessions...'):
                # Open h5 file to stream and correct/update stored frames and scalar angles.
                path = str(path)
                with h5py.File(path, mode='a') as f:
                    output_movie = path.replace('.h5', '_flipped.mp4')
    
                    frames = f[frame_path]
                    frame_batches = gen_batch_sequence(len(frames)-1, chunk_size, chunk_overlap)
    
                    for batch in tqdm(frame_batches, desc=f'Adjusting flips: {key}', disable=not verbose):
                        frame_batch = frames[batch]
    
                        # apply flip classifier on each batch to find which frames to flip
                        flips = get_flips(frame_batch.copy(), flip_file=clf_path, smoothing=smoothing)
                        flip_indices = np.where(flips)
    
                        # rewrite the frames with the newly classified orientation
                        frame_batch[flip_indices] = np.rot90(f[frame_path][batch][flip_indices], k=2, axes=(1, 2))
                        f[frame_path][batch] = frame_batch
    
                        # augment recorded scalar value to reflect orientation switches
                        f['scalars/angle'][flip_indices] += np.pi
    
                        if write_movie:
                            try:
                                # Writing frame batch to mp4 file
                                video_pipe = write_frames_preview(output_movie,
                                                                  frame_batch,
                                                                  pipe=video_pipe,
                                                                  close_pipe=False,
                                                                  depth_min=0,
                                                                  depth_max=100,
                                                                  fps=fps,
                                                                  progress_bar=verbose)
                            except AttributeError as e:
                                warnings.warn(f'Could not generate flipped movie for {key}:{path}. Skipping...')
                                print(e)
                                print(e.__traceback__)
                                break
                    # Check if video is done writing. If not, wait.
                    if video_pipe is not None:
                        video_pipe.communicate()
                        video_pipe = None

    @property
    def training_data_path(self):
        return str(self.data_path.parent / 'flip-training-frame-ranges.p')