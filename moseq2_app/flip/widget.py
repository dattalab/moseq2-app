import io
import html
import h5py
import pickle
import warnings
import panel as pn
import holoviews as hv
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path
from moseq2_common.records import RECORD_ATTRIBUTE, fingerprint, record_text
from moseq2_common.timestamps import frame_rate
from moseq2_common.yaml import read_yaml
from collections import defaultdict
from moseq2_extract.extract.proc import get_flips
from moseq2_extract.util import gen_batch_sequence
from moseq2_common.video import write_frames_preview
from moseq2_app.flip.train import SAVE_AND_RETRAIN
from moseq2_app.labels import (FLIP_LABELS_STEP, STOP, WIPE, labelled_in, remove_labels, save_record, stale,
                               stale_labels_message, stopped_message)


# Workaround for HoloViews/Bokeh _buffers parameter issue
# Bokeh 2.4.2 includes '_buffers' in messages but HoloViews 1.14.7 
# doesn't filter it before passing to Param, causing ValueError
def _patch_holoviews_stream_update():
    """Patch HoloViews Stream._set_stream_parameters to handle _buffers error."""
    from holoviews.streams import Stream
    
    # Store original method
    _original_set_stream_parameters = Stream._set_stream_parameters
    
    def _patched_set_stream_parameters(self, **kwargs):
        """Wrapper that catches and handles _buffers parameter error."""
        try:
            # Try with all parameters first (might work in some version combos)
            return _original_set_stream_parameters(self, **kwargs)
        except ValueError as e:
            # Only intervene if the error is specifically about _buffers
            if '_buffers' in str(e) and 'is not a parameter' in str(e):
                # Remove _buffers and retry
                filtered_kwargs = {k: v for k, v in kwargs.items() if k != '_buffers'}
                return _original_set_stream_parameters(self, **filtered_kwargs)
            # Re-raise any other ValueError
            raise
    
    # Replace the method with our patched version
    Stream._set_stream_parameters = _patched_set_stream_parameters

# Apply the patch when this module is imported
_patch_holoviews_stream_update()


def _extraction_complete(file_path: Path):
    """Whether ``file_path`` is a finished extraction: an ``.h5`` under its final name holding frames."""
    try:
        with h5py.File(file_path, 'r') as f:
            return 'frames' in f and 'metadata/uuid' in f
    except OSError:
        return False


def _note_flip_correction(f, clf_path, settings):
    """Note in the extraction's record (its ``/provenance`` attribute) the classifier that flipped its frames
    and when (``applied``), which makes it another state of the session's frames (``moseq2_common.records.run_of``)."""
    stored = f.attrs.get(RECORD_ATTRIBUTE)
    if stored is None:
        return
    record = read_yaml(io.StringIO(stored.decode() if isinstance(stored, bytes) else str(stored)))
    record.setdefault('notes', {})['flip_correction'] = {
        'classifier': str(Path(clf_path).resolve()), 'fingerprint': fingerprint(clf_path), **settings,
        'applied': datetime.now().astimezone().isoformat()}
    f.attrs[RECORD_ATTRIBUTE] = record_text(record)


def _find_extractions(data_path: str):
    files = Path(data_path).glob('**/*.h5')
    files = sorted(f for f in files if _extraction_complete(f))
    if len(set([f.name for f in files])) < len(files):
        files = {f.parent.name + '/' + f.name: f for f in files}
    else:
        files = {f.name: f for f in files}
    return files


class FlipClassifierWidget:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.sessions = _find_extractions(data_path)
        if len(self.sessions) == 0:
            raise ValueError(
                f"No completed extractions found in '{data_path}'. "
                "Make sure you have run the 'aggregate results' step in the "
                "extraction notebook and that input_dir points to the folder "
                "containing your extracted .h5 files (e.g. the "
                "aggregate_results/ directory)."
            )
        self.selected_frame_ranges_dict = defaultdict(list)
        self.curr_total_selected_frames = 0
        self.status = pn.pane.HTML('')
        self.unsaved = False

        self.session_select_dropdown = pn.widgets.Select(options=list(self.sessions), name='Session', value=list(self.sessions)[0])
        self.frame_num_slider = pn.widgets.IntSlider(name='Current Frame', start=0, end=1000, step=1, value=1)
        self.start_button = pn.widgets.Button(name='Start Range', button_type='primary')
        self.face_left_button = pn.widgets.Button(name='Facing Left', button_type='success', width=140, visible=False)
        self.face_right_button = pn.widgets.Button(name='Facing Right', button_type='success', width=140, visible=False)
        self.selected_ranges = pn.widgets.MultiSelect(name='Selected Ranges', options=[])
        self.delete_selection_button = pn.widgets.Button(name='Delete Selection', button_type='danger')
        self.save_button = pn.widgets.Button(name='Save Labels', button_type='success')
        self.curr_total_label = pn.pane.Markdown(f'Current Total Selected Frames: {self.curr_total_selected_frames}')

        # saved labels made for frames that have changed since, until the person wipes them or stops
        self.stale_labels = None
        self.labels_check = pn.pane.HTML('')
        self.wipe_button = pn.widgets.Button(name=WIPE, button_type='danger')
        self.stop_button = pn.widgets.Button(name=STOP)
        self.labels_box = pn.Column(self.labels_check, pn.Row(self.wipe_button, self.stop_button), visible=False)

        self.facing_info = pn.pane.Markdown('To finish selection, click on direction the animal is facing', visible=False)
        self.facing_row = pn.Row(self.face_left_button, self.face_right_button)
        self.range_box = pn.Column(
            self.curr_total_label,
            pn.pane.Markdown('#### Selected Correct Frame Ranges'),
            self.selected_ranges,
            self.delete_selection_button,
            self.save_button,
            self.status,
        )

        self.forward_button = pn.widgets.Button(name='Forward', button_type='primary', width=142)
        self.backward_button = pn.widgets.Button(name='Backward', button_type='primary', width=142)

        self.frame_advancer_row = pn.Row(self.backward_button, self.forward_button)

        self.widgets = pn.Column(
            self.labels_box,
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
        self.save_button.on_click(lambda event: self.save_frame_ranges())
        self.wipe_button.on_click(lambda event: self.wipe_labels())
        self.stop_button.on_click(lambda event: self.stop_on_labels())

        self.forward_button.on_click(self.advance_frame)
        self.backward_button.on_click(self.rewind_frame)

        self.session_select_dropdown.param.watch(self.changed_selected_session, 'value')
        self.session_select_dropdown.value = list(self.sessions)[0]

        self.load_frame_ranges()

    def load_frame_ranges(self):
        """Show the ranges saved in ``training_data_path``, if there are any.

        Ranges saved for an extraction that is not in ``data_path`` cannot be
        shown. They stay in the file until the next save, which writes only
        what the widget shows, so the labels are marked unsaved and the
        sessions are named.
        """
        path = Path(self.training_data_path)
        if not path.exists():
            return
        with open(path, 'rb') as f:
            saved_paths, saved_ranges = pickle.load(f)

        # ranges label the frames as they were; an extraction written since (re-extracted, or flipped by
        # a classifier) carries the record of another run and is not what they label
        made_for = {entry.get('session'): entry for entry in labelled_in(path)}
        found = [stale(made_for[session], self.sessions[session], 'extracted')
                 for session in made_for if session in self.sessions]
        found = [one for one in found if one is not None]
        if found:
            self.stale_labels = stale_labels_message(path, found)
            self.labels_check.object = f'<p style="color: red;">{html.escape(self.stale_labels)}</p>'
            self.labels_box.visible = True
            self.save_button.disabled = True
            return

        missing = {}
        for session, ranges in saved_ranges.items():
            if not ranges:
                continue
            if session not in self.sessions:
                missing[session] = ranges
                continue
            for left, selected_range in ranges:
                self.add_range(session, left, selected_range)

        if missing:
            listed = '\n'.join(
                f'  {session} ({len(ranges)} range(s), {sum(len(r) for _, r in ranges)} frames), '
                f'saved from {saved_paths.get(session, "an unrecorded path")}'
                for session, ranges in missing.items()
            )
            print(
                f'{path} holds labeled frame ranges for extractions that are not in {self.data_path}:\n'
                f'{listed}\n'
                'They are not shown, and saving the labels removes them from the file. To keep them, '
                'do not save; set input_dir to the folder that holds those extractions and run the '
                'cells again.'
            )
            self.mark_unsaved()

    def wipe_labels(self):
        """Delete the saved labels made for other frames, and label from nothing."""
        remove_labels(self.training_data_path)
        self.stale_labels = None
        self.labels_box.visible = False
        self.save_button.disabled = False

    def stop_on_labels(self):
        """Leave the saved labels as they are, for the person to copy before labeling again."""
        self.labels_check.object = (f'<p style="color: red;">{html.escape(self.stale_labels)}</p>'
                                    f'<p>{html.escape(stopped_message(self.training_data_path))}</p>')
        self.wipe_button.disabled = self.stop_button.disabled = True

    def mark_unsaved(self):
        self.unsaved = True
        self.status.object = (
            '<p style="color: red;">You have unsaved changes to your labels. '
            'Hit save when done editing to keep them.</p>'
        )

    def mark_saved(self):
        self.unsaved = False
        self.status.object = '<p style="color: green;">Flip labels saved.</p>'

    def warn_if_unsaved(self):
        """Warn when training would read labels other than the ones the widget shows.

        Training reads ``training_data_path``, which changes only on a save. A
        missing file or one without labels stops training with its own message,
        so the warning is for the case that trains without one: older labels
        were saved, and the widget has changed since.
        """
        path = Path(self.training_data_path)
        if not self.unsaved or not path.exists():
            return
        with open(path, 'rb') as f:
            _, saved = pickle.load(f)
        saved_ranges = [r for ranges in saved.values() for _, r in ranges]
        if not saved_ranges:
            return
        shown_ranges = [r for ranges in self.selected_frame_ranges_dict.values() for _, r in ranges]
        warnings.warn(
            f'The flip labels have unsaved changes. The widget shows {len(shown_ranges)} range(s) '
            f'({sum(len(r) for r in shown_ranges)} frames), but training reads only the labels saved in '
            f'{path}: {len(saved_ranges)} range(s) ({sum(len(r) for r in saved_ranges)} frames), and '
            f'trains on those. To train on the ranges the widget shows, {SAVE_AND_RETRAIN}.',
            UserWarning,
            stacklevel=2,
        )

    def add_range(self, session, left, selected_range):
        """Hold one labeled range and list it under Selected Ranges."""
        beginning = 'L' if left else 'R'
        display_selected_range = f'{beginning} - {selected_range} - {session}'

        self.curr_total_selected_frames += len(selected_range)
        self.curr_total_label.object = f'Current Total Selected Frames: {self.curr_total_selected_frames}'

        self.selected_frame_ranges_dict[session].append((left, selected_range))

        self.selected_ranges.options = self.selected_ranges.options + [display_selected_range]
        self.selected_ranges.value = []

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
        self.add_range(self.session_select_dropdown.value, left, range(self.start, self.stop))
        self.mark_unsaved()

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
            self.mark_unsaved()

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
        """Write the ranges the widget shows to ``training_data_path``, replacing what it held, and the save's
        record beside it, naming each labeled session's extraction and the run that wrote it
        (``moseq2_app.labels``)."""
        if self.stale_labels is not None:
            raise ValueError(f'{self.stale_labels} Nothing was saved: choose {WIPE} or {STOP} in the labeling '
                             'widget above.')
        ranges = dict(self.selected_frame_ranges_dict)
        labelled = [{'path': self.sessions[session], 'session': session} for session, labeled in ranges.items()
                    if labeled and session in self.sessions]
        with open(self.training_data_path, 'wb') as f:
            pickle.dump((self.sessions, ranges), f)
        save_record(self.training_data_path, FLIP_LABELS_STEP, labelled)
        self.mark_saved()
            
    def apply_flip_classifier(self, clf_path, chunk_size=4000, chunk_overlap=0,
                              smoothing=51, frame_path='frames', fps=None,
                              write_movie=False, verbose=True):
        """
        Apply a trained flip classifier on previously extracted data to flip the mice to the correct orientation.
    
        Args:
        chunk_size (int): size of frame chunks to process in batches.
        chunk_overlap (int): number of frames to overlap between chunks to improve classification precision between chunks.
        smoothing (int): kernel size of the applied median filter on the flip classifier results
        fps (float): the flipped preview movie's frame rate; None for each session's own (``/metadata/fps``).
        verbose (bool): displays the tqdm progress bars for each session.
        """
        video_pipe = None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            for key, path in tqdm(self.sessions.items(), desc='Flipping extracted sessions...'):
                # Open h5 file to stream and correct/update stored frames and scalar angles.
                path = str(path)
                with h5py.File(path, mode='a') as f:
                    # Frames are rotated in place, so a second pass would flip them back. The file
                    # records 'in progress' before the first frame changes and 'complete' at the end.
                    state = f.attrs.get('flip_correction')
                    if state == 'complete':
                        print(f'Flip correction was already applied to {key}; skipping it.')
                        continue
                    if state is not None:
                        raise RuntimeError(f'Flip correction of {key} did not finish, so some of its frames are '
                                           'already rotated. Re-extract this session before applying the flip '
                                           'classifier again.')

                    output_movie = path.replace('.h5', '_flipped.mp4')
                    session_movie = write_movie
                    movie_fps = fps if fps is not None or not write_movie else frame_rate(f)
    
                    frames = f[frame_path]
                    frame_batches = list(gen_batch_sequence(len(frames), chunk_size, chunk_overlap))
                    committed = np.zeros(len(frames), dtype=bool)
    
                    for batch_number, batch in enumerate(tqdm(frame_batches, desc=f'Adjusting flips: {key}', disable=not verbose)):
                        frame_batch = frames[batch]
    
                        # apply flip classifier on each batch to find which frames to flip
                        flips = get_flips(frame_batch.copy(), flip_file=clf_path, smoothing=smoothing)
                        batch_indices = np.asarray(list(batch))
                        pending = ~committed[batch_indices]
                        if chunk_overlap > 0 and batch_number < len(frame_batches) - 1:
                            pending[-chunk_overlap:] = False
                        flip_indices = np.where(flips & pending)
                        committed[batch_indices[pending]] = True

                        # The classifier loads in the first get_flips call, so a bad classifier fails
                        # above with the session untouched; the marker goes on just before the first write.
                        if batch_number == 0:
                            f.attrs['flip_correction'] = 'in progress'
                            f.flush()

                        # rewrite the frames with the newly classified orientation
                        frame_batch[flip_indices] = np.rot90(f[frame_path][batch][flip_indices], k=2, axes=(1, 2))
                        f[frame_path][batch] = frame_batch
    
                        # augment recorded scalar value to reflect orientation switches
                        flip_absolute = np.asarray(list(batch))[flip_indices]
                        f['scalars/angle'][flip_absolute] += np.pi
    
                        if session_movie:
                            try:
                                # Write the frames this batch committed, so a frame shared with the next batch appears once
                                video_pipe = write_frames_preview(output_movie,
                                                                  frame_batch[pending],
                                                                  pipe=video_pipe,
                                                                  close_pipe=False,
                                                                  depth_min=0,
                                                                  depth_max=100,
                                                                  fps=movie_fps,
                                                                  progress_bar=verbose)
                            except AttributeError as e:
                                # the movie is only a preview; the remaining frames still need correcting
                                print(f'Could not generate flipped movie for {key}:{path}. Continuing without it.')
                                print(e)
                                session_movie = False
                    # Check if video is done writing. If not, wait.
                    if video_pipe is not None:
                        video_pipe.communicate()
                        video_pipe = None

                    _note_flip_correction(f, clf_path, {'chunk_size': chunk_size, 'chunk_overlap': chunk_overlap,
                                                        'smoothing': smoothing})
                    f.attrs['flip_correction'] = 'complete'

    @property
    def training_data_path(self):
        return str(self.data_path.parent / 'flip-training-frame-ranges.p')
