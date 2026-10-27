"""Cell 16: find where the pose changes, without a model.

    changepoints_filename = 'changepoints'

    with update_config(progress_paths['config_file']) as config_data:
        config_data['dims'] = 300

    progress_paths = update_progress(progress_filepath, 'changepoints_path', changepoints_filename)
    compute_changepoints_command(progress_paths['train_data_dir'], progress_paths, changepoints_filename)

    display(Image(join(progress_paths['pca_dirname'], changepoints_filename + '_dist.png')))

Not an output of the pipeline so much as a yardstick for it. The cell looks
for moments where the PC scores change sharply and calls those the boundaries
between behavioural units, using no model at all. Cell 19 then scores a fitted
model by how closely its syllable durations match these, which is how a kappa
is chosen.

Being model-free is the whole value: if the yardstick came from a model it
would agree with that model by construction. The notebook's warning not to
touch the peak-finding parameters for C57 mice is about keeping the yardstick
comparable between projects.

``dims=300`` sets how many random projections the scores are compared against,
so the result moves a little run to run. The tests pin what the changepoints
must be rather than where they landed: inside the recording, in order, and
spaced like behaviour rather than like noise.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from test_05_arena_mask import RECORDINGS
from test_15_scores import uuids

FRAMES = 600
FPS = 30

#: A syllable lasts on the order of a few hundred milliseconds. A mean
#: duration outside this is not a disagreement about behaviour, it is the
#: changepoint finder having failed.
PLAUSIBLE_DURATION_S = (0.1, 3.0)


def test_every_recording_gets_changepoints(changepoints, grouped):
    """1. The cell's output, one set per recording plus the frame rate."""
    with h5py.File(changepoints) as f:
        assert sorted(f) == ["cps", "cps_score", "metadata"]
        assert sorted(f["cps"]) == sorted(uuids(grouped).values())
        assert sorted(f["cps_score"]) == sorted(f["cps"])
        assert f["metadata/fps"][()] == FPS


def test_the_score_is_computed_for_every_frame(changepoints, grouped):
    """2. What the changepoints are picked out of.

    One value per frame saying how fast the pose is changing there; the
    changepoints are its peaks.
    """
    with h5py.File(changepoints) as f:
        for uuid in uuids(grouped).values():
            score = f[f"cps_score/{uuid}"]
            assert score.shape == (FRAMES,)
            assert score.dtype == np.float32


def test_the_changepoints_fall_inside_the_recording(changepoints, grouped):
    """3. A boundary outside the recording is a boundary between nothing.

    The changepoints are held in seconds-like units rather than frame
    indices, so an off-by-one in the conversion would put them past the end
    without anything else looking wrong.
    """
    with h5py.File(changepoints) as f:
        for name, uuid in uuids(grouped).items():
            cps = f[f"cps/{uuid}"][:]
            assert cps.ndim == 2 and cps.shape[1] == 1, name
            assert len(cps) > 0, f"{name} has no changepoints at all"
            assert np.isfinite(cps).all(), name
            assert (cps > 0).all(), name
            assert (cps <= FRAMES / FPS).all(), f"{name}: a changepoint past the end"


def test_the_boundaries_are_spaced_like_behaviour(changepoints, grouped):
    """4. The yardstick is only useful if it measures something.

    The stored values are the positions of the boundaries in seconds, so the
    durations cell 19 compares a model against are the gaps between them. A
    changepoint finder that fired on every frame, or twice in a recording,
    would still produce a well-formed file and would still be compared
    against. What it would not do is say anything about how long a behaviour
    lasts.
    """
    with h5py.File(changepoints) as f:
        for name, uuid in uuids(grouped).items():
            positions = f[f"cps/{uuid}"][:].ravel()
            durations = np.diff(positions)
            low, high = PLAUSIBLE_DURATION_S

            assert len(durations) > 5, f"{name}: only {len(positions)} boundaries"
            assert (durations > 0).all(), f"{name}: the boundaries are not in order"
            assert low < durations.mean() < high, (
                f"{name}: units average {durations.mean():.2f}s"
            )


def test_the_distribution_figure_is_drawn(changepoints, pca):
    """5. What the cell displays, and what a user reads a kappa from.

    The notebook shows this plot so a user can see the duration distribution
    the model will be asked to match.
    """
    from PIL import Image

    for suffix in ("png", "pdf"):
        assert (pca / f"changepoints_dist.{suffix}").exists()

    figure = Image.open(pca / "changepoints_dist.png").convert("L")
    pixels = np.asarray(figure)
    assert figure.size[0] >= 400 and figure.size[1] >= 300
    assert (pixels < 250).mean() > 0.01, "the distribution plot is blank"


def test_the_changepoints_are_recorded_in_progress(changepoints, progress_file, pca):
    """6. Where cell 19 finds the yardstick."""
    from moseq2_app.gui.progress import restore_progress_vars

    recorded = restore_progress_vars(str(progress_file), init=False, overwrite=False)
    assert recorded["changepoints_path"] == "changepoints"
    assert changepoints == pca / "changepoints.h5"
