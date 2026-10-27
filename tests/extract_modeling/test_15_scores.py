"""Cell 15: express every frame in the basis the decomposition learned.

    scores_filename = 'pca_scores'
    scores_file = join(progress_paths['pca_dirname'], scores_filename + '.h5')
    progress_paths = update_progress(progress_filepath, 'scores_path', scores_file)

    apply_pca_command(progress_paths, scores_filename)

Cell 13 learned what mouse poses look like. This projects each recording's
frames onto that basis, so a frame stops being 6400 pixels and becomes a
couple of dozen coordinates. The model in cell 18 is fit on these and never
sees a pixel.

The projection is deterministic given the basis, so unlike cell 13 there is
no randomness here to work around. What it inherits instead is the gaps: a
frame the extraction could not measure has no pose to project, and that
absence has to survive into the scores rather than being quietly filled with
zeros, which the model would read as a real posture.

Scores are keyed by uuid, which is what lets the model's output be matched
back to recordings in the analysis notebook.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from test_05_arena_mask import RECORDINGS
from test_06_extract import proc
from test_09_aggregate import entries
from test_13_pca import PIXELS, RANK, decomposition

FRAMES = 600


def uuids(index):
    """Each recording's uuid, by session name."""
    return {name: entry["uuid"] for name, entry in entries(index).items()}


def test_every_recording_is_projected(scores, grouped):
    """1. The cell's output, one block per recording."""
    with h5py.File(scores) as f:
        assert sorted(f) == ["metadata", "scores", "scores_idx"]
        assert sorted(f["scores"]) == sorted(uuids(grouped).values())
        assert sorted(f["scores_idx"]) == sorted(f["scores"])


def test_a_frame_becomes_one_coordinate_per_component(scores, grouped):
    """2. The shape of the reduction, and the whole point of it.

    6400 numbers a frame become RANK, which is what makes an AR-HMM over
    these affordable at all.
    """
    with h5py.File(scores) as f:
        for uuid in uuids(grouped).values():
            assert f[f"scores/{uuid}"].shape == (FRAMES, RANK)
            assert f[f"scores/{uuid}"].dtype == np.float32
            assert f[f"scores_idx/{uuid}"].shape == (FRAMES,)
    assert RANK < PIXELS


def test_the_scores_are_keyed_by_uuid(scores, grouped):
    """3. How a model's syllables are matched back to recordings.

    Names and paths change between projects; the uuid written at extraction
    does not, so it is what every later join is on.
    """
    expected = uuids(grouped)
    with h5py.File(scores) as f:
        for name, uuid in expected.items():
            assert f[f"metadata/{uuid}/SessionName"][()].decode() == name


def test_a_frame_with_no_mouse_becomes_the_mean_pose(scores, grouped, project):
    """4. Current behaviour, captured.

    The opening frames of each recording have no mouse, because the
    background was built from a single frame and cancels it. Those frames
    project to a vector of zeros, which in this basis is not a gap: it is the
    mean pose exactly. Nothing else in the file marks them, so the model in
    cell 18 reads them as the animal having held the average posture.

    Zero in every component is a pose no real frame produces, so the frames
    remain identifiable, and at full recording length they are a handful in
    tens of thousands. It is recorded here because the alternative, leaving
    them as NaN for the model to skip, is a different decision rather than an
    obviously better one.
    """
    with h5py.File(scores) as f:
        for name, uuid in uuids(grouped).items():
            with h5py.File(proc(project, name) / "results_00.h5") as extraction:
                empty = (extraction["frames"][:] > 0).sum(axis=(1, 2)) == 0
            projected = f[f"scores/{uuid}"][:]
            blank = (projected == 0).all(axis=1)

            assert empty.any(), f"{name} has no empty frames to check"
            np.testing.assert_array_equal(blank, empty, err_msg=name)
            assert np.isfinite(projected).all(), f"{name}: a score is not a number"


def test_the_frame_index_runs_the_length_of_the_recording(scores, grouped):
    """5. What `scores_idx` holds, which is less than its name suggests.

    It is the frame number and nothing else: it does not mark which frames
    were measurable, so it cannot be used to tell a real pose from a filled
    one.
    """
    with h5py.File(scores) as f:
        for uuid in uuids(grouped).values():
            np.testing.assert_array_equal(
                f[f"scores_idx/{uuid}"][:], np.arange(FRAMES, dtype=np.float32)
            )


def test_the_leading_components_carry_the_most_movement(scores, grouped):
    """6. The scores inherit the decomposition's ordering.

    Cell 17 keeps the first ten components. That is only sensible if the
    variance in the scores falls off the same way the decomposition said it
    would, which is a different claim from cell 13's: it is about these
    recordings rather than about the basis.
    """
    with h5py.File(scores) as f:
        pooled = np.concatenate(
            [f[f"scores/{uuid}"][:] for uuid in uuids(grouped).values()]
        )
    spread = np.nanvar(pooled, axis=0)
    assert spread[:10].sum() > 0.85 * spread.sum()


def test_the_index_records_where_the_scores_went(scores, progress_file, pca):
    """7. Where cells 16 and 18 find them."""
    from moseq2_app.gui.progress import restore_progress_vars

    recorded = restore_progress_vars(str(progress_file), init=False, overwrite=False)
    assert recorded["scores_path"] == str(scores)
    assert scores == pca / "pca_scores.h5"
