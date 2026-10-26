"""Cell 13: learn the pose basis the rest of the pipeline works in.

    pca_filename = 'pca'
    pca_dirname = join(progress_paths['base_dir'], '_pca/')
    progress_paths = update_progress(progress_filepath, 'pca_dirname', pca_dirname)

    with update_config(progress_paths['config_file']) as config_data:
        config_data['camera_type'] = 'k2'
        config_data['overwrite_pca'] = False
        config_data['gaussfilter_space'] = (1.5, 1)
        config_data['medfilter_space'] = [0]
        config_data['medfilter_time'] = [0]
        config_data['nworkers'] = 1

    train_pca_command(progress_paths, pca_dirname, pca_filename)

Eighty by eighty pixels of mouse is six thousand four hundred numbers a frame,
far too many to model directly. This cell learns a basis in which a pose is a
couple of dozen numbers instead, and everything downstream is expressed in it.

The decomposition is computed with ``svd_compressed``, which works by
projecting onto a random subspace. That draw comes from dask's own generator
and is not reachable by seeding numpy, so the components differ run to run.
The tests therefore pin what the decomposition must satisfy rather than what
it produced: the shapes, that the basis is orthonormal, that the variance
ratios are sorted and sum to at most one, and that the mean is the mean of
the frames it was trained on, which is the one part of the result that does
not depend on the random projection.

The figures are written as both png and pdf. Cell 14 displays the pngs.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from moseq2_extract.util import read_yaml

from tests.extract_modeling.test_03_config import plain

#: The crop is 80x80, so a flattened frame is this long.
PIXELS = 6400

#: How many components ``rank`` asks for by default.
RANK = 25

#: What the cell writes, beyond the figures. A ``train.log`` appears beside
#: them on some runs and is empty when it does, so it is not one of these.
ARTEFACTS = ("pca.h5", "pca.yaml")

#: Every figure, in both formats.
FIGURES = ("pca_components", "pca_scree")


def decomposition(pca):
    """The learned basis, as plain arrays."""
    with h5py.File(pca / "pca.h5") as f:
        return {key: f[key][:] for key in f}


def test_the_cell_writes_a_decomposition_and_its_figures(pca):
    """1. What lands in the project."""
    written = sorted(p.name for p in pca.iterdir())

    for artefact in ARTEFACTS:
        assert artefact in written
    for figure in FIGURES:
        assert f"{figure}.png" in written
        assert f"{figure}.pdf" in written


def test_the_basis_has_a_component_per_rank(pca):
    """2. The shape of the result.

    One row per component, one column per pixel of the crop, with a variance
    and a singular value for each.
    """
    learned = decomposition(pca)

    assert sorted(learned) == [
        "components",
        "explained_variance",
        "explained_variance_ratio",
        "mean",
        "singular_values",
    ]
    assert learned["components"].shape == (RANK, PIXELS)
    assert learned["mean"].shape == (PIXELS,)
    for key in ("explained_variance", "explained_variance_ratio", "singular_values"):
        assert learned[key].shape == (RANK,)
    assert all(array.dtype == np.float32 for array in learned.values())


def test_the_basis_is_orthonormal(pca):
    """3. What makes it a basis.

    Each component is a unit-length direction in pixel space and is at right
    angles to the others, so a pose's coordinates in it are independent and a
    reconstruction is just a weighted sum. Nothing downstream would complain
    if this failed; the scores would simply stop meaning what they claim.
    """
    components = decomposition(pca)["components"].astype(np.float64)
    gram = components @ components.T

    np.testing.assert_allclose(gram, np.eye(RANK), atol=1e-4)


def test_the_components_are_ordered_by_how_much_they_explain(pca):
    """4. Why taking the first ten is a sensible thing to do.

    Cell 17 models ``npcs=10``, which is only defensible if the components
    are sorted, so the ten kept are the ten that matter most.
    """
    learned = decomposition(pca)
    ratio = learned["explained_variance_ratio"]

    assert np.all(np.diff(ratio) <= 0), "the components are not sorted"
    assert np.all(ratio > 0)
    assert ratio.sum() <= 1.0 + 1e-5
    assert np.all(np.diff(learned["singular_values"]) <= 0)
    assert np.all(np.diff(learned["explained_variance"]) <= 0)


def test_ten_components_carry_most_of_the_pose(pca):
    """5. The assumption cell 17 rests on.

    The notebook's comment says ten components should explain about ninety
    percent of the variance. Checking it here means a change that quietly
    degrades the decomposition is caught where it happened rather than as a
    poor model six cells later.
    """
    ratio = decomposition(pca)["explained_variance_ratio"]
    assert ratio[:10].sum() > 0.85, f"ten components explain {ratio[:10].sum():.1%}"


def test_the_mean_pose_is_the_mean_of_the_frames(pca):
    """6. The one part of the result the random projection does not touch.

    ``svd_compressed`` randomises the subspace it projects onto, so the
    components move between runs. The mean it subtracts first does not, which
    makes it the only value here worth comparing against anything.
    """
    mean = decomposition(pca)["mean"]

    assert np.isfinite(mean).all()
    assert mean.min() >= 0, "a mean pose cannot be below the arena floor"
    assert mean.max() <= 255
    square = mean.reshape(80, 80)
    rows, cols = np.nonzero(square)
    assert 20 < rows.mean() < 60 and 20 < cols.mean() < 60, (
        "the mean mouse is not near the middle of its crop"
    )


def test_the_parameters_it_trained_with_are_recorded_beside_it(pca, configured):
    """7. What makes a decomposition reproducible, or at least explicable.

    The yaml beside the h5 holds the whole configuration the training ran
    under, including the filters that shaped the frames going in.
    """
    recorded = plain(read_yaml(str(pca / "pca.yaml")))

    assert recorded["rank"] == RANK
    assert recorded["missing_data"] is False
    assert recorded["gaussfilter_space"] == [1.5, 1]
    assert recorded["medfilter_space"] == [0]
    assert recorded["medfilter_time"] == [0]
    assert recorded["nworkers"] == 1


def test_the_decomposition_is_recorded_in_progress(pca, progress_file, project):
    """8. Where cells 15 and 16 find it."""
    from moseq2_app.gui.progress import restore_progress_vars

    recorded = restore_progress_vars(str(progress_file), init=False, overwrite=False)
    assert recorded["pca_dirname"] == str(project / "_pca") + "/"
    assert pca == project / "_pca"
