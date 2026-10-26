"""Cell 14: look at what the decomposition learned.

    from IPython.display import display, Image
    images = [join(progress_paths['pca_dirname'], 'pca_components.png'),
              join(progress_paths['pca_dirname'], 'pca_scree.png')]
    for im in images:
        display(Image(im))

The thinnest cell in the notebook, and the only one that imports ``display``
itself rather than calling into moseq2-app. It shows the two figures cell 13
wrote, so there is nothing of moseq2's to exercise here beyond whether those
figures are worth showing.

That is the test. The components plot is a grid of the learned basis, which
should look like mice; the scree plot is the variance curve a user reads to
pick ``npcs``. Both are written as png and pdf, and the cell names the pngs.
A figure that exists but is blank passes any check of the filenames, so the
images are decoded and their content looked at.
"""

from __future__ import annotations

import numpy as np
import pytest
from IPython.display import Image

#: The figures the cell names, in the order it shows them.
SHOWN = ("pca_components.png", "pca_scree.png")


@pytest.fixture
def figures(pca):
    """The two pngs the cell displays, decoded."""
    from PIL import Image as PILImage

    return {name: PILImage.open(pca / name).convert("L") for name in SHOWN}


def test_both_figures_are_where_the_cell_looks_for_them(pca):
    """1. The cell names the files by hand, so the names have to match."""
    for name in SHOWN:
        path = pca / name
        assert path.exists(), f"the cell would raise on {name}"
        assert Image(filename=str(path)).data, "the image is empty"


def test_the_figures_are_large_enough_to_read(figures):
    """2. A thumbnail of twenty-five components is not worth displaying."""
    for name, image in figures.items():
        width, height = image.size
        assert width >= 400 and height >= 300, f"{name} is {width}x{height}"


def test_the_figures_have_something_drawn_on_them(figures):
    """3. The failure this catches is a blank canvas.

    A plotting call that silently drew nothing still writes a valid png of
    the right size, and the cell still displays it. A user sees white and has
    no way to tell whether that is the figure or the bug.
    """
    for name, image in figures.items():
        pixels = np.asarray(image)
        ink = (pixels < 250).mean()
        assert ink > 0.01, f"{name} is {1 - ink:.1%} blank"
        assert pixels.std() > 5, f"{name} has no contrast"


def test_the_components_figure_is_a_grid_of_poses(figures):
    """4. What the components plot is for.

    Each panel is one component rendered back into the crop, so a user can
    see whether the basis learned mouse shapes or noise. It is wider than it
    is tall because the components are laid out in a grid.
    """
    width, height = figures["pca_components.png"].size
    assert width >= height, f"the components grid is {width}x{height}"


def test_the_scree_figure_is_the_curve_npcs_is_chosen_from(figures):
    """5. Why a user looks at the second figure.

    Cell 17 asks for ten components on the grounds that they explain most of
    the variance. This plot is where that is read off, so it has to carry
    ink across its width rather than collapsing into a corner.
    """
    pixels = np.asarray(figures["pca_scree.png"])
    columns_with_ink = (pixels < 250).any(axis=0)
    assert columns_with_ink.mean() > 0.3, "the curve does not span the plot"
