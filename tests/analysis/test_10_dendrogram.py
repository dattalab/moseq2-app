"""Cell 10: which syllables are versions of the same movement.

    from moseq2_app.main import show_dendrogram
    show_dendrogram(progress_paths, max_syllable=max_syllables, color_by_cluster=False)

The model's syllables are not independent. Several will be slow forward
walks that differ only in speed, and a few will be near-duplicates the
fitting split apart. This cell measures how alike their dynamics are and
draws the result as a tree, so a user can see which syllables sit
together before reading anything into a difference between them.

The distance is taken from the model itself rather than from the
recordings: each syllable is an autoregressive matrix saying how the
mouse's pose carries forward while it is in that state, and ``ar[init]``
compares those matrices. So the tree describes what the model learned,
not what any one animal did.

The leaves are named from ``syll_info.yaml``, which is where cell 9 put
the names a user typed. Unnamed syllables leave the tree labelled by
number alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moseq2_app.main import show_dendrogram

#: What ``plot_dendrogram`` writes into the plot directory.
FIGURE = "syllable_dendrogram"


@pytest.fixture(scope="module")
def dendrogram(analysis_paths, labeled):
    """Cell 10, run once. Returns the figure it drew and where it saved it."""
    import matplotlib.pyplot as plt

    plt.close("all")
    assert show_dendrogram(analysis_paths, max_syllable=labeled, color_by_cluster=False) is None
    figure = plt.gcf()
    yield figure, Path(analysis_paths["plot_path"])
    plt.close("all")


@pytest.fixture
def image(dendrogram):
    from PIL import Image as PILImage

    _, plots = dendrogram
    return PILImage.open(plots / f"{FIGURE}.png").convert("L")


def test_the_tree_is_saved_where_the_cell_points_it(dendrogram):
    """1. Into the model's own plot directory, as png and pdf."""
    _, plots = dendrogram
    for extension in ("png", "pdf"):
        path = plots / f"{FIGURE}.{extension}"
        assert path.exists(), f"{path.name} was not written"
        assert path.stat().st_size > 0, f"{path.name} is empty"


def test_the_tree_is_large_enough_to_read(image):
    """2. Sixty leaves with rotated labels needs the width."""
    width, height = image.size
    assert width >= 400 and height >= 300, f"the figure is {width}x{height}"


def test_the_tree_has_something_drawn_on_it(image):
    """3. The failure this catches is a blank canvas.

    A plotting call that silently drew nothing still writes a valid png
    of the right size, and the cell still shows it.
    """
    pixels = np.asarray(image)
    ink = (pixels < 250).mean()
    assert ink > 0.005, f"the figure is {1 - ink:.1%} blank"
    assert pixels.std() > 5, "the figure has no contrast"


def test_the_tree_spans_the_width_of_the_figure(image):
    """4. Every syllable is a leaf, so the leaves run edge to edge.

    A tree that collapsed into one corner would mean the distances came
    back degenerate, which is also what a broken comparison looks like.
    """
    pixels = np.asarray(image)
    columns_with_ink = (pixels < 250).any(axis=0)
    assert columns_with_ink.mean() > 0.5, "the tree occupies less than half the width"


def test_there_is_one_leaf_per_syllable_being_analysed(dendrogram, labeled):
    """5. The tree covers exactly the syllables the rest of the notebook does.

    Cell 9 settled on a count and cells 11, 12 and 14 all honour it. A
    tree drawn over a different number would be clustering syllables that
    appear nowhere else, or omitting ones that do.
    """
    figure, _ = dendrogram
    labels = [tick.get_text() for tick in figure.axes[0].get_xticklabels()]
    assert len(labels) == labeled, f"{len(labels)} leaves for {labeled} syllables"


def test_each_leaf_names_the_syllable_it_stands_for(dendrogram, analysis_paths, labeled):
    """6. A tree of unnamed branches is not readable.

    Leaves are built as the name a user typed in cell 9 followed by the
    syllable's number, so a syllable is identifiable whether or not it
    has been named yet.
    """
    from moseq2_viz.util import read_yaml

    figure, _ = dendrogram
    labels = [tick.get_text() for tick in figure.axes[0].get_xticklabels()]
    syll_info = read_yaml(analysis_paths["syll_info"])

    for text in labels:
        name, _, number = text.rpartition("-")
        assert number.isdigit(), f"{text!r} does not end in a syllable number"
        assert int(number) < labeled
        assert name == syll_info[int(number)]["label"]

    assert len({text.rpartition("-")[2] for text in labels}) == labeled, "a syllable appears twice"


def test_the_distances_are_a_symmetric_comparison_of_every_pair(analysis_paths, labeled):
    """7. What the tree is built from.

    ``ar[init]`` compares the autoregressive matrix of each syllable
    against every other, giving a square matrix of distances. The tree is
    only as meaningful as this: it has to be symmetric, zero down the
    diagonal and positive elsewhere, or the clustering is reading noise.
    """
    from moseq2_viz.model.dist import get_behavioral_distance
    from moseq2_viz.util import get_sorted_index

    index = get_sorted_index(analysis_paths["index_file"])
    distances = get_behavioral_distance(
        index, analysis_paths["model_path"], max_syllable=labeled, distances="ar[init]"
    )["ar[init]"]

    assert distances.shape == (labeled, labeled)
    np.testing.assert_allclose(distances, distances.T, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(np.diag(distances), 0.0, atol=1e-9)

    off_diagonal = distances[~np.eye(labeled, dtype=bool)]
    assert (off_diagonal > 0).all(), "two syllables are indistinguishable"
    assert np.isfinite(distances).all()


def test_a_label_file_naming_more_syllables_than_the_tree_still_draws(analysis_paths, labeled, tmp_path):
    """8. The tree takes the names of the syllables it draws.

    Cell 9 keeps every name a user typed, including names for syllables
    beyond the count it was last opened at, so the label file can hold
    more entries than the tree has leaves.
    """
    import matplotlib.pyplot as plt

    from moseq2_app.util import write_yaml
    from moseq2_viz.util import read_yaml

    syll_info = read_yaml(analysis_paths["syll_info"])
    for syllable in syll_info:
        syll_info[syllable]["label"] = f"name {syllable}"
    info = tmp_path / "syll_info.yaml"
    write_yaml(syll_info, str(info))
    fewer = labeled - 4
    paths = {**analysis_paths, "syll_info": str(info), "plot_path": str(tmp_path / "plots")}

    plt.close("all")
    show_dendrogram(paths, max_syllable=fewer, color_by_cluster=False)
    labels = [tick.get_text() for tick in plt.gcf().axes[0].get_xticklabels()]
    plt.close("all")

    assert sorted(labels) == sorted(f"name {syllable}-{syllable}" for syllable in range(fewer))
