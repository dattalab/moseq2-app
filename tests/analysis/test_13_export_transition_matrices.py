"""Cell 13: write each group's transition matrix out as csv.

    for group_name, tm, usage in zip(group, trans_mats, usages):
        out = join(save_path, f'{group_name}_{normalize}_transition_matrix.csv')
        np.savetxt(out, tm, delimiter=',')
        out = join(save_path, f'{group_name}_syllable_counts.csv')
        np.savetxt(out, np.array(list(usage.items())), delimiter=',', header='syllable id,counts')

The last of the three exports, and the one most likely to be read by
something other than Python: a square grid of transition probabilities
per group, and the counts the graph scales its nodes by.

Written with ``np.savetxt`` rather than pandas, which has consequences a
reader meets straight away. There are no row or column labels on the
matrix, so position is the only thing identifying a syllable. The counts
file's header is written as a comment, the way numpy writes headers, so
it is not a csv header row. And every number goes out in full
double-precision exponential form, including the syllable ids, which are
integers.
"""

from __future__ import annotations

from os.path import join
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def exported(analysis_paths, transitions):
    """The cell, run for real, cleaned up afterwards."""
    save_path = analysis_paths["model_session_path"]
    written = []

    for group_name, tm, usage in zip(
        transitions["group"], transitions["trans_mats"], transitions["usages"]
    ):
        out = join(save_path, f"{group_name}_bigram_transition_matrix.csv")
        np.savetxt(out, tm, delimiter=",")
        written.append(Path(out))

        out = join(save_path, f"{group_name}_syllable_counts.csv")
        np.savetxt(
            out, np.array(list(usage.items())), delimiter=",", header="syllable id,counts"
        )
        written.append(Path(out))

    yield written
    for path in written:
        path.unlink(missing_ok=True)


def test_each_group_gets_a_matrix_and_a_count_file(exported, transitions, analysis_paths):
    """1. Two files per group, named after it.

    Both groups' exports land in one directory, so the group name in the
    filename is the only thing keeping them apart.
    """
    assert len(exported) == 2 * len(transitions["group"])
    for path in exported:
        assert path.exists(), f"{path.name} was not written"
        assert path.parent == Path(analysis_paths["model_session_path"])

    for group_name in transitions["group"]:
        names = [path.name for path in exported]
        assert f"{group_name}_bigram_transition_matrix.csv" in names
        assert f"{group_name}_syllable_counts.csv" in names


def test_the_matrix_comes_back_exactly(exported, transitions, labeled):
    """2. Full precision, so the export is the result rather than a view.

    ``np.savetxt`` defaults to eighteen significant digits, which is more
    than a float64 carries. A transition probability read back out of
    this file is the one that was computed.
    """
    for group_name, matrix in zip(transitions["group"], transitions["trans_mats"]):
        path = next(p for p in exported if p.name.startswith(f"{group_name}_bigram"))
        back = np.loadtxt(path, delimiter=",")

        assert back.shape == (labeled, labeled)
        np.testing.assert_array_equal(back, matrix)


def test_the_matrix_carries_no_labels_so_position_is_the_syllable(exported, labeled):
    """3. How a reader knows which syllable a row is.

    There is no header and no index column: row *i* is syllable *i*,
    numbered the way cell 9 numbered them. A reader who sorts the file or
    drops a row has silently renamed every syllable after it.
    """
    path = next(p for p in exported if "transition_matrix" in p.name)
    rows = path.read_text().splitlines()

    assert len(rows) == labeled, "the file has a header or a trailing row"
    assert not rows[0].startswith("#")
    assert len(rows[0].split(",")) == labeled


def test_the_counts_cover_every_syllable_being_analysed(exported, transitions, labeled):
    """4. One row per syllable, as id and count.

    These are the numbers cell 14 scales its graph nodes by, so a reader
    reproducing that figure outside the notebook needs all of them.
    """
    for group_name, usage in zip(transitions["group"], transitions["usages"]):
        path = next(p for p in exported if p.name == f"{group_name}_syllable_counts.csv")
        back = np.loadtxt(path, delimiter=",")

        assert back.ndim == 2 and back.shape[1] == 2, f"{path.name} is {back.shape}"
        ids = back[:, 0].astype(int)
        assert set(range(labeled)) <= set(ids), f"{path.name} is missing syllables"

        counts = dict(zip(ids, back[:, 1]))
        for syllable in range(labeled):
            assert counts[syllable] == usage[syllable]


def test_the_counts_header_is_a_comment_rather_than_a_csv_header(exported):
    """5. What the first line of the counts file actually is.

    The cell passes a header string, and ``np.savetxt`` writes headers
    commented. ``np.loadtxt`` skips it, which is the matching reader; a
    reader that does not treat ``#`` as a comment gets ``# syllable id``
    as a column name and the counts shifted under it.
    """
    path = next(p for p in exported if p.name.endswith("_syllable_counts.csv"))
    first = path.read_text().splitlines()[0]

    assert first.startswith("#"), f"the header is {first!r}"
    assert "syllable id" in first and "counts" in first


def test_the_counts_are_written_as_floats(exported):
    """6. Integers arrive in exponential notation.

    Syllable ids and bout counts are both whole numbers, and both go out
    through the same default format as the probabilities. A spreadsheet
    opening this shows ``0.000000000000000000e+00`` for syllable zero.
    """
    path = next(p for p in exported if p.name.endswith("_syllable_counts.csv"))
    body = [line for line in path.read_text().splitlines() if not line.startswith("#")]

    first, second = body[0].split(",")
    assert "e+" in first or "e-" in first, f"syllable id written as {first!r}"
    assert float(first) == 0.0
    assert float(second) == int(float(second)), "a bout count is not a whole number"
