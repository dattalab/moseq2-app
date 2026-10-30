"""Naming the two groups a pairwise test compares.

``run_pairwise_stats`` runs the z-test, t-test or Mann-Whitney between two
groups named by the caller. The syllable-stats cell fills its group menus
from the table, but a caller of the Python API types the names, and a name
the table does not have is an error rather than a comparison of nothing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from moseq2_viz.model.stat import run_pairwise_stats


def table(groups):
    """A per-session syllable table from ``{group: [each session's usage]}``, two syllables."""
    return pd.DataFrame(
        [
            {"group": g, "uuid": f"{g}{i}", "syllable": s, "usage": usage + 0.001 * s}
            for g, usages in groups.items()
            for i, usage in enumerate(usages)
            for s in range(2)
        ]
    )


@pytest.mark.parametrize("test_type", ["mw", "z_test", "t_test"])
def test_a_group_the_table_does_not_have_is_named_in_the_error(test_type):
    """'Amph' where the table says 'amph': every test stops and lists the groups there are."""
    df = table({"amph": [0.10, 0.12, 0.14], "saline": [0.30, 0.31, 0.32]})

    with pytest.raises(ValueError, match=r"^Group 'Amph' is not in the data\. Its groups are: amph, saline\.$"):
        run_pairwise_stats(df, "saline", "Amph", test_type=test_type, max_syllable=2)
