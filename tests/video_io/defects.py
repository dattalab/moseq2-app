"""Defects in the video layer that 1.4.3 fixes.

A test here asserts what the function should return. Against an older
moseq2-extract it is expected to fail, and ``xfail_strict`` means it has to:
if one of these starts passing against a release that still carries the
defect, the assertion was not testing what it claims to.

The condition is what lets both sides be recorded at once. The suite runs
against moseq2-extract from PyPI, and a pull request in that repository lays
its own checkout over the release, so the same test file meets both the
defect and its fix.
"""

from __future__ import annotations

import moseq2_extract
import pytest
from packaging.version import Version

#: The release that fixes everything named in this module.
FIXED_IN = Version("1.4.3")

_installed = Version(moseq2_extract.__version__)


def fixed_in_1_4_3(reason):
    """Mark a test that fails against a moseq2-extract older than 1.4.3."""
    return pytest.mark.xfail(
        _installed < FIXED_IN,
        reason=f"moseq2-extract {_installed}: {reason}",
        strict=True,
    )


BYTE_ORDER = fixed_in_1_4_3(
    "an mkv depth track is 16-bit big-endian and is read at native byte "
    "order, so every sample comes back byte-swapped"
)

SEEK_RATE = fixed_in_1_4_3(
    "the seek divides the first frame index by the rate passed in, which "
    "defaults to 30, rather than by the recording's own rate"
)
