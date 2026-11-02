"""Every dependency the six packages declare has a lower bound.

The floors are the oldest versions the CI suite has passed with. A
dependency with no floor lets someone installing into an existing
environment keep any version at all, including ones nothing here has ever
run with, so every requirement a user's install reads names one.

Read from the installed metadata, so the test checks what the packages
actually ship. The optional ``tests`` extra is checked too, since the suite
installs it; extras CI never installs (``docs``, ``graphviz``) have no tested
version to set a floor from.
"""

from __future__ import annotations

from importlib.metadata import distribution

import pytest
from packaging.requirements import Requirement

PACKAGES = (
    "moseq2-app",
    "moseq2-extract",
    "moseq2-pca",
    "moseq2-model",
    "moseq2-viz",
    "moseq2-common",
)

#: Operators that put a lower bound on the version.
LOWER_BOUND = {">=", "==", "~=", "==="}


def declared():
    """(package, requirement) for each requirement an install of the suite reads."""
    found = []
    for package in PACKAGES:
        for raw in distribution(package).requires or []:
            req = Requirement(raw)
            if req.marker is None or req.marker.evaluate({"extra": "tests"}):
                found.append((package, req))
    return found


@pytest.mark.parametrize(
    "package, requirement", declared(), ids=lambda v: str(v) if isinstance(v, str) else v.name
)
def test_every_dependency_has_a_lower_bound(package, requirement):
    operators = {spec.operator for spec in requirement.specifier}
    assert operators & LOWER_BOUND, (
        f"{package} requires {requirement.name!r} as {str(requirement)!r}, with no lower "
        f"bound. Give it a floor: the oldest version the CI suite has passed with, the "
        f"lowest {requirement.name} any suite job's 'Record the resolution' step lists in "
        f"the latest successful nightly run of the tests workflow."
    )
