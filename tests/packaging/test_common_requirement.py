"""Every MoSeq2 package requires moseq2-common, at exactly its own version.

``moseq2-common`` holds the code the other five share, starting with the YAML
helpers. It releases with them from one tag, so each requires it at exactly
the version it is itself. Its base install needs only numpy, ruamel.yaml and
h5py (the video libraries come with its ``video`` extra), and requiring it must
not make a core package pull in a sibling: ``python -m pip install
moseq2-model`` installs moseq2-model and moseq2-common, nothing else from
MoSeq2.

Read from the installed metadata, like ``test_dependency_floors.py``, so the
tests check what the packages ship.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from importlib.metadata import distribution, version
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO = Path(__file__).resolve().parents[2]
CORE = ("moseq2-extract", "moseq2-pca", "moseq2-model", "moseq2-viz")


def moseq2_requirements(package):
    """name -> Requirement, for the moseq2-* requirements ``package`` declares."""
    found = {}
    for raw in distribution(package).requires or []:
        req = Requirement(raw)
        name = canonicalize_name(req.name)
        if name.startswith("moseq2-"):
            found[name] = req
    return found


def test_moseq2_common_base_requirements():
    reqs = [Requirement(r) for r in distribution("moseq2-common").requires or []]
    names = {req.name for req in reqs if req.marker is None}
    assert names == {"numpy", "ruamel.yaml", "h5py"}


def test_the_video_extra_holds_the_video_libraries():
    reqs = [Requirement(r) for r in distribution("moseq2-common").requires or []]
    video = {
        req.name for req in reqs
        if req.marker is not None and req.marker.evaluate({"extra": "video"})
    }
    assert video == {"av", "matplotlib", "opencv-python-headless", "tqdm"}


@pytest.mark.parametrize("package", CORE)
def test_extract_pca_and_viz_require_the_video_extra(package):
    req = moseq2_requirements(package)["moseq2-common"]
    expected = set() if package == "moseq2-model" else {"video"}
    assert req.extras == expected


@pytest.mark.parametrize("package", ("moseq2-app",) + CORE)
def test_requires_moseq2_common_at_its_own_version(package):
    req = moseq2_requirements(package).get("moseq2-common")
    assert req is not None, f"{package} does not require moseq2-common"
    assert str(req.specifier) == f"=={version(package)}"
    assert version("moseq2-common") == version(package)


@pytest.mark.parametrize("package", CORE)
def test_a_core_package_pulls_in_no_sibling(package):
    assert set(moseq2_requirements(package)) == {"moseq2-common"}


@pytest.mark.network
@pytest.mark.parametrize("sibling", [None, *CORE])
def test_moseq2_common_installs_alone_and_with_each_sibling(sibling, tmp_path):
    packages = REPO / "packages"
    if not packages.is_dir():
        pytest.skip(f"{packages} is missing, so this is not a checkout (an unpacked sdist?)")
    if importlib.util.find_spec("pip") is None:
        pytest.skip(f"pip is not installed in {sys.executable}")
    targets = [str(packages / "moseq2-common")]
    if sibling is not None:
        targets.append(str(packages / sibling))
    report = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
            "--quiet", "--report", str(report), *targets,
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr
    installed = {
        canonicalize_name(item["metadata"]["name"])
        for item in json.loads(report.read_text())["install"]
    }
    moseq2 = {name for name in installed if name.startswith("moseq2-")}
    assert moseq2 == {"moseq2-common"} | ({sibling} if sibling else set())
